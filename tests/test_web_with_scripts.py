"""The web app with scripts registered, i.e. the demo wired in.

`test_web.py` runs with nothing registered, which is what a fresh install looks like.
Here the demo registry is loaded, so adding data and running an evaluation go all the
way through an Ingestor, a Runner and an Evaluator.
"""

from __future__ import annotations

import sys

import pytest

from cairn import registry
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.storage import Storage
from cairn.demo import pipeline
from cairn.web.app import create_app

try:
    from fastapi.testclient import TestClient
except RuntimeError as exc:  # starlette needs httpx2, which is not a declared dev dependency
    pytest.skip(f"fastapi's TestClient is unavailable: {exc}", allow_module_level=True)

DATASET = "sensor-anomaly-A"
TABLE = "anomaly"


@pytest.fixture
def env(tmp_path, monkeypatch) -> Storage:
    """The demo, seeded, with its registry as the one the app loads."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setenv("CAIRN_ROOT", f"file://{tmp_path}/cairn")
    monkeypatch.setenv("CAIRN_SCRIPTS", str(scripts))
    monkeypatch.setenv("CAIRN_REGISTRY", "cairn.demo.registry")
    monkeypatch.setenv("CAIRN_USER", "tester")
    monkeypatch.setenv("CAIRN_LANG", "en")
    for store in (registry._PROBLEMS, registry._INGESTORS, registry._EVALUATORS,
                  registry._INFERENCE, registry._SCANNED, registry._LOADED):
        store.clear()
    # The registry module registers on import, and an earlier test may have imported it.
    sys.modules.pop("cairn.demo.registry", None)

    st = Storage(f"file://{tmp_path}/cairn")
    pipeline.seed(st)
    return st


@pytest.fixture
def client(env) -> TestClient:
    return TestClient(create_app())


# --- adding data through the ingest script ----------------------------------

def test_the_ingest_form_is_generated_from_the_scripts_input(client):
    body = client.get(f"/datasets/{DATASET}/ingest").text
    assert "Script" in body and "JSONL" in body
    for field in ("n", "seed", "site", "anomaly_rate"):
        assert f'name="{field}"' in body


def test_the_script_previews_rows_and_then_appends_them(client, env):
    before = ds.count_rows(env, DATASET)
    # Site C, so the generated ids are new rows rather than upserts of the seeded ones.
    form = {"mode": "script", "n": "3", "seed": "5", "site": "C", "anomaly_rate": "0.5"}

    preview = client.post(f"/datasets/{DATASET}/ingest/preview", data=form)
    assert preview.status_code == 200
    assert "Add 3 rows" in preview.text

    confirm = client.post(f"/datasets/{DATASET}/ingest", data=form, follow_redirects=False)
    assert confirm.status_code == 303
    assert ds.count_rows(env, DATASET) == before + 3


# --- running an evaluation --------------------------------------------------

def test_the_run_form_offers_the_config_of_the_run_script(client):
    body = client.get(f"/evals/{TABLE}/run").text
    assert 'name="threshold"' in body and 'name="w_vib"' in body
    assert "<option>v1</option><option>v2</option>" in body   # the versions it found


def _run_form(**over) -> dict[str, str]:
    return {"title": "from the web", "dataset": DATASET, "evaluator": "v2",
            "threshold": "0.6", "w_temp": "0.02", "w_vib": "1.2", "w_pressure": "0.05", **over}


def test_the_form_issues_a_run_even_though_a_script_is_registered(client, env):
    """A registered script does not make the web run it: the page only issues the run."""
    before = evals.count_runs(env, TABLE)

    r = client.post(f"/evals/{TABLE}/run", data=_run_form(), follow_redirects=False)

    assert r.status_code == 303
    eval_id = r.headers["location"].split("started=")[1]
    assert evals.count_runs(env, TABLE) == before           # nothing scored
    assert evals.count_predictions(env, TABLE, eval_id) == 0
    meta = evals.get_meta(env, TABLE, eval_id)
    assert meta["title"] == "from the web"
    assert meta["evaluator_version"] == "v2"
    assert meta["config"]["threshold"] == 0.6               # the form value, coerced to float


@pytest.mark.parametrize("form, detail", [
    ({"dataset": "nope"}, "dataset not found"),
    ({"title": "  "}, "Title is required"),
])
def test_a_bad_run_request_is_refused_before_anything_is_created(client, env, form, detail):
    before = evals.count_runs(env, TABLE)

    r = client.post(f"/evals/{TABLE}/run", data=_run_form(**form))

    assert r.status_code == 400 and detail in r.text
    assert evals.count_runs(env, TABLE) == before


def test_a_table_with_no_registered_script_only_issues_the_run(client, env):
    """Same request, different table: with no script there is nothing here to perform it."""
    evals.create_eval_table(env, "name: unwired\ncolumns:\n  - {name: score, type: float}\n")

    r = client.post("/evals/unwired/run", data=_run_form(), follow_redirects=False)

    assert r.status_code == 303 and "started=" in r.headers["location"]
    assert evals.count_runs(env, "unwired") == 0


def test_the_help_page_reports_no_problems_for_the_demo(client):
    assert "No problems found" in client.get("/help").text


# --- runs whose inference happens elsewhere ---------------------------------

def _started(env, title: str = "for the pipeline") -> str:
    """A run started the way the pipeline gets one: nothing has performed it yet."""
    return evals.create_run(env, TABLE, dataset=DATASET, evaluator_version="v2", title=title)


def test_a_run_the_pipeline_has_not_performed_yet_is_listed_as_running(client, env):
    """Whether the inference is done is known where it runs; the button is always offered."""
    eval_id = _started(env)

    page = client.get(f"/evals/{TABLE}?started={eval_id}").text

    assert "Running" in page and eval_id in page
    assert "Evaluate" in page


def test_a_run_scored_by_something_else_says_so_instead(client, env):
    """Predictions are in, but nothing here knows how to score that eval version."""
    eval_id = evals.create_run(env, TABLE, dataset=DATASET, evaluator_version="v9", title="v9 run")
    evals.write_prediction(env, TABLE, eval_id, "A_000", b"{}")

    page = client.get(f"/evals/{TABLE}").text

    assert "no evaluator for v9 here" in page


def test_a_running_run_is_evaluated_from_the_page_once_its_predictions_are_in(client, env):
    eval_id = _started(env, "scored later")
    for sample_id in evals.target_ids(env, TABLE, eval_id):
        evals.write_prediction(env, TABLE, eval_id, sample_id, b'{"pred": 1}')

    scored = client.post(f"/evals/{TABLE}/runs/{eval_id}/score", follow_redirects=False)

    assert scored.status_code == 303
    assert evals.pending_runs(env, TABLE) == []
    assert evals.get_run(env, TABLE, eval_id)["row"]["title"] == "scored later"


def test_evaluating_a_run_nothing_wrote_to_records_exactly_that(client, env):
    """Evaluating is a decision, not a check: what it scores is whatever arrived."""
    eval_id = _started(env, "evaluated too early")

    r = client.post(f"/evals/{TABLE}/runs/{eval_id}/score", follow_redirects=False)

    assert r.status_code == 303
    assert evals.pending_runs(env, TABLE) == []          # it has a result now, empty as it is
    assert evals.get_run(env, TABLE, eval_id)["row"]["f1"] == 0.0


def test_evaluating_needs_a_run_and_an_evaluator_for_its_version(client, env):
    assert client.post(f"/evals/{TABLE}/runs/e_00000000000000000000000000/score").status_code == 404

    eval_id = evals.create_run(env, TABLE, dataset=DATASET, evaluator_version="v9",
                               title="scored by something else")
    r = client.post(f"/evals/{TABLE}/runs/{eval_id}/score")
    assert r.status_code == 400 and "No evaluator registered for v9" in r.text
