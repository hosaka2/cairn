"""A table's `pipeline.py`: starting a run hands it over, evaluating gathers what came back.

The pipeline here stands in for an orchestrator: it notes what it was handed in a file
beside itself, and has finished once a `done` file is there.
"""

from __future__ import annotations

import json
import sys

import pytest
from typer.testing import CliRunner

from cairn import contract, registry, scripts
from cairn.cli.main import app
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.storage import Storage
from cairn.demo import pipeline as demo
from cairn.interfaces.pipeline import NotReady
from cairn.runner import collect_run, submit_run
from cairn.web.app import create_app

try:
    from fastapi.testclient import TestClient
except RuntimeError as exc:  # starlette needs httpx2, which is not a declared dev dependency
    pytest.skip(f"fastapi's TestClient is unavailable: {exc}", allow_module_level=True)

DATASET = "sensor-anomaly-A"
TABLE = "anomaly"

PIPELINE = '''
import json
from pathlib import Path

from cairn.interfaces.pipeline import NotReady

HERE = Path(__file__).parent


class Orchestrator:
    def submit(self, eval_id, sample_ids, config):
        if (HERE / "unreachable").exists():
            raise RuntimeError("the orchestrator is unreachable")
        (HERE / "submitted.json").write_text(json.dumps({"eval_id": eval_id, "n": len(sample_ids),
                                                         "config": config}))

    def collect(self, eval_id, sample_ids, config):
        if not (HERE / "done").exists():
            raise NotReady("3 of 5 jobs still running")
        return [(sid, b'{"pred": 1}') for sid in sample_ids]


PIPELINE = Orchestrator
'''

SUBMIT_ONLY = '''
class Orchestrator:
    def submit(self, eval_id, sample_ids, config):
        pass


PIPELINE = Orchestrator
'''


def _clear_registry() -> None:
    for store in (registry._PROBLEMS, registry._INGESTORS, registry._EVALUATORS, registry._INFERENCE,
                  registry._PIPELINES, registry._SCANNED, registry._LOADED, registry._LAZY):
        store.clear()


@pytest.fixture
def env(tmp_path, monkeypatch) -> Storage:
    """The demo, seeded, with a pipeline.py for its table."""
    here = tmp_path / "scripts" / "evals" / TABLE
    here.mkdir(parents=True)
    (here / "pipeline.py").write_text(PIPELINE)
    monkeypatch.setenv("CAIRN_ROOT", f"file://{tmp_path}/cairn")
    monkeypatch.setenv("CAIRN_SCRIPTS", str(tmp_path / "scripts"))
    monkeypatch.setenv("CAIRN_REGISTRY", "cairn.demo.registry")
    monkeypatch.setenv("CAIRN_USER", "tester")
    monkeypatch.setenv("CAIRN_LANG", "en")
    _clear_registry()
    sys.modules.pop("cairn.demo.registry", None)
    st = Storage(f"file://{tmp_path}/cairn")
    demo.seed(st)
    return st


@pytest.fixture
def here(tmp_path):
    return tmp_path / "scripts" / "evals" / TABLE


@pytest.fixture
def client(env) -> TestClient:
    return TestClient(create_app())


def _run_form(**over: str) -> dict[str, str]:
    form = {"title": "handed over", "dataset": DATASET, "evaluator": "v2", "config": "{}"}
    form.update(over)
    return form


# --- the web ------------------------------------------------------------------

def test_starting_a_run_hands_it_to_the_pipeline(client, env, here):
    r = client.post(f"/evals/{TABLE}/run", data=_run_form(), follow_redirects=False)

    assert r.status_code == 303
    eval_id = r.headers["location"].split("started=")[1]
    handed = json.loads((here / "submitted.json").read_text())
    assert handed["eval_id"] == eval_id
    assert handed["n"] == len(evals.target_ids(env, TABLE, eval_id))
    assert "Handed to the pipeline when started" in client.get(f"/evals/{TABLE}").text


def test_a_run_the_pipeline_refused_is_withdrawn_not_left_running(client, env, here):
    (here / "unreachable").touch()

    r = client.post(f"/evals/{TABLE}/run", data=_run_form())

    assert r.status_code == 502 and "the orchestrator is unreachable" in r.text
    assert evals.pending_runs(env, TABLE) == []


def test_evaluating_before_the_pipeline_has_finished_scores_nothing(client, env):
    eval_id = evals.create_run(env, TABLE, dataset=DATASET, evaluator_version="v2", title="early")

    r = client.post(f"/evals/{TABLE}/runs/{eval_id}/score")

    assert r.status_code == 409 and "3 of 5 jobs still running" in r.text
    assert [p["eval_id"] for p in evals.pending_runs(env, TABLE)] == [eval_id]
    assert evals.count_predictions(env, TABLE, eval_id) == 0


def test_evaluating_gathers_the_predictions_and_scores_them(client, env, here):
    eval_id = evals.create_run(env, TABLE, dataset=DATASET, evaluator_version="v2", title="finished")
    (here / "done").touch()

    r = client.post(f"/evals/{TABLE}/runs/{eval_id}/score", follow_redirects=False)

    assert r.status_code == 303
    assert evals.count_predictions(env, TABLE, eval_id) == len(evals.target_ids(env, TABLE, eval_id))
    assert evals.get_run(env, TABLE, eval_id)["row"]["title"] == "finished"


# --- gathering ----------------------------------------------------------------

def test_a_pipeline_that_does_not_gather_leaves_the_predictions_alone(env, here):
    (here / "pipeline.py").write_text(SUBMIT_ONLY)
    eval_id = evals.create_run(env, TABLE, dataset=DATASET, evaluator_version="v2", title="self-writing")
    pipeline = registry.load_pipeline(TABLE)

    submit_run(env, TABLE, eval_id, pipeline)
    assert collect_run(env, TABLE, eval_id, pipeline) is None
    assert evals.count_predictions(env, TABLE, eval_id) == 0


def test_not_ready_writes_nothing(env):
    eval_id = evals.create_run(env, TABLE, dataset=DATASET, evaluator_version="v2", title="running")
    with pytest.raises(NotReady):
        collect_run(env, TABLE, eval_id, registry.load_pipeline(TABLE))
    assert evals.count_predictions(env, TABLE, eval_id) == 0


# --- the scripts --------------------------------------------------------------

def test_the_scan_registers_a_pipeline_and_names_one_that_is_not_there(env, here, tmp_path):
    registry.ensure_loaded()
    assert registry.get_pipeline(TABLE) is not None
    assert scripts.has_pipeline(TABLE)
    assert not scripts.has_pipeline("unwired")

    broken = tmp_path / "scripts" / "evals" / "broken"
    broken.mkdir()
    (broken / "pipeline.py").write_text("x = 1\n")
    assert any("`PIPELINE` is missing" in p["what"] for p in scripts.problems())


@pytest.mark.parametrize("obj, error", [
    (object(), "Assign a class to PIPELINE"),
    (type("NoSubmit", (), {}), "`submit` method is missing"),
    (type("BadCollect", (), {"submit": lambda self, a, b, c: None, "collect": 1}), "`collect` is not callable"),
])
def test_the_contract_names_what_a_pipeline_lacks(obj, error):
    assert any(error in e for e in contract.check_export("PIPELINE", obj))


def test_a_pipeline_with_submit_alone_honours_the_contract():
    assert contract.check_export("PIPELINE", type("P", (), {"submit": lambda self, a, b, c: None})) == []


# --- the command line ---------------------------------------------------------

def _cli(*args: str):
    return CliRunner().invoke(app, list(args))


def test_create_run_hands_the_run_over_unless_told_not_to(env, here):
    made = _cli("eval", "create-run", TABLE, "--dataset", DATASET, "--evaluator-version", "v2",
                "--title", "from the shell", "--config", '{"weights": "w1"}')
    assert made.exit_code == 0, made.output
    assert json.loads((here / "submitted.json").read_text())["config"] == {"weights": "w1"}

    (here / "submitted.json").unlink()
    kept = _cli("eval", "create-run", TABLE, "--dataset", DATASET, "--evaluator-version", "v2",
                "--title", "kept here", "--no-submit")
    assert kept.exit_code == 0, kept.output
    assert not (here / "submitted.json").exists()


def test_score_gathers_first_and_waits_for_a_pipeline_that_has_not_finished(env, here):
    eval_id = evals.create_run(env, TABLE, dataset=DATASET, evaluator_version="v2", title="shell")

    early = _cli("eval", "score", TABLE, eval_id, "--evaluator", "cairn.demo.evaluate:AnomalyEvalV2")
    assert early.exit_code != 0 and "has not finished" in early.output

    (here / "done").touch()
    scored = _cli("eval", "score", TABLE, eval_id, "--evaluator", "cairn.demo.evaluate:AnomalyEvalV2")
    assert scored.exit_code == 0, scored.output
    assert evals.count_predictions(env, TABLE, eval_id) == len(ds.read_rows(
        env, DATASET, evals.get_snapshot(env, TABLE, eval_id)["keys"]))
