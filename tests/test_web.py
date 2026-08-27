"""The web layer: every page renders, unknown names 404, and the write flows work.

The routes are thin wrappers over `cairn.core`, so the assertions here are about the
HTTP surface — status codes, rendered fragments, redirects, headers — and about what
ends up in storage afterwards, read back through `cairn.core`.
"""

from __future__ import annotations

import json

import pytest

from cairn import registry
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.records import EvalResult, Metric
from cairn.core.storage import Storage
from cairn.web.app import create_app

try:
    from fastapi.testclient import TestClient
except RuntimeError as exc:  # starlette needs httpx2, which is not a declared dev dependency
    pytest.skip(f"fastapi's TestClient is unavailable: {exc}", allow_module_level=True)

DATASET_SCHEMA = """
name: clips
kind: tabular
description: five clips to page through
key: id
columns:
  - {name: id, type: str, required: true}
  - {name: value, type: float}
  - {name: label, type: int, required: true}
"""

TABLE_SCHEMA = """
name: clip-acc
description: clip-acc accuracy
columns:
  - {name: accuracy, type: float, display: "accuracy%", primary: true, direction: higher, scale: [0, 1]}
  - {name: coverage, type: float}
"""

# A one-pixel PNG, so the asset route has something with a real image type to serve.
PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
                    "890000000a49444154789c6300010000050001")

REPORT_MD = "# report\n\n![plot](assets/plot.png)\n"


class _StubEval:
    """An evaluator that returns a fixed row, so seeded numbers are exact."""

    def __init__(self, accuracy: float, coverage: float = 1.0, *, with_assets: bool = False) -> None:
        self._row = {"accuracy": accuracy, "coverage": coverage}
        self._with_assets = with_assets

    def score(self, ctx) -> EvalResult:
        assets = {"plot.png": PNG, "notes.txt": b"<script>alert(1)</script>"} if self._with_assets else {}
        return EvalResult(
            row=dict(self._row),
            metrics=[Metric(name="accuracy", value=self._row["accuracy"])],
            report_md=REPORT_MD if self._with_assets else "",
            assets=assets,
            metadata={"note": "from the evaluator"},
        )


def _pin_created_at(st: Storage, table: str, eval_id: str, when: str) -> None:
    """Give a run an explicit timestamp, so listings never depend on the wall clock."""
    for rel in (f"evals/{table}/runs/{eval_id}/_meta.json",
                f"evals/{table}/runs/{eval_id}/result/row.json"):
        doc = st.read_json(rel)
        doc["created_at"] = when
        st.write_json(rel, doc)


def _seed_run(st: Storage, *, title: str, version: str, accuracy: float, when: str,
              with_assets: bool = False) -> str:
    eval_id = evals.create_run(st, "clip-acc", dataset="clips", evaluator_version=version,
                               title=title, comment=f"note for {title}", created_by="tester",
                               config={"threshold": 0.5})
    evals.write_prediction(st, "clip-acc", eval_id, "r0", b"{}")
    evals.score_run(st, "clip-acc", eval_id, _StubEval(accuracy, with_assets=with_assets))
    _pin_created_at(st, "clip-acc", eval_id, when)
    return eval_id


@pytest.fixture
def env(tmp_path, monkeypatch) -> Storage:
    """A fresh storage root, no user scripts, English prose."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setenv("CAIRN_ROOT", f"file://{tmp_path}/cairn")
    monkeypatch.setenv("CAIRN_SCRIPTS", str(scripts))
    monkeypatch.setenv("CAIRN_REGISTRY", "")
    monkeypatch.setenv("CAIRN_USER", "tester")
    monkeypatch.setenv("CAIRN_LANG", "en")
    registry._PROBLEMS.clear()
    registry._INGESTORS.clear()
    registry._EVALUATORS.clear()
    registry._INFERENCE.clear()
    registry._SCANNED.clear()
    return Storage(f"file://{tmp_path}/cairn")


@pytest.fixture
def client(env) -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def seeded(env) -> dict[str, str]:
    """One dataset of five rows and one eval table with three scored runs.

    Two runs are scored with v2 (the current version) and one with v1, which is what
    makes "not comparable across eval versions" observable.
    """
    ds.create_dataset(env, DATASET_SCHEMA, created_by="tester")
    ds.append(env, "clips", [{"id": f"r{i}", "value": i / 10, "label": i} for i in range(5)],
              created_by="tester", input={"jsonl": "seed.jsonl"})
    evals.create_eval_table(env, TABLE_SCHEMA)
    return {
        "old": _seed_run(env, title="old run", version="v1", accuracy=0.5,
                         when="2026-01-01T09:00:00+00:00"),
        "base": _seed_run(env, title="baseline run", version="v2", accuracy=0.7,
                          when="2026-01-02T09:00:00+00:00"),
        "top": _seed_run(env, title="top run", version="v2", accuracy=0.9,
                         when="2026-01-03T09:00:00+00:00", with_assets=True),
    }


# --- every page renders -----------------------------------------------------

PAGES = [
    ("/", "Evals"),
    ("/datasets", "clips"),
    ("/datasets/clips", "Ingest history"),
    ("/datasets/new", "Create dataset"),
    ("/datasets/clips/ingest", "Add data"),
    ("/evals", "Primary metric"),
    ("/evals/clip-acc", "top run"),
    ("/evals/new", "Create eval table"),
    ("/evals/clip-acc/run", "New run"),
    ("/help", "How to use"),
    ("/help/manual", "How to use"),
    ("/help/scripting", "Writing scripts"),
]


@pytest.mark.parametrize(("path", "fragment"), PAGES)
def test_page_renders_with_real_data(client, seeded, path, fragment):
    r = client.get(path)
    assert r.status_code == 200
    assert fragment in r.text


def test_dataset_page_shows_rows_and_ingest_history(client, seeded):
    body = client.get("/datasets/clips").text
    assert "5 rows" in body
    assert ">r0<" in body and ">r4<" in body
    assert "+5" in body                      # the single ingest, as a stone


def test_eval_list_shows_the_primary_metric_and_its_delta(client, seeded):
    body = client.get("/evals").text
    assert "clip-acc" in body
    assert "0.900" in body                   # the latest run's primary metric
    assert "+0.200" in body                  # against the previous run scored the same way
    assert "v2: 2" in body                   # two of the three runs use the current version


def test_eval_detail_shows_the_report_with_asset_links_rewritten(client, seeded):
    body = client.get(f"/evals/clip-acc?run={seeded['top']}").text
    assert f'src="/evals/clip-acc/runs/{seeded["top"]}/assets/plot.png"' in body
    assert "from the evaluator" in body      # metadata recorded by the evaluator
    assert "note for top run" in body        # the run comment


# --- empty states -----------------------------------------------------------

def test_dataset_list_is_empty_without_datasets(client):
    r = client.get("/datasets")
    assert r.status_code == 200
    assert "No datasets yet" in r.text


def test_eval_list_is_empty_without_tables(client):
    r = client.get("/evals")
    assert r.status_code == 200
    assert "No eval tables yet" in r.text


def test_dataset_without_rows_says_so(client, env):
    ds.create_dataset(env, DATASET_SCHEMA, created_by="tester")
    r = client.get("/datasets/clips")
    assert r.status_code == 200
    assert "No rows yet" in r.text


def test_eval_table_without_a_scored_run_says_so(client, env):
    evals.create_eval_table(env, TABLE_SCHEMA)
    r = client.get("/evals/clip-acc")
    assert r.status_code == 200
    assert "No scored runs yet" in r.text


# --- unknown names ----------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/datasets/nope",
    "/datasets/nope/ingest",
    "/evals/nope",
    "/evals/nope/run",
    "/help/nope",
    "/evals/clip-acc/runs/e_nope/assets/plot.png",
])
def test_unknown_name_returns_404(client, seeded, path):
    assert client.get(path).status_code == 404


def test_unknown_asset_of_a_real_run_returns_404(client, seeded):
    assert client.get(f"/evals/clip-acc/runs/{seeded['top']}/assets/missing.png").status_code == 404


# --- pagination -------------------------------------------------------------

def test_offset_and_page_size_select_a_slice(client, seeded):
    body = client.get("/datasets/clips?offset=2&page_size=2").text
    assert "3–4 / 5 rows" in body
    assert ">r2<" in body and ">r3<" in body
    assert ">r0<" not in body and ">r4<" not in body


def test_page_size_is_clamped(client, seeded):
    assert "1–1 / 5 rows" in client.get("/datasets/clips?page_size=-5").text
    assert "1–5 / 5 rows" in client.get("/datasets/clips?page_size=99999999").text


def test_negative_offset_starts_at_the_beginning(client, seeded):
    body = client.get("/datasets/clips?offset=-10").text
    assert "1–5 / 5 rows" in body
    assert "Prev" not in body


# --- write flows ------------------------------------------------------------

NEW_SCHEMA = """name: pins
key: pin_id
columns:
  - {name: pin_id, type: str, required: true}
  - {name: score, type: float}
"""

NEW_TABLE = """name: coverage
columns:
  - {name: score, type: float, primary: true}
"""


def test_posting_a_schema_creates_the_dataset(client, env):
    r = client.post("/datasets/new", data={"yaml": NEW_SCHEMA})
    assert r.status_code == 200
    assert str(r.url).endswith("/datasets/pins")
    assert ds.list_datasets(env) == ["pins"]
    assert ds.load_schema(env, "pins").key == "pin_id"


def test_posting_an_invalid_schema_re_renders_the_form(client, env):
    r = client.post("/datasets/new", data={"yaml": "name: broken\nkey: id\ncolumns: []\n"})
    assert r.status_code == 200
    assert "Error" in r.text
    assert "name: broken" in r.text           # what was typed comes back
    assert ds.list_datasets(env) == []


def test_posting_a_table_creates_the_eval_table(client, env):
    client.post("/datasets/new", data={"yaml": NEW_SCHEMA})
    r = client.post("/evals/new", data={"yaml": NEW_TABLE, "dataset": "pins"})
    assert r.status_code == 200
    assert str(r.url).endswith("/evals/coverage")
    assert evals.load_eval_table(env, "coverage").dataset == "pins"


def test_selected_dataset_wins_over_the_pasted_table_yaml(client, env):
    client.post("/datasets/new", data={"yaml": NEW_SCHEMA})
    r = client.post("/evals/new", data={"yaml": NEW_TABLE + "dataset: written-in-yaml\n",
                                        "dataset": "pins"})
    assert r.status_code == 200
    assert evals.load_eval_table(env, "coverage").dataset == "pins"


def test_posting_an_invalid_table_re_renders_the_form(client, env):
    r = client.post("/evals/new", data={"yaml": "name: bad\ncolumns: [{name: eval_id, type: str}]\n"})
    assert r.status_code == 200
    assert "Error" in r.text
    assert evals.list_eval_tables(env) == []


JSONL = "\n".join([
    json.dumps({"id": "n1", "value": 1.0, "label": 1}),
    json.dumps({"id": "n2", "value": 2.0, "label": 2}),
    json.dumps({"id": "bad", "value": 3.0}),          # missing the required label
])


def test_ingest_preview_reports_rows_that_fail_validation(client, seeded, env):
    r = client.post("/datasets/clips/ingest/preview", data={"mode": "jsonl", "jsonl": JSONL})
    assert r.status_code == 200
    assert "first 3 of 3" in r.text
    assert "lines were rejected" in r.text
    assert "Add 2 rows" in r.text              # the invalid row is not counted
    assert ds.count_rows(env, "clips") == 5   # a preview saves nothing


def test_ingest_preview_reports_malformed_json(client, seeded):
    """The preview exists to catch bad input, so a typo must not be a server error."""
    r = client.post("/datasets/clips/ingest/preview", data={"mode": "jsonl", "jsonl": "not json"})
    assert r.status_code == 200
    assert "not valid JSON" in r.text

    # A bad line is dropped; the good ones around it still go in.
    mixed = '{"id": "ok", "value": 0.1, "label": 1}\noops\n{"id": "ok2", "value": 0.2, "label": 2}\n'
    r = client.post("/datasets/clips/ingest/preview", data={"mode": "jsonl", "jsonl": mixed})
    assert r.status_code == 200
    assert "Add 2 rows" in r.text


def test_ingest_appends_only_the_valid_rows(client, seeded, env):
    r = client.post("/datasets/clips/ingest", data={"mode": "jsonl", "jsonl": JSONL})
    assert r.status_code == 200
    assert str(r.url).endswith("/datasets/clips")
    keys = {row["id"] for row in ds.read_rows(env, "clips")}
    assert {"n1", "n2"} <= keys
    assert "bad" not in keys
    assert ds.count_rows(env, "clips") == 7


# --- run comparison ---------------------------------------------------------

def test_comparing_two_runs_shows_both_rows_and_a_delta(client, seeded):
    """The delta says how the selected run did against the run it is compared with."""
    body = client.get(f"/evals/clip-acc?run={seeded['top']}&cmp={seeded['base']}").text
    assert "Comparison (2)" in body
    assert "top run" in body and "baseline run" in body
    assert "this run" in body
    assert "+0.200" in body                  # the selected run scores higher
    assert "did against each run" in body

    # Selecting the other run turns the same comparison around.
    flipped = client.get(f"/evals/clip-acc?run={seeded['base']}&cmp={seeded['top']}").text
    assert "-0.200" in flipped


def test_a_run_from_another_eval_version_is_not_compared(client, seeded):
    body = client.get(f"/evals/clip-acc?run={seeded['top']}&cmp={seeded['old']}").text
    assert "Comparison" not in body
    assert "Result: top run" in body
    assert "old run" not in body


def test_an_unknown_cmp_id_is_ignored(client, seeded):
    r = client.get(f"/evals/clip-acc?run={seeded['top']}&cmp=e_nope")
    assert r.status_code == 200
    assert "Comparison" not in r.text


# --- security headers -------------------------------------------------------

@pytest.mark.parametrize("path", [p for p, _ in PAGES])
def test_html_pages_carry_the_security_headers(client, seeded, path):
    r = client.get(path)
    csp = r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in csp
    assert "script-src" not in csp                                    # scripting stays off
    assert "'unsafe-inline'" in csp.split("style-src")[1].split(";")[0]  # stone widths are styles


# --- assets -----------------------------------------------------------------

def test_image_asset_is_served_with_its_image_type(client, seeded):
    r = client.get(f"/evals/clip-acc/runs/{seeded['top']}/assets/plot.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == PNG


def test_non_image_asset_is_not_served_as_an_active_document(client, seeded):
    r = client.get(f"/evals/clip-acc/runs/{seeded['top']}/assets/notes.txt")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"


MARK = '<span class="tag t-false"'          # the "other data" tag on a comparison row


def test_a_run_scored_on_other_data_is_marked(client, env, seeded):
    """Same eval version is only half of comparable; the other half is the same data."""
    ds.append(env, "clips", [{"id": "r9", "value": 0.9, "label": 9}],
              created_by="tester", input={"jsonl": "more.jsonl"})
    newer = _seed_run(env, title="run after more rows", version="v2", accuracy=0.8,
                      when="2026-01-04T09:00:00+00:00")

    # Offered, but the picker says so before the click.
    assert "(other data)" in client.get(f"/evals/clip-acc?run={newer}").text

    body = client.get(f"/evals/clip-acc?run={newer}&cmp={seeded['top']}").text
    assert "Comparison (2)" in body
    assert MARK in body                      # ... and again on the row it produced

    # Two runs on the same rows carry no such mark.
    same = client.get(f"/evals/clip-acc?run={seeded['top']}&cmp={seeded['base']}").text
    assert MARK not in same


def test_an_eval_table_with_no_runs_yet_still_lists(client, env):
    """A table created a minute ago has no stones to draw, and that is not an error."""
    evals.create_eval_table(env, "name: fresh\ncolumns: [{name: score, type: float}]\n")

    body = client.get("/evals").text

    assert "fresh" in body and "<svg" not in body.split("fresh")[1].split("</tr>")[0]


def test_jsonl_that_is_not_rows_is_reported_line_by_line(client, seeded):
    """The paste box takes whatever a person pastes; the preview is where that is caught."""
    text = ('{"id": "ok", "label": 1}\n'
            "\n"                                   # blank lines are skipped, not counted
            "{oops\n"
            "[1, 2]\n")

    body = client.post("/datasets/clips/ingest/preview",
                       data={"mode": "jsonl", "jsonl": text}).text

    assert "line 3 is not valid JSON" in body
    assert "line 4 is not a JSON object" in body
    assert "Add 1 rows" in body


# --- issuing a run when nothing is registered -------------------------------

def test_a_run_is_started_for_the_pipeline_to_perform(client, env, seeded):
    """cairn pins the data and never infers: the pipeline does that and writes back."""
    form = client.get("/evals/clip-acc/run").text
    assert ">Run</button>" in form
    assert 'name="config"' in form                    # free JSON instead of generated fields

    r = client.post("/evals/clip-acc/run", follow_redirects=False, data={
        "title": "run on the cluster", "dataset": "clips",
        "evaluator": "v2", "config": '{"weights": "s3://models/2026-08-26.pt"}'})

    assert r.status_code == 303
    eval_id = r.headers["location"].split("started=")[1]
    meta = evals.get_meta(env, "clip-acc", eval_id)
    assert meta["config"] == {"weights": "s3://models/2026-08-26.pt"}
    assert meta["evaluator_version"] == "v2"
    assert [w["eval_id"] for w in evals.pending_runs(env, "clip-acc")] == [eval_id]


@pytest.mark.parametrize("form, detail", [
    ({"dataset": "nope"}, "dataset not found"),
    ({"evaluator": "latest"}, "eval version must be v1, v2"),
    ({"title": "  "}, "Title is required"),
    ({"config": "{oops"}, "config is not valid JSON"),
    ({"config": "[1, 2]"}, "config must be a JSON object"),
])
def test_a_run_request_is_checked_before_the_run_exists(client, env, seeded, form, detail):
    before = len(evals.pending_runs(env, "clip-acc"))

    r = client.post("/evals/clip-acc/run", data={
        "title": "t", "dataset": "clips", "evaluator": "v2", **form})

    assert r.status_code == 400 and detail in r.text
    assert len(evals.pending_runs(env, "clip-acc")) == before


def test_a_refusal_is_a_page_rather_than_json(client, seeded):
    """These are read by a person who clicked something, not by a program."""
    r = client.post("/evals/clip-acc/run", data={"title": "", "dataset": "clips", "evaluator": "v2"},
                    headers={"referer": "http://testserver/evals/clip-acc/run"})

    assert r.status_code == 400
    assert "text/html" in r.headers["content-type"]
    assert "Title is required" in r.text
    assert 'href="http://testserver/evals/clip-acc/run"' in r.text     # back where they were


def test_a_refusal_falls_back_to_a_link_it_knows(client, seeded):
    """Without a referer, or with one from somewhere else, the way back is our own."""
    r = client.get("/datasets/nope", headers={"referer": "https://example.com/"})

    assert r.status_code == 404
    assert 'href="/evals"' in r.text


# --- an index page with more than one row -----------------------------------

def test_the_index_pages_hold_up_with_several_of_each(client, env, seeded):
    """Each row is an independent lookup, and they are built at the same time.

    Against a bucket a row is almost entirely waiting on the network, so a page built
    one row after another gets slower with every table someone adds.
    """
    ds.create_dataset(env, DATASET_SCHEMA.replace("name: clips", "name: clips-b"),
                      created_by="tester")
    ds.append(env, "clips-b", [{"id": "b1", "value": 0.1, "label": 1}], created_by="tester")
    evals.create_eval_table(env, TABLE_SCHEMA.replace("name: clip-acc", "name: clip-acc-b"))

    datasets, tables = client.get("/datasets"), client.get("/evals")

    assert datasets.status_code == 200 and "clips-b" in datasets.text and "clips" in datasets.text
    assert tables.status_code == 200 and "clip-acc-b" in tables.text and "clip-acc" in tables.text


def test_the_ingest_form_comes_from_the_declaration_file(client, seeded, tmp_path):
    """`input.py` is read to draw the form; the ingest script is not touched."""
    decl = tmp_path / "scripts" / "datasets" / "clips" / "input.py"
    decl.parent.mkdir(parents=True, exist_ok=True)
    decl.write_text("from pydantic import BaseModel\n\n\n"
                    "class Input(BaseModel):\n    rows: int = 5\n    note: str\n", encoding="utf-8")

    body = client.get("/datasets/clips/ingest").text

    assert 'name="rows"' in body and 'name="note"' in body
