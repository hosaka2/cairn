"""The command line, driven the way a user drives it.

Every command resolves CAIRN_ROOT, so each test gets its own root and its own registry
state — `cairn check` writes CAIRN_SCRIPTS and repopulates the in-process registry.
Assertions are on what the user sees (exit code, printed lines) and on what was stored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cairn import registry
from cairn.cli.main import app
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.storage import Storage

runner = CliRunner()

DATASET_SCHEMA = """name: things
kind: tabular
description: two labelled things
key: id
columns:
  - {name: id, type: str, required: true}
  - {name: label, type: int, required: true}
"""

EVAL_TABLE = """name: acc
description: share of positive labels
columns:
  - {name: positives, type: float, display: "positives", primary: true, direction: higher, scale: [0, 1]}
  - {name: coverage, type: float, display: "coverage"}
default_sort: created_at desc
"""

EVALUATOR_MODULE = '''"""Evaluator used by the CLI test: reads the pinned rows, reports its config."""
from pydantic import BaseModel

from cairn.core.records import EvalResult, Metric


class Positives:
    class Config(BaseModel):
        threshold: float = 0.5
        model_config = {"extra": "ignore"}

    def score(self, ctx) -> EvalResult:
        rows = ctx.dataset.rows()
        share = sum(int(r["label"]) for r in rows) / len(rows)
        return EvalResult(
            row={"positives": share, "coverage": ctx.actual_n / max(ctx.expected_n, 1)},
            metrics=[Metric(name="positives", value=share)],
            report_md=f"# positives\\nthreshold={ctx.config.threshold}",
        )
'''


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No CAIRN_* value and no registration survives a test."""
    monkeypatch.setenv("CAIRN_ROOT", f"file://{tmp_path}/env-store")
    monkeypatch.setenv("CAIRN_SCRIPTS", str(tmp_path / "no-scripts"))
    monkeypatch.setenv("CAIRN_USER", "tester")
    monkeypatch.setenv("CAIRN_LANG", "en")
    _forget_registrations()
    yield
    _forget_registrations()


def _forget_registrations() -> None:
    registry._PROBLEMS.clear()
    registry._INGESTORS.clear()
    registry._EVALUATORS.clear()
    registry._INFERENCE.clear()


@pytest.fixture
def root(tmp_path, monkeypatch) -> str:
    """A storage root of this test's own, exported the way a user exports it."""
    value = f"file://{tmp_path}/store"
    monkeypatch.setenv("CAIRN_ROOT", value)
    return value


def _run(*args: str):
    result = runner.invoke(app, list(args))
    return result


def _ok(*args: str):
    result = _run(*args)
    assert result.exit_code == 0, result.output
    return result


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- root -------------------------------------------------------------------

def test_root_prints_the_resolved_root_and_the_option_beats_the_environment(tmp_path):
    """--root wins over CAIRN_ROOT; without it the environment is what is reported."""
    from_env = _ok("root")
    assert "CAIRN_ROOT = file://" in from_env.output
    assert str(tmp_path / "env-store") in from_env.output
    assert "tag_prefix = cairn/" in from_env.output

    override = f"file://{tmp_path}/elsewhere"
    from_flag = _ok("--root", override, "root")
    assert f"CAIRN_ROOT = {override}" in from_flag.output
    assert "env-store" not in from_flag.output


# --- datasets ---------------------------------------------------------------

def test_dataset_is_created_ingested_shown_and_listed(root, tmp_path):
    schema = _write(tmp_path / "schema.yaml", DATASET_SCHEMA)
    jsonl = _write(tmp_path / "rows.jsonl", '{"id": "a", "label": 1}\n{"id": "b", "label": 0}\n')

    created = _ok("dataset", "create", "--schema", str(schema))
    assert "created dataset 'things'" in created.output

    ingested = _ok("dataset", "ingest", "things", "--jsonl", str(jsonl))
    assert "added 2 / rejected 0" in ingested.output
    assert "ingest ing_" in ingested.output

    shown = _ok("dataset", "show", "things")
    assert "# things  rows=2" in shown.output
    printed = [json.loads(line) for line in shown.output.splitlines() if line.startswith("  {")]
    assert printed == [{"id": "a", "label": 1}, {"id": "b", "label": 0}]

    listed = _ok("dataset", "ls")
    assert "things" in listed.output and "rows=2" in listed.output and "ingests=1" in listed.output

    st = Storage(root)
    assert ds.list_datasets(st) == ["things"]
    assert ds.count_rows(st, "things") == 2


def test_rows_that_fail_the_schema_are_reported_and_not_stored(root, tmp_path):
    schema = _write(tmp_path / "schema.yaml", DATASET_SCHEMA)
    jsonl = _write(
        tmp_path / "rows.jsonl",
        '{"id": "a", "label": 1}\n{"id": "bad", "label": "not-an-int"}\n{"label": 3}\n',
    )
    _ok("dataset", "create", "--schema", str(schema))

    ingested = _ok("dataset", "ingest", "things", "--jsonl", str(jsonl))
    assert "added 1 / rejected 2" in ingested.output
    assert "✗ bad:" in ingested.output
    assert "must be of type int" in ingested.output

    shown = _ok("dataset", "show", "things")
    assert "rows=1" in shown.output
    assert "bad" not in shown.output
    assert {r["id"] for r in ds.read_rows(Storage(root), "things")} == {"a"}


def test_vacuum_reclaims_old_checkpoints_and_keeps_the_rows(root, tmp_path):
    schema = _write(tmp_path / "schema.yaml", DATASET_SCHEMA)
    _ok("dataset", "create", "--schema", str(schema))
    st = Storage(root)
    for i in range(ds.CHECKPOINT_EVERY * 2):
        ds.append(st, "things", [{"id": f"x{i}", "label": i % 2}], created_by="tester")
    assert len(ds._checkpoints(st, "things")) >= 2

    # Fresh checkpoints are left alone: a read that chose one is still allowed to finish.
    assert "removed_checkpoints=0" in _ok("vacuum").output
    assert "removed_checkpoints=0" in _ok("vacuum", "--keep", "1").output   # superseded, still young

    vacuumed = _ok("vacuum", "--min-age-days", "0", "--keep", "1")
    assert "things" in vacuumed.output
    removed = int(vacuumed.output.split("removed_checkpoints=")[1].split()[0])
    assert removed >= 1
    assert len(ds._checkpoints(st, "things")) == 1

    still_there = _ok("dataset", "show", "things", "--limit", "100")
    assert f"rows={ds.CHECKPOINT_EVERY * 2}" in still_there.output
    assert ds.count_rows(st, "things") == ds.CHECKPOINT_EVERY * 2


# --- evals ------------------------------------------------------------------

def _seeded_dataset(tmp_path: Path) -> None:
    schema = _write(tmp_path / "schema.yaml", DATASET_SCHEMA)
    jsonl = _write(tmp_path / "rows.jsonl", '{"id": "a", "label": 1}\n{"id": "b", "label": 0}\n')
    _ok("dataset", "create", "--schema", str(schema))
    _ok("dataset", "ingest", "things", "--jsonl", str(jsonl))


def test_eval_table_run_and_score_are_created_shown_and_listed(root, tmp_path, monkeypatch):
    _seeded_dataset(tmp_path)
    table = _write(tmp_path / "table.yaml", EVAL_TABLE)
    _write(tmp_path / "scoring.py", EVALUATOR_MODULE)
    monkeypatch.chdir(tmp_path)   # the CLI puts the current directory on sys.path

    created = _ok("eval", "create-table", "--table", str(table))
    assert "created eval table 'acc'" in created.output

    made = _ok("eval", "create-run", "acc", "--dataset", "things",
               "--evaluator-version", "v1", "--title", "first pass",
               "--comment", "baseline", "--config", '{"threshold": 0.25}')
    eval_id = made.output.strip()
    assert eval_id.startswith("e_")

    scored = _ok("eval", "score", "acc", eval_id, "--evaluator", "scoring:Positives")
    row = json.loads(scored.output)
    assert row["positives"] == 0.5
    assert row["status"] == "scored"
    assert row["evaluator_version"] == "v1"
    assert row["title"] == "first pass"
    assert row["created_by"] == "tester"

    shown = _ok("eval", "show", "acc", eval_id)
    assert '"threshold": 0.25' in shown.output      # the run config was pinned
    assert "# result row" in shown.output
    assert "# report.md" in shown.output
    assert "threshold=0.25" in shown.output         # and reached the evaluator

    tables = _ok("eval", "ls")
    assert "acc" in tables.output and "runs=1" in tables.output

    runs = _ok("eval", "ls", "--table", "acc")
    assert eval_id in runs.output and "v1" in runs.output and "first pass" in runs.output

    st = Storage(root)
    assert evals.list_eval_tables(st) == ["acc"]
    stored = evals.get_run(st, "acc", eval_id)
    assert stored["row"]["positives"] == 0.5
    assert stored["meta"]["comment"] == "baseline"


def test_score_rejects_an_evaluator_that_is_not_module_colon_class(root, tmp_path):
    _seeded_dataset(tmp_path)
    _ok("eval", "create-table", "--table", str(_write(tmp_path / "table.yaml", EVAL_TABLE)))
    made = _ok("eval", "create-run", "acc", "--dataset", "things",
               "--evaluator-version", "v1", "--title", "first pass")

    result = _run("eval", "score", "acc", made.output.strip(), "--evaluator", "scoring.Positives")
    assert result.exit_code != 0
    assert "module.path:Attr" in result.output


def test_create_run_without_a_title_fails_and_creates_nothing(root, tmp_path):
    _seeded_dataset(tmp_path)
    table = _write(tmp_path / "table.yaml", EVAL_TABLE)
    _ok("eval", "create-table", "--table", str(table))

    result = _run("eval", "create-run", "acc", "--dataset", "things", "--evaluator-version", "v1")
    assert result.exit_code != 0
    assert "--title" in result.output

    assert evals.list_runs(Storage(root), "acc") == []
    assert _ok("eval", "ls").output.strip() == "acc                  runs=0"


# --- check ------------------------------------------------------------------

def test_check_reports_no_problems_for_a_freshly_scaffolded_tree(tmp_path):
    scripts = tmp_path / "scripts"
    _ok("new", "dataset", "readings", "--dir", str(scripts))
    _ok("new", "eval", "quality", "--dir", str(scripts))

    result = _ok("check", "--dir", str(scripts))
    assert "no problems" in result.output
    assert "1 dataset(s) / 1 eval(s)" in result.output


def test_check_fails_and_names_the_script_whose_export_is_missing(tmp_path):
    scripts = tmp_path / "scripts"
    _ok("new", "dataset", "readings", "--dir", str(scripts))
    ingest = scripts / "datasets" / "readings" / "ingest.py"
    ingest.write_text(ingest.read_text(encoding="utf-8").replace("INGESTOR = Ingest", ""),
                      encoding="utf-8")

    result = _run("check", "--dir", str(scripts))
    assert result.exit_code != 0
    assert "1 problem(s)" in result.output
    assert str(ingest) in result.output
    assert "`INGESTOR` is missing" in result.output


def test_check_reports_a_schema_that_does_not_parse(tmp_path):
    scripts = tmp_path / "scripts"
    _ok("new", "dataset", "readings", "--dir", str(scripts))
    (scripts / "datasets" / "readings" / "schema.yaml").write_text("columns: [\n", encoding="utf-8")

    result = _run("check", "--dir", str(scripts))
    assert result.exit_code != 0
    assert "schema.yaml" in result.output


def test_check_reports_a_dataset_directory_without_a_schema(tmp_path):
    scripts = tmp_path / "scripts"
    _ok("init", "--dir", str(scripts))
    (scripts / "datasets" / "readings").mkdir()

    result = _run("check", "--dir", str(scripts))
    assert result.exit_code != 0
    assert "schema.yaml is missing" in result.output


# --- docs -------------------------------------------------------------------

def test_docs_lists_the_pages_when_none_is_named():
    result = _ok("docs")
    assert "usage: cairn docs <page>" in result.output
    assert "manual" in result.output and "scripting" in result.output


def test_docs_prints_the_manual():
    result = _ok("docs", "manual")
    assert result.output.startswith("# Usage")
    assert "CAIRN_ROOT" in result.output


def test_docs_rejects_an_unknown_page_without_a_traceback():
    result = _run("docs", "nope")
    assert result.exit_code != 0
    assert "No such page: nope" in result.output


# --- demo -------------------------------------------------------------------

def test_demo_seed_populates_a_fresh_root_with_scored_runs(root):
    result = _ok("demo-seed")
    assert "Demo data loaded" in result.output

    st = Storage(root)
    assert ds.list_datasets(st) == ["sensor-anomaly-A", "sensor-anomaly-B"]
    for name in ds.list_datasets(st):
        info = ds.info(st, name)
        assert info["rows"] > 0
        assert info["ingests"] >= 2          # appended in batches, then corrected

    assert evals.list_eval_tables(st) == ["anomaly", "anomaly-siteB"]
    for table in evals.list_eval_tables(st):
        runs = evals.list_runs(st, table)
        assert len(runs) == 5
        assert {r["evaluator_version"] for r in runs} == {"v1", "v2"}
        for r in runs:
            assert r["status"] == "scored"
            assert 0 < r["f1"] <= 1 and 0 < r["precision"] <= 1 and 0 < r["recall"] <= 1
        assert len({r["f1"] for r in runs}) > 1   # the metrics move with the config

    listed = _ok("dataset", "ls")
    assert "sensor-anomaly-A" in listed.output and "sensor-anomaly-B" in listed.output
    assert "runs=5" in _ok("eval", "ls").output


# --- rescoring and deleting -------------------------------------------------

RESCORING_MODULE = '''"""A second way of reading the same predictions."""
from cairn.core.records import EvalResult


class Doubled:
    def score(self, ctx) -> EvalResult:
        rows = ctx.dataset.rows()
        share = sum(int(r["label"]) for r in rows) / len(rows)
        return EvalResult(row={"positives": min(share * 2, 1.0), "coverage": 1.0},
                          report_md="# doubled")
'''


def test_rescoring_keeps_the_original_and_records_where_it_came_from(root, tmp_path, monkeypatch):
    """Reading the same predictions another way is a new run, not an overwrite."""
    _seeded_dataset(tmp_path)
    _write(tmp_path / "table.yaml", EVAL_TABLE)
    _write(tmp_path / "scoring.py", EVALUATOR_MODULE)
    _write(tmp_path / "scoring2.py", RESCORING_MODULE)
    monkeypatch.chdir(tmp_path)

    _ok("eval", "create-table", "--table", "table.yaml")
    first = _ok("eval", "create-run", "acc", "--dataset", "things",
                "--evaluator-version", "v1", "--title", "first pass").output.strip()
    _ok("eval", "score", "acc", first, "--evaluator", "scoring:Positives")

    second = _ok("eval", "rescore", "acc", first, "--evaluator", "scoring2:Doubled",
                 "--evaluator-version", "v2", "--title", "read as v2").output.strip()
    assert second.startswith("e_") and second != first

    st = Storage(root)
    old, new = evals.get_run(st, "acc", first)["row"], evals.get_run(st, "acc", second)["row"]
    assert (old["positives"], old["evaluator_version"], old["title"]) == (0.5, "v1", "first pass")
    assert (new["positives"], new["evaluator_version"], new["title"]) == (1.0, "v2", "read as v2")
    assert new["snapshot_id"] == old["snapshot_id"]          # the same data
    assert evals.get_meta(st, "acc", second)["rescored_from"] == first
    assert "runs=2" in _ok("eval", "ls").output


def test_rescoring_to_the_version_already_used_is_refused(root, tmp_path, monkeypatch):
    _seeded_dataset(tmp_path)
    _write(tmp_path / "table.yaml", EVAL_TABLE)
    _write(tmp_path / "scoring.py", EVALUATOR_MODULE)
    monkeypatch.chdir(tmp_path)

    _ok("eval", "create-table", "--table", "table.yaml")
    eid = _ok("eval", "create-run", "acc", "--dataset", "things",
              "--evaluator-version", "v1", "--title", "only").output.strip()
    _ok("eval", "score", "acc", eid, "--evaluator", "scoring:Positives")

    result = _run("eval", "rescore", "acc", eid, "--evaluator", "scoring:Positives",
                  "--evaluator-version", "v1")
    assert result.exit_code != 0
    assert evals.count_runs(Storage(root), "acc") == 1


def test_deleting_rows_reports_the_snapshot_moving(root, tmp_path):
    """A delete is an append, and it puts the dataset on different data."""
    _seeded_dataset(tmp_path)
    st = Storage(root)
    before = ds.snapshot(st, "things")["snapshot_id"]

    deleted = _ok("dataset", "delete", "things", "--key", "b")

    assert "deleted 1" in deleted.output
    assert before in deleted.output                       # says what it moved from
    assert "no longer comparable" in deleted.output or "different data" in deleted.output
    assert [r["id"] for r in ds.read_rows(st, "things")] == ["a"]
    assert ds.snapshot(st, "things")["snapshot_id"] != before
    assert len(ds.list_ingests(st, "things")) == 2         # the tombstone is its own ingest
    assert "rows=1" in _ok("dataset", "ls").output


def test_deleting_an_unknown_key_changes_nothing(root, tmp_path):
    _seeded_dataset(tmp_path)
    st = Storage(root)
    before = ds.snapshot(st, "things")["snapshot_id"]

    _ok("dataset", "delete", "things", "--key", "nope")

    assert ds.count_rows(st, "things") == 2
    assert ds.snapshot(st, "things")["snapshot_id"] == before   # content is what decides


def test_check_ignores_stray_files_beside_the_script_directories(tmp_path):
    """A README or a notebook next to the datasets is not a script to complain about."""
    scripts = tmp_path / "scripts"
    _ok("new", "dataset", "readings", "--dir", str(scripts))
    _ok("new", "eval", "quality", "--dir", str(scripts))
    (scripts / "datasets" / "notes.md").write_text("scratch", encoding="utf-8")
    (scripts / "evals" / "notes.md").write_text("scratch", encoding="utf-8")

    result = _ok("check", "--dir", str(scripts))
    assert "no problems" in result.output


def test_web_starts_uvicorn_on_the_address_it_was_given(tmp_path, monkeypatch):
    """The command is a thin wrapper, so what matters is what it hands to the server."""
    import uvicorn

    import cairn

    monkeypatch.setenv("CAIRN_ROOT", f"file://{tmp_path}/store")
    called: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: called.update(target=target, **kw))

    _ok("web", "--host", "0.0.0.0", "--port", "9999", "--reload")

    assert called["target"] == "cairn.web.app:app"
    assert (called["host"], called["port"], called["reload"]) == ("0.0.0.0", 9999, True)
    # cairn itself is watched too, or an editable install never reloads
    assert str(Path(cairn.__file__).parent) in str(called["reload_dirs"])


def test_web_says_what_is_missing_when_there_is_no_root(tmp_path, monkeypatch):
    import uvicorn

    monkeypatch.chdir(tmp_path)                     # no .env to fall back on
    monkeypatch.delenv("CAIRN_ROOT", raising=False)
    monkeypatch.setattr("cairn.core.config._DOTENV_LOADED", False)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: pytest.fail("must not start"))

    result = _run("web")

    assert result.exit_code != 0
    assert "CAIRN_ROOT" in str(result.exception or result.output)


def test_eval_show_prints_the_metadata_of_a_run_that_has_not_been_scored(tmp_path, root):
    """A run exists from the moment it is created; only the result comes later."""
    st = Storage(root)
    ds.create_dataset(st, DATASET_SCHEMA)
    evals.create_eval_table(st, EVAL_TABLE)
    eid = evals.create_run(st, "acc", dataset="things", evaluator_version="v1", title="pending")

    out = _ok("eval", "show", "acc", eid).output

    assert '"title": "pending"' in out
    assert "# result row" not in out and "# report.md" not in out


# --- driving a run from outside this process --------------------------------

def _started_run(root: str) -> tuple[Storage, str]:
    st = Storage(root)
    ds.create_dataset(st, DATASET_SCHEMA)
    ds.append(st, "things", [{"id": "a", "label": 1}, {"id": "b", "label": 0}])
    evals.create_eval_table(st, EVAL_TABLE)
    return st, _ok("eval", "create-run", "acc", "--dataset", "things",
                   "--evaluator-version", "v1", "--title", "from dagster").output.strip()


def test_targets_lists_the_samples_a_run_is_pinned_to(tmp_path, root):
    st, eid = _started_run(root)
    ds.append(st, "things", [{"id": "later", "label": 1}])      # after the snapshot

    assert _ok("eval", "targets", "acc", eid).output.split() == ["a", "b"]


def test_a_prediction_can_be_written_from_a_file_or_from_stdin(tmp_path, root, monkeypatch):
    st, eid = _started_run(root)
    payload = _write(tmp_path / "a.json", '{"pred": 1}')

    _ok("eval", "put-prediction", "acc", eid, "--sample-id", "a", "--file", str(payload),
        "--ext", "json")
    result = runner.invoke(app, ["eval", "put-prediction", "acc", eid, "--sample-id", "b",
                                 "--file", "-", "--ext", "json"], input='{"pred": 0}')
    assert result.exit_code == 0, result.output

    assert evals.count_predictions(st, "acc", eid) == 2


def test_a_batch_of_predictions_can_be_written_from_jsonl(tmp_path, root):
    st, eid = _started_run(root)
    batch = _write(tmp_path / "preds.jsonl",
                   '{"sample_id": "a", "prediction": {"pred": 1}}\n'
                   "\n"
                   '{"sample_id": "b", "prediction": "raw text"}\n')

    out = _ok("eval", "put-prediction", "acc", eid, "--jsonl", str(batch)).output

    assert "wrote 2 prediction(s)" in out
    assert dict(evals._prediction_loader(st, "acc", eid)()) == {"a": b'{"pred": 1}', "b": b"raw text"}


def test_put_prediction_says_what_it_needs(tmp_path, root):
    _, eid = _started_run(root)
    result = _run("eval", "put-prediction", "acc", eid, "--sample-id", "a")
    assert result.exit_code != 0
    assert "--jsonl" in result.output


def test_a_run_started_here_is_scored_after_the_predictions_arrive(tmp_path, root, monkeypatch):
    """The whole point of the split: issue, write predictions elsewhere, then score."""
    st, eid = _started_run(root)
    _write(tmp_path / "scoring.py", EVALUATOR_MODULE)
    monkeypatch.chdir(tmp_path)

    for sid in _ok("eval", "targets", "acc", eid).output.split():
        payload = _write(tmp_path / f"{sid}.json", '{"pred": 1}')
        _ok("eval", "put-prediction", "acc", eid, "--sample-id", sid,
            "--file", str(payload), "--ext", "json")

    _ok("eval", "score", "acc", eid, "--evaluator", "scoring:Positives")

    assert evals.get_run(st, "acc", eid)["row"]["positives"] == 0.5


def test_a_registered_script_can_perform_a_started_run_here(tmp_path, root, monkeypatch):
    """The other side of the split: small enough to infer in this process, on purpose."""
    scripts = tmp_path / "scripts"
    _ok("demo-init", "--dir", str(scripts))
    monkeypatch.setenv("CAIRN_SCRIPTS", str(scripts))
    monkeypatch.setenv("CAIRN_REGISTRY", "")
    registry._SCANNED.clear()
    registry._LOADED.clear()

    st = Storage(root)
    ds.create_dataset(st, (scripts / "datasets" / "sensor-anomaly-A" / "schema.yaml").read_text())
    ds.append(st, "sensor-anomaly-A", [{"sample_id": f"A_{i}", "site": "A", "temp": 60.0 + i,
                                        "vibration": 0.4, "pressure": 101.0, "calibrated": True,
                                        "gt": i % 2} for i in range(6)])
    evals.create_eval_table(st, (scripts / "evals" / "anomaly" / "table.yaml").read_text())
    eval_id = _ok("eval", "create-run", "anomaly", "--dataset", "sensor-anomaly-A",
                  "--evaluator-version", "v2", "--title", "run it here").output.strip()

    out = _ok("eval", "run", "anomaly", eval_id, "--chunk-size", "4").output

    assert "wrote 6 prediction(s)" in out
    assert evals.count_predictions(st, "anomaly", eval_id) == 6
    assert evals.pending_runs(st, "anomaly")[0]["eval_id"] == eval_id      # not scored yet


def test_running_a_table_with_no_script_says_what_is_missing(tmp_path, root, monkeypatch):
    scripts = tmp_path / "empty"
    scripts.mkdir()
    monkeypatch.setenv("CAIRN_SCRIPTS", str(scripts))
    monkeypatch.setenv("CAIRN_REGISTRY", "")
    registry._SCANNED.clear()
    registry._LOADED.clear()

    result = _run("eval", "run", "acc", "e_whatever")

    assert result.exit_code != 0
    assert "no inference script is registered" in result.output


def _said(result) -> str:
    """The CLI wraps errors in a box; compare against the text, not its line breaks."""
    return " ".join(result.output.split())


def test_predictions_for_a_run_that_does_not_exist_are_refused(tmp_path, root):
    """A typo in the eval_id would otherwise write predictions nothing ever reads."""
    _started_run(root)
    payload = _write(tmp_path / "p.json", '{"pred": 1}')

    result = _run("eval", "put-prediction", "acc", "e_NOPE", "--sample-id", "a",
                  "--file", str(payload))

    assert result.exit_code != 0 and "no run 'e_NOPE'" in _said(result)


def test_a_batch_names_the_line_it_could_not_read(tmp_path, root):
    """A refused batch writes nothing: the whole file is read before any of it lands.

    Half a chunk would make what the run holds depend on where the bad line was, and
    the job that has to fix the file simply sends the chunk again.
    """
    st, eid = _started_run(root)
    batch = _write(tmp_path / "preds.jsonl",
                   '{"sample_id": "a", "prediction": {"pred": 1}}\n'
                   '{"sample_id": "b"}\n')                       # no prediction

    result = _run("eval", "put-prediction", "acc", eid, "--jsonl", str(batch))

    assert result.exit_code != 0 and "line 2" in _said(result)
    assert evals.count_predictions(st, "acc", eid) == 0


def test_a_batch_refuses_a_sample_the_run_does_not_have(tmp_path, root):
    """Stale ids are the normal orchestrator mistake; they must not score as nothing."""
    st, eid = _started_run(root)
    batch = _write(tmp_path / "preds.jsonl", '{"sample_id": "typo", "prediction": 1}\n')

    result = _run("eval", "put-prediction", "acc", eid, "--jsonl", str(batch))

    said = _said(result)
    assert result.exit_code != 0 and "'typo'" in said and "sample of this run" in said
    assert evals.count_predictions(st, "acc", eid) == 0


def test_a_run_can_be_withdrawn_from_the_cli(tmp_path, root, monkeypatch):
    st, eid = _started_run(root)
    _write(tmp_path / "scoring.py", EVALUATOR_MODULE)
    monkeypatch.chdir(tmp_path)
    for sid in _ok("eval", "targets", "acc", eid).output.split():
        _ok("eval", "put-prediction", "acc", eid, "--sample-id", sid,
            "--file", str(_write(tmp_path / f"{sid}.json", '{"pred": 1}')), "--ext", "json")
    _ok("eval", "score", "acc", eid, "--evaluator", "scoring:Positives")
    assert evals.count_runs(st, "acc") == 1

    out = _ok("eval", "withdraw", "acc", eid, "--reason", "wrong weights").output

    assert f"withdrew {eid}" in out
    assert evals.count_runs(st, "acc") == 0
    assert evals.get_meta(st, "acc", eid)["title"] == "from dagster"     # still on the ledger


def test_predictions_are_refused_once_the_run_is_evaluated(tmp_path, root, monkeypatch):
    st, eid = _started_run(root)
    _write(tmp_path / "scoring.py", EVALUATOR_MODULE)
    monkeypatch.chdir(tmp_path)
    payload = _write(tmp_path / "a.json", '{"pred": 1}')
    _ok("eval", "put-prediction", "acc", eid, "--sample-id", "a", "--file", str(payload), "--ext", "json")
    _ok("eval", "score", "acc", eid, "--evaluator", "scoring:Positives")

    result = _run("eval", "put-prediction", "acc", eid, "--sample-id", "b",
                  "--file", str(payload), "--ext", "json")

    assert result.exit_code != 0 and "already evaluated" in _said(result)
    assert evals.count_predictions(st, "acc", eid) == 1


# --- what a failure looks like ----------------------------------------------

def _cairn(monkeypatch, *args: str) -> int:
    """Run through the console entry point, the way the installed `cairn` runs."""
    from cairn.cli.main import main

    monkeypatch.setattr("sys.argv", ["cairn", *args])
    try:
        main()
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def test_a_missing_table_is_named_rather_than_its_path(monkeypatch, capsys, root):
    """The user mistyped a table; the path under the root is cairn's business, not theirs."""
    code = _cairn(monkeypatch, "eval", "create-run", "nosuch", "--dataset", "things",
                  "--evaluator-version", "v1", "--title", "x")
    err = " ".join(capsys.readouterr().err.split())

    assert code == 1
    assert "no eval table 'nosuch' in this registry" in err
    assert "Traceback" not in err and "table.yaml" not in err


def test_a_missing_dataset_and_a_missing_run_are_named_too(monkeypatch, capsys, root):
    _ok("eval", "create-table", "--table", str(_write(Path(root[7:]) / "t.yaml", EVAL_TABLE)))

    code = _cairn(monkeypatch, "eval", "create-run", "acc", "--dataset", "nosuch",
                  "--evaluator-version", "v1", "--title", "x")
    assert code == 1 and "no dataset 'nosuch' in this registry" in " ".join(capsys.readouterr().err.split())

    code = _cairn(monkeypatch, "eval", "targets", "acc", "e_NOPE")
    assert code == 1
    assert "no run 'e_NOPE' in eval table 'acc'" in " ".join(capsys.readouterr().err.split())


def test_scoring_a_run_twice_is_a_message_not_a_traceback(tmp_path, monkeypatch, capsys, root):
    """The refusal was already right; what an orchestrator's log gets is the message."""
    _, eid = _started_run(root)
    _write(tmp_path / "scoring.py", EVALUATOR_MODULE)
    monkeypatch.chdir(tmp_path)
    _ok("eval", "put-prediction", "acc", eid, "--sample-id", "a",
        "--file", str(_write(tmp_path / "a.json", '{"pred": 1}')), "--ext", "json")
    _ok("eval", "score", "acc", eid, "--evaluator", "scoring:Positives")

    code = _cairn(monkeypatch, "eval", "score", "acc", eid, "--evaluator", "scoring:Positives")
    err = " ".join(capsys.readouterr().err.split())

    assert code == 1 and "already has a result" in err and "Traceback" not in err


def test_the_traceback_is_one_environment_variable_away(monkeypatch, root):
    """Hiding the frames helps a job log and hurts whoever is debugging a script."""
    monkeypatch.setenv("CAIRN_TRACEBACK", "1")

    with pytest.raises(FileNotFoundError, match="no eval table 'nosuch'"):
        _cairn(monkeypatch, "eval", "create-run", "nosuch", "--dataset", "things",
               "--evaluator-version", "v1", "--title", "x")


def test_a_command_that_works_still_exits_zero(monkeypatch, capsys, root):
    code = _cairn(monkeypatch, "root")

    assert code == 0 and "CAIRN_ROOT" in capsys.readouterr().out
