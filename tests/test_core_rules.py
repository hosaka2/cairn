"""What core refuses, and how it reads values back.

The platform fixes layout, naming and immutability; these are the rules that come with
that, stated one by one. Anything a script decides for itself is not here.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.config import load_config, load_dotenv
from cairn.core.records import EvalResult, Metric
from cairn.core.schema import parse_dataset_schema, parse_eval_table
from cairn.core.storage import Storage

DS = "name: d\nkey: id\ncolumns: [{name: id, type: str, required: true}, {name: v, type: int}]\n"
TABLE = "name: t\ncolumns: [{name: score, type: float}]\n"


def _st(tmp_path: Path) -> Storage:
    return Storage(f"file://{tmp_path}/cairn")


# --- names are claimed once -------------------------------------------------

def test_a_dataset_name_cannot_be_taken_twice(tmp_path):
    """Creating over an existing ledger would hide its history behind a new schema."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    with pytest.raises(FileExistsError, match="already exists"):
        ds.create_dataset(st, DS, created_by="t")


def test_an_eval_table_name_cannot_be_taken_twice(tmp_path):
    st = _st(tmp_path)
    evals.create_eval_table(st, TABLE)
    with pytest.raises(FileExistsError, match="already exists"):
        evals.create_eval_table(st, TABLE)


# --- what a run must carry --------------------------------------------------

def test_a_run_needs_a_title(tmp_path):
    """A run nobody can name in a list is a run nobody can find later."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    evals.create_eval_table(st, TABLE)
    with pytest.raises(ValueError, match="title is required"):
        evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="  ")


def test_the_code_that_produced_a_run_can_be_stored_beside_it(tmp_path):
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    evals.create_eval_table(st, TABLE)

    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="run",
                           code={"v1.py": "EVALUATOR = Eval\n"})

    assert st.read_bytes(f"evals/t/runs/{eid}/code/v1.py") == b"EVALUATOR = Eval\n"


def test_a_row_that_does_not_match_the_table_is_refused(tmp_path):
    """table.yaml is the shape of the list; a row that breaks it would break the page."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="run")

    class Wrong:
        def score(self, ctx):
            return EvalResult(row={"score": 0.5, "surprise": 1}, metrics=[])

    with pytest.raises(ValueError, match="does not match table.yaml"):
        evals.score_run(st, "t", eid, Wrong())


# --- metrics ----------------------------------------------------------------

def test_metrics_are_read_back_as_written(tmp_path):
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="run")

    class Eval:
        def score(self, ctx):
            return EvalResult(row={"score": 0.5},
                              metrics=[Metric(name="score", value=0.5),
                                       Metric(name="score", value=1.0, sample_id="a")])

    evals.score_run(st, "t", eid, Eval())
    metrics = evals.read_metrics(st, "t", eid)

    assert {(m["name"], m["value"], m["sample_id"]) for m in metrics} == {
        ("score", 0.5, None), ("score", 1.0, "a")}


def test_a_run_with_no_metrics_file_reads_as_no_metrics(tmp_path):
    st = _st(tmp_path)
    evals.create_eval_table(st, TABLE)
    assert evals.read_metrics(st, "t", "e_nothing") == []


def test_a_table_without_a_scored_run_counts_zero(tmp_path):
    st = _st(tmp_path)
    evals.create_eval_table(st, TABLE)
    assert evals.count_runs(st, "t") == 0
    assert evals.list_runs(st, "t") == []


# --- column types -----------------------------------------------------------

@pytest.mark.parametrize("value, ok", [
    ("2026-01-02T09:00:00+00:00", True),
    ("2026-01-02T09:00:00Z", True),
    (dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc), True),
    (dt.date(2026, 1, 2), True),
    ("yesterday", False),
    (17, False),
])
def test_a_datetime_column_accepts_what_can_be_read_as_a_time(value, ok):
    schema = parse_dataset_schema(
        "name: d\nkey: id\ncolumns: [{name: id, type: str}, {name: at, type: datetime}]\n")
    assert (schema.validate_row({"id": "a", "at": value}) == []) is ok


@pytest.mark.parametrize("value, ok", [
    ([0.1, 2], True),          # ints are acceptable floats
    ([], True),
    (["a"], False),
    ("0.1", False),
])
def test_a_list_column_checks_what_is_inside_it(value, ok):
    schema = parse_dataset_schema(
        "name: d\nkey: id\ncolumns: [{name: id, type: str}, {name: xs, type: 'list[float]'}]\n")
    assert (schema.validate_row({"id": "a", "xs": value}) == []) is ok


def test_exactly_one_column_can_be_the_primary_metric():
    """Two stone widths is no stone width."""
    with pytest.raises(ValueError, match="exactly one column"):
        parse_eval_table("name: t\ncolumns:\n"
                         "  - {name: a, type: float, primary: true}\n"
                         "  - {name: b, type: float, primary: true}\n")


def test_a_result_row_is_checked_against_the_declared_columns():
    schema = parse_eval_table("name: t\ncolumns: [{name: score, type: float}]\n")
    assert schema.validate_result_row({"score": 0.5}) == []
    assert "columns not in table.yaml: ['extra']" in schema.validate_result_row(
        {"score": 0.5, "extra": 1})
    assert "'score' must be of type float" in schema.validate_result_row({"score": "high"})


# --- storage paths ----------------------------------------------------------

def test_a_native_path_maps_back_to_a_root_relative_one(tmp_path):
    st = _st(tmp_path)
    assert st.rel_of(f"{st.base}/evals/t/runs/e1/") == "evals/t/runs/e1"
    assert st.rel_of("/somewhere/else") == "somewhere/else"      # not under the root


def test_duckdb_reads_local_roots_as_paths_and_remote_roots_as_urls(tmp_path):
    assert _st(tmp_path).duckdb_path("runs/*.json").startswith("/")
    assert Storage("s3://bucket/cairn").duckdb_path("runs/*.json") == "s3://bucket/cairn/runs/*.json"


# --- configuration ----------------------------------------------------------

def test_dotenv_fills_in_what_the_environment_does_not_have(tmp_path, monkeypatch):
    """`.env` is a convenience, so it must never override what the shell already set."""
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nnot a setting\n"
                   'export CAIRN_ROOT="file:///from/dotenv"\n'
                   "CAIRN_LANG='ja'\n", encoding="utf-8")
    monkeypatch.delenv("CAIRN_ROOT", raising=False)
    monkeypatch.setenv("CAIRN_LANG", "en")
    monkeypatch.setattr("cairn.core.config._DOTENV_LOADED", False)

    load_dotenv(env)

    assert os.environ["CAIRN_ROOT"] == "file:///from/dotenv"   # filled in
    assert os.environ["CAIRN_LANG"] == "en"                   # left alone


def test_without_a_root_the_error_says_how_to_set_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # no .env to pick up
    monkeypatch.delenv("CAIRN_ROOT", raising=False)
    monkeypatch.setattr("cairn.core.config._DOTENV_LOADED", False)

    with pytest.raises(RuntimeError, match="CAIRN_ROOT"):
        load_config()


# --- stores that are not a local disk ----------------------------------------

def test_a_store_with_no_directories_writes_the_same_way():
    """Object stores have no directories and publish a PUT atomically; no temp file."""
    st = Storage("memory://cairn")
    st.write_bytes("evals/t/runs/e1/result/row.json", b'{"score": 1}')

    assert st.read_bytes("evals/t/runs/e1/result/row.json") == b'{"score": 1}'
    assert st.exists("evals/t/runs/e1/result/row.json")


@pytest.mark.parametrize("root", ["s3:::bucket/prefix", "s3:/bucket/prefix", "s3:bucket"])
def test_a_root_that_is_not_a_url_is_refused(root):
    """A mistyped scheme quietly becomes a local directory of that name.

    The registry would come up empty, or fill up somewhere nobody is looking, and
    "0 datasets" reads exactly like an empty registry.
    """
    with pytest.raises(ValueError, match="is not a URL"):
        Storage(root)


@pytest.mark.parametrize("root", ["s3://", "s3:///", "file://"])
def test_a_root_that_names_no_place_is_refused(root):
    """`s3://` parses as the whole of S3, where nothing of ours is ever found."""
    with pytest.raises(ValueError, match="names no bucket or directory"):
        Storage(root)


def test_a_batch_of_keys_is_one_trip_out_and_one_trip_back():
    """On a store with no directories, a batch is written and read as a batch.

    Each key is its own round trip, and a chunk of predictions or a merge over a log
    would otherwise pay for them one after another.
    """
    st = Storage("memory://batched")

    st.write_many({"p/a.json": b'{"pred": 1}', "p/b.json": b'{"pred": 0}'})

    assert st.read_many(["p/a.json", "p/b.json"]) == {"p/a.json": b'{"pred": 1}',
                                                      "p/b.json": b'{"pred": 0}'}
    assert st.read_many_json(["p/a.json"]) == {"p/a.json": {"pred": 1}}
    assert st.read_many(["p/gone.json"], missing_ok=True) == {}
    st.write_many({})                      # nothing to write is not a trip at all
    assert sorted(st.ls("p")) == ["a.json", "b.json"]


def test_a_backend_that_refuses_to_make_directories_is_not_an_error(monkeypatch):
    st = Storage("memory://nodirs")

    def no_dirs(*args, **kwargs):
        raise OSError("this backend has no directories")

    monkeypatch.setattr(st.fs, "makedirs", no_dirs)
    st.write_bytes("a/b.json", b"{}")

    assert st.read_bytes("a/b.json") == b"{}"


# --- schemas can be passed as objects, not only as text ----------------------

def test_a_parsed_schema_can_be_handed_in_instead_of_yaml(tmp_path):
    """The CLI parses the file first so it can report a bad one; core takes either."""
    st = _st(tmp_path)
    ds.create_dataset(st, parse_dataset_schema(DS), created_by="t")
    evals.create_eval_table(st, parse_eval_table(TABLE))

    assert ds.list_datasets(st) == ["d"] and evals.list_eval_tables(st) == ["t"]


def test_rows_written_after_a_checkpoint_are_merged_in(tmp_path):
    """Paging reads the checkpoint; anything appended since has to be added on top."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    for i in range(ds.CHECKPOINT_EVERY):
        ds.append(st, "d", [{"id": f"r{i}", "v": i}], created_by="t", input={})
    assert st.ls("datasets/d/manifest")                      # a checkpoint exists
    ds.append(st, "d", [{"id": "late", "v": 99}], created_by="t", input={})

    ids = [r["id"] for r in ds.rows_page(st, "d", limit=100)]

    assert "late" in ids and len(ids) == ds.CHECKPOINT_EVERY + 1


def test_an_operation_from_a_newer_writer_is_skipped(tmp_path):
    """Rows are append-only, so an unknown op must be ignored rather than fatal."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}], created_by="t", input={})
    st.write_json("datasets/d/rows/01FUTURE.json", {"ops": [{"op": "reticulate", "row": {"id": "b"}}]})

    assert [r["id"] for r in ds.read_rows(st, "d")] == ["a"]


def test_deleting_a_key_that_is_not_there_is_not_an_error(tmp_path):
    st = _st(tmp_path)
    st.rm("nothing/here.json")


# --- runs whose inference happens elsewhere ----------------------------------

def test_a_run_is_pending_until_it_is_scored(tmp_path):
    """Between issuing a run and scoring it, the predictions are being written elsewhere.

    How far along that is belongs to whatever is writing them, not to the ledger.
    """
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}, {"id": "b", "v": 2}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="started")

    waiting = evals.pending_runs(st, "t")
    assert [(w["eval_id"], w["title"]) for w in waiting] == [(eid, "started")]

    evals.write_prediction(st, "t", eid, "a", b"{}")
    assert [w["eval_id"] for w in evals.pending_runs(st, "t")] == [eid]

    class Eval:
        def score(self, ctx):
            return EvalResult(row={"score": 1.0})

    evals.score_run(st, "t", eid, Eval())
    assert evals.pending_runs(st, "t") == []


def test_pending_runs_are_newest_first_and_skip_a_half_written_one(tmp_path):
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    first = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="first")
    second = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="second")
    st.write_bytes("evals/t/runs/e_HALFWAY/predictions/x.bin", b"{}")   # no _meta.json yet

    assert [w["eval_id"] for w in evals.pending_runs(st, "t")] == [second, first]


def test_the_targets_of_a_run_are_the_data_as_it_was_pinned(tmp_path):
    """What a job running elsewhere asks for: whatever was appended later is not its work."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="started")

    ds.append(st, "d", [{"id": "later", "v": 2}], created_by="t", input={})

    assert evals.target_ids(st, "t", eid) == ["a"]


# --- taking a run back -------------------------------------------------------

def _eval(score: float = 0.5):
    class Eval:
        def score(self, ctx):
            return EvalResult(row={"score": score})
    return Eval()


def test_a_withdrawn_run_leaves_the_listings_but_not_the_ledger(tmp_path):
    """A mistake should stop being read as a result without pretending it never happened."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="wrong")
    evals.write_prediction(st, "t", eid, "a", b"{}")
    evals.score_run(st, "t", eid, _eval())

    note = evals.withdraw_run(st, "t", eid, created_by="tester", reason="scored too early")

    assert evals.list_runs(st, "t") == [] and evals.count_runs(st, "t") == 0
    assert evals.pending_runs(st, "t") == []
    assert note["reason"] == "scored too early"
    assert evals.get_meta(st, "t", eid)["title"] == "wrong"       # the run is still there
    assert evals.count_predictions(st, "t", eid) == 1


def test_a_run_that_never_finished_can_be_withdrawn_too(tmp_path):
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="job died")

    evals.withdraw_run(st, "t", eid, created_by="tester")

    assert evals.pending_runs(st, "t") == []
    assert evals.withdrawn_runs(st, "t") == {eid}


def test_withdrawing_is_final(tmp_path):
    """Otherwise a result could come back under an id someone already wrote off."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="gone")
    evals.write_prediction(st, "t", eid, "a", b"{}")
    evals.withdraw_run(st, "t", eid, created_by="tester")

    with pytest.raises(FileExistsError, match="was withdrawn"):
        evals.score_run(st, "t", eid, _eval())


def test_a_run_that_does_not_exist_cannot_be_withdrawn(tmp_path):
    st = _st(tmp_path)
    evals.create_eval_table(st, TABLE)
    with pytest.raises(FileNotFoundError):
        evals.withdraw_run(st, "t", "e_nope", created_by="tester")


def test_a_run_stops_taking_predictions_once_it_has_a_result(tmp_path):
    """A result stands on the evidence that was there; more arriving after would not match."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}, {"id": "b", "v": 2}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="done")
    evals.write_prediction(st, "t", eid, "a", b"{}")

    evals.ensure_writable(st, "t", eid)          # while it is running, fine
    evals.score_run(st, "t", eid, _eval())

    with pytest.raises(FileExistsError, match="already evaluated"):
        evals.ensure_writable(st, "t", eid)


def test_a_withdrawn_run_takes_no_predictions_either(tmp_path):
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="gone")
    evals.withdraw_run(st, "t", eid, created_by="t")

    with pytest.raises(FileExistsError, match="was withdrawn"):
        evals.ensure_writable(st, "t", eid)


def test_the_samples_a_run_is_pinned_to_come_from_the_snapshot(tmp_path):
    """Asking what to work on is one small read, not a merge of the whole dataset."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a", "v": 1}, {"id": "b", "v": 2}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="pinned")
    ds.append(st, "d", [{"id": "later", "v": 3}], created_by="t", input={})

    assert evals.target_ids(st, "t", eid) == ["a", "b"]

    # A run pinned before snapshots named their samples still answers, the long way.
    snap = evals.get_snapshot(st, "t", eid)
    del snap["sample_ids"]
    st.write_json(f"evals/t/runs/{eid}/_snapshot.json", snap)
    assert evals.target_ids(st, "t", eid) == ["a", "b"]


# --- the order keys are written in -------------------------------------------

def test_keys_made_one_after_another_only_ever_sort_that_way():
    """Everything rests on this: rows are merged in key order, so key order has to be
    write order. Drawing the tail at random put half of each consecutive pair the wrong
    way round, because a millisecond holds far more than one write."""
    from cairn.core import ids

    made = [ids.ulid() for _ in range(5000)]

    assert made == sorted(made)
    assert len(set(made)) == len(made)


def test_a_stopped_clock_still_orders_what_it_cannot_time(monkeypatch):
    """Inside one millisecond the clock says nothing, and the tail has to say it instead."""
    from cairn.core import ids

    monkeypatch.setattr(ids.time, "time", lambda: 1_700_000_000.0)
    made = [ids.ulid() for _ in range(500)]

    assert made == sorted(made) and len(set(made)) == 500


def test_a_clock_that_steps_backwards_does_not_reorder_the_log(monkeypatch):
    """An NTP correction must not make the next write sort before the last one."""
    from cairn.core import ids

    monkeypatch.setattr(ids.time, "time", lambda: 1_700_000_000.0)
    before = ids.ulid()
    monkeypatch.setattr(ids.time, "time", lambda: 1_600_000_000.0)   # an hour of drift, backwards

    assert ids.ulid() > before


def test_a_row_written_and_then_deleted_stays_deleted(tmp_path):
    """The two writes land in the same millisecond, and the delete still wins."""
    st = Storage(f"file://{tmp_path}")
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}, {"id": "b", "v": 2}])
    ds.delete_rows(st, "d", ["a"])

    assert [r["id"] for r in ds.read_rows(st, "d")] == ["b"]
