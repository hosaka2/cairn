"""Regressions for the failure modes found in review.

Each test states the property being protected, not the incident.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.storage import Storage

DS = "name: d\nkey: id\ncolumns: [{name: id, type: str, required: true}, {name: v, type: int}]\n"
TABLE = "name: t\ncolumns: [{name: score, type: float}]\n"


def _st(tmp_path: Path) -> Storage:
    return Storage(f"file://{tmp_path}/cairn")


def _client(tmp_path: Path, monkeypatch, web_app):
    """A TestClient bound to a storage root, skipped when the HTTP client is absent."""
    try:
        from fastapi.testclient import TestClient
    except RuntimeError as e:  # starlette needs httpx to build one
        pytest.skip(str(e))

    monkeypatch.setenv("CAIRN_ROOT", f"file://{tmp_path}/cairn")
    monkeypatch.setenv("CAIRN_SCRIPTS", str(tmp_path / "scripts"))
    return TestClient(web_app.create_app())


def test_names_cannot_escape_the_root(tmp_path):
    st = _st(tmp_path)
    with pytest.raises(ValidationError):
        ds.create_dataset(st, "name: ../../pwned\nkey: id\ncolumns: [{name: id, type: str}]\n")
    with pytest.raises(ValueError):
        st.write_bytes("datasets/../../pwned", b"x")
    assert not (tmp_path.parent / "pwned").exists()


def test_schema_requires_key_and_columns(tmp_path):
    st = _st(tmp_path)
    with pytest.raises(ValidationError):
        ds.create_dataset(st, "name: a\nkey: id\ncolumns: []\n")
    with pytest.raises(ValidationError):
        ds.create_dataset(st, "name: b\nkey: id\ncolumns: [{name: v, type: int}]\n")


def test_result_columns_cannot_shadow_common_metadata(tmp_path):
    """Otherwise a script could rewrite snapshot_id and break comparability."""
    st = _st(tmp_path)
    with pytest.raises(ValidationError):
        evals.create_eval_table(st, "name: t\ncolumns: [{name: snapshot_id, type: str}]\n")


def test_any_sample_id_round_trips(tmp_path):
    """Sample ids are dataset keys, so they may contain slashes, dots or anything else."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="x")

    ids = ["plain", "a/b", "../_meta", "x.y.z", "spaced id"]
    for sid in ids:
        evals.write_prediction(st, "t", eid, sid, b"{}", ext="tar.gz")

    assert sorted(s for s, _ in evals._prediction_loader(st, "t", eid)()) == sorted(ids)
    assert evals.count_predictions(st, "t", eid) == len(ids)
    assert "eval_id" in json.loads((tmp_path / "cairn/evals/t/runs" / eid / "_meta.json").read_text())


def test_append_survives_an_unwritable_checkpoint(tmp_path):
    """A checkpoint is a cache: failing to build one must not block further appends."""
    st = _st(tmp_path)
    ds.create_dataset(st, "name: j\nkey: id\ncolumns: [{name: id, type: str, required: true}, "
                          "{name: blob, type: json}]\n")
    for i in range(ds.CHECKPOINT_EVERY + 4):
        ds.append(st, "j", [{"id": f"r{i}", "blob": {"x": 1} if i % 2 else [1, 2, 3]}])
    assert ds.count_rows(st, "j") == ds.CHECKPOINT_EVERY + 4


def test_reader_prefers_the_checkpoint_that_covers_most(tmp_path):
    """A checkpoint written late but covering less must not win."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    for i in range(ds.CHECKPOINT_EVERY):
        ds.append(st, "d", [{"id": f"r{i}", "v": i}])
    ds.write_checkpoint(st, "d")
    wide = ds._newest_checkpoint(st, "d")
    st.write_json("datasets/d/manifest/ZZZZ.json", {"covered": [], "count": 0})
    assert ds._newest_checkpoint(st, "d") == wide


def test_nested_tables_must_be_named_when_ambiguous(tmp_path):
    st = _st(tmp_path)
    ds.create_dataset(st, "name: n\nkey: id\ncolumns: [{name: id, type: str, required: true}]\n"
                          "nested:\n  a: {schema: [{name: k, type: int}]}\n"
                          "  b: {schema: [{name: k, type: int}]}\n")
    ds.append(st, "n", [{"id": "s1", "a": [{"k": 1}], "b": [{"k": 2}]}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="n", evaluator_version="v1", title="x")

    seen = {}

    class Eval:
        def score(self, ctx):
            from cairn.core.records import EvalResult

            seen["b"] = ctx.dataset.frames("s1", "b")
            with pytest.raises(ValueError):
                ctx.dataset.frames("s1")
            return EvalResult(row={"score": 1.0})

    evals.score_run(st, "t", eid, Eval())
    assert seen["b"] == [{"k": 2}]


def test_csp_allows_the_inline_styles_the_templates_use():
    """Scripting stays blocked, but stone widths are set with a style attribute."""
    from cairn.web.app import _CSP

    assert "script-src" not in _CSP and "default-src 'none'" in _CSP
    assert "'unsafe-inline'" in _CSP.split("style-src")[1].split(";")[0]


def test_report_opens_on_its_own_page(tmp_path, monkeypatch):
    """A long report is readable at full width, without the surrounding run view."""
    from cairn.core.records import EvalResult

    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="a run")

    class Eval:
        def score(self, ctx):
            return EvalResult(row={"score": 1.0}, report_md="## heading\n\nbody text")

    evals.score_run(st, "t", eid, Eval())

    from cairn.web import app as web_app

    client = _client(tmp_path, monkeypatch, web_app)
    page = client.get(f"/evals/t/runs/{eid}/report")
    assert page.status_code == 200
    assert "body text" in page.text and "a run" in page.text
    # the run view links to it, and opens it in a new tab
    run_view = client.get(f"/evals/t?run={eid}").text
    assert f"/evals/t/runs/{eid}/report" in run_view
    assert 'target="_blank"' in run_view
    assert client.get("/evals/t/runs/e_MISSING/report").status_code == 404


def _scored(st, table, eval_id, row):
    from cairn.core.records import EvalResult

    class Eval:
        def score(self, ctx):
            return EvalResult(row=row)

    evals.score_run(st, table, eval_id, Eval())


def test_the_run_view_survives_awkward_input(tmp_path, monkeypatch):
    """A table with no scored run, a path-shaped id, a missing column, an old comparison."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, "name: t\ncolumns: [{name: score, type: float}, "
                                "{name: err, type: float}]\n")

    from cairn.web import app as web_app

    client = _client(tmp_path, monkeypatch, web_app)

    # nothing scored yet: the empty state, not a crash
    empty = client.get("/evals/t")
    assert empty.status_code == 200 and "No scored runs yet" in empty.text

    ids = []
    for i in range(10):
        eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title=f"run{i}")
        _scored(st, "t", eid, {"score": round(0.1 * i, 2)} | ({"err": 5.0} if i else {}))
        ids.append(eid)

    # an id shaped like a path is not found, not a 500
    assert client.get("/evals/t/runs/%2e%2e/report").status_code == 404
    assert client.get("/evals/t/runs/%2e%2e/assets/x.png").status_code == 404

    # the oldest run is outside the eight shown, yet still comparable
    page = client.get(f"/evals/t?run={ids[-1]}&cmp={ids[0]}").text
    assert "Comparison (2)" in page
    # run0 never reported `err`, which reads as absent rather than blank
    assert "—" in page.split("<tbody>")[1].split("</tbody>")[0]


def test_the_run_view_shows_which_data_it_was_scored_on(tmp_path, monkeypatch):
    """The snapshot id is what makes two runs comparable, so it has to be visible."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="a run")
    _scored(st, "t", eid, {"score": 1.0})
    snapshot_id = evals.get_meta(st, "t", eid)["snapshot_id"]

    from cairn.web import app as web_app

    client = _client(tmp_path, monkeypatch, web_app)
    assert snapshot_id in client.get(f"/evals/t?run={eid}").text
    assert snapshot_id in client.get(f"/evals/t/runs/{eid}/report").text


def test_scoring_another_way_becomes_its_own_run(tmp_path):
    """A different evaluator version is a different result, so it must not overwrite."""
    from cairn.core.records import EvalResult

    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="original")
    evals.write_prediction(st, "t", eid, "a", b"{}", ext="json")

    class V1:
        def score(self, ctx):
            return EvalResult(row={"score": 0.5})

    class V2:
        def score(self, ctx):
            return EvalResult(row={"score": 0.9})

    evals.score_run(st, "t", eid, V1(), evaluator_version="v1")

    # scoring the same run another way is refused, so a result never claims the wrong version
    with pytest.raises(ValueError):
        evals.score_run(st, "t", eid, V2(), evaluator_version="v2")

    new_id = evals.rescore(st, "t", eid, V2(), evaluator_version="v2", title="with v2")
    old, new = evals.get_run(st, "t", eid)["row"], evals.get_run(st, "t", new_id)["row"]

    assert (old["score"], old["evaluator_version"]) == (0.5, "v1")   # untouched
    assert (new["score"], new["evaluator_version"]) == (0.9, "v2")
    assert new["snapshot_id"] == old["snapshot_id"]                  # same data
    assert evals.count_predictions(st, "t", new_id) == 1             # same evidence
    assert evals.get_meta(st, "t", new_id)["rescored_from"] == eid
    assert evals.count_runs(st, "t") == 2


def test_deleting_rows_changes_what_a_later_run_is_scored_on(tmp_path):
    """A tombstone is an append, and it moves the dataset to a new snapshot."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}, {"id": "b", "v": 2}])
    before = ds.snapshot(st, "d")["snapshot_id"]

    ds.delete_rows(st, "d", ["b"], created_by="tester")

    assert [r["id"] for r in ds.read_rows(st, "d")] == ["a"]
    assert ds.count_rows(st, "d") == 1
    assert ds.snapshot(st, "d")["snapshot_id"] != before
    assert len(ds.list_ingests(st, "d")) == 2      # the delete is itself an ingest


def test_a_run_whose_files_are_incomplete_does_not_take_the_page_down(tmp_path, monkeypatch):
    """Storage can be missing pieces: a run still being written, or one half-deleted."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)

    scored = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="scored")
    _scored(st, "t", scored, {"score": 0.5})
    unscored = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="not scored yet")

    from cairn.web import app as web_app

    client = _client(tmp_path, monkeypatch, web_app)

    # A run with no result yet is listed as waiting, and contributes no row to a comparison.
    page = client.get(f"/evals/t?run={scored}&cmp={unscored}")
    assert page.status_code == 200
    assert "not scored yet" in page.text
    assert "not scored yet" not in page.text.split("<tbody>")[1].split("</tbody>")[0]

    # Asking for a run that does not exist falls back to the latest one.
    fallback = client.get("/evals/t?run=e_00000000000000000000000000")
    assert fallback.status_code == 200 and "Result: scored" in fallback.text

    # A result whose metadata is gone is dropped rather than raising.
    st.fs.rm(st._p(f"evals/t/runs/{scored}/_meta.json"))
    assert client.get("/evals/t").status_code == 200


def test_a_half_written_checkpoint_is_ignored_by_the_reader(tmp_path):
    """Checkpoints are a cache: an unreadable one must not hide the rows themselves."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    st.write_bytes("datasets/d/manifest/01BROKEN.json", b"{ not json")

    assert [r["id"] for r in ds.read_rows(st, "d")] == ["a"]
    assert ds.count_rows(st, "d") == 1


def test_asking_for_a_nested_table_that_does_not_exist_says_so(tmp_path):
    from cairn.core.records import EvalResult

    st = _st(tmp_path)
    ds.create_dataset(st, "name: d\nkey: id\ncolumns: [{name: id, type: str, required: true}]\n"
                          "nested:\n  readings:\n    schema: [{name: t, type: float}]\n")
    ds.append(st, "d", [{"id": "a", "readings": [{"t": 0.0}]}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="run")
    evals.write_prediction(st, "t", eid, "a", b"{}")

    seen: dict[str, object] = {}

    class Eval:
        def score(self, ctx):
            seen["ok"] = ctx.dataset.frames("a", "readings")
            with pytest.raises(ValueError, match="not a nested table"):
                ctx.dataset.frames("a", "typo")
            return EvalResult(row={"score": 1.0})

    evals.score_run(st, "t", eid, Eval())
    assert seen["ok"] == [{"t": 0.0}]


def test_a_result_is_written_once(tmp_path):
    """A stone that is already on the pile does not change shape underneath a reader."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="once")
    evals.write_prediction(st, "t", eid, "a", b"{}")
    _scored(st, "t", eid, {"score": 0.5})

    with pytest.raises(FileExistsError, match="already has a result"):
        _scored(st, "t", eid, {"score": 0.9})

    assert evals.get_run(st, "t", eid)["row"]["score"] == 0.5


def test_a_scoring_that_dies_halfway_can_be_done_again(tmp_path):
    """`row.json` is the marker of an evaluated run, so it is written after everything else."""
    from cairn.core.records import EvalResult

    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="interrupted")
    evals.write_prediction(st, "t", eid, "a", b"{}")

    class Dies:
        def score(self, ctx):
            return EvalResult(row={"score": 0.5}, report_md="…", assets={"plot.svg": b"<svg/>"})

    real_write = st.write_bytes

    def fail_on_assets(rel: str, data: bytes) -> None:
        if "/assets/" in rel:
            raise OSError("the network went away")
        real_write(rel, data)

    st.write_bytes = fail_on_assets                      # type: ignore[method-assign]
    with pytest.raises(OSError, match="network"):
        evals.score_run(st, "t", eid, Dies())
    st.write_bytes = real_write                          # type: ignore[method-assign]

    assert [p["eval_id"] for p in evals.pending_runs(st, "t")] == [eid]   # still unevaluated
    _scored(st, "t", eid, {"score": 0.5})                                # and it can be redone
    assert evals.get_run(st, "t", eid)["row"]["score"] == 0.5


def test_a_reader_is_never_served_a_listing_it_remembers(tmp_path):
    """The web runs for hours while an orchestrator writes; it must see what arrived.

    fsspec caches directory listings per process, and on a bucket that means a reader
    keeps answering with the runs that existed when it started — a local root never
    showed it, because LocalFileSystem has nothing to cache.
    """
    st = Storage(f"file://{tmp_path}")

    st.fs.dircache["anything"] = [{"name": "remembered"}]

    assert "anything" not in st.fs.dircache


def test_a_run_written_by_another_process_shows_up_as_running(tmp_path):
    """What `pending_runs` reports is what storage holds now, not what it held before."""
    root = f"file://{tmp_path}"
    writer, reader = Storage(root), Storage(root)
    ds.create_dataset(writer, DS)
    ds.append(writer, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(writer, TABLE)
    assert evals.pending_runs(reader, "t") == []

    eid = evals.create_run(writer, "t", dataset="d", evaluator_version="v1", title="from a job")

    assert [m["eval_id"] for m in evals.pending_runs(reader, "t")] == [eid]


def test_a_read_that_failed_is_not_read_as_a_file_that_is_not_there(monkeypatch):
    """A bucket refusing a read must not look like a run that was never evaluated.

    Storage fetches keys in batches, and dropping the ones that errored would turn a
    throttled GET into "no result here" — which is what lets predictions into a run
    that already has one.
    """
    st = Storage("memory://flaky")
    st.write_many({"a.json": b"{}"})
    monkeypatch.setattr(st.fs, "cat", lambda paths, **kw: {st._p("a.json"): OSError("503 SlowDown")})

    with pytest.raises(OSError, match="503"):
        st.read_many(["a.json"], missing_ok=True)


def test_a_withdrawal_that_dies_halfway_leaves_no_result_standing(tmp_path, monkeypatch):
    """Withdrawing is final, so the result goes first and the note second.

    The other order leaves a run whose result is still listed and whose withdrawal is
    invisible — nothing on the screen says anything is wrong.
    """
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="a mistake")
    _scored(st, "t", eid, {"score": 0.5})
    assert [r["eval_id"] for r in evals.list_runs(st, "t")] == [eid]

    monkeypatch.setattr(st, "write_json", _dies)
    with pytest.raises(OSError):
        evals.withdraw_run(st, "t", eid, reason="wrong weights")

    assert evals.list_runs(st, "t") == []                      # not read as a result any more
    assert evals.runs_view(st, "t").pending() == [eid]         # visible, and can be withdrawn again


def _dies(*args, **kwargs):
    raise OSError("the note never landed")


def test_a_run_that_never_finished_starting_says_so(tmp_path):
    """`create_run` writes the metadata and then the snapshot; scoring in between said
    `KeyError: 'keys'`, which reaches an orchestrator's log as a traceback."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="half started")
    st.rm(f"evals/t/runs/{eid}/_snapshot.json")

    with pytest.raises(FileNotFoundError, match="never finished starting"):
        evals.score_run(st, "t", eid, _NeverCalled())


def test_one_unreadable_result_does_not_hide_the_rest(tmp_path, monkeypatch):
    """A table's listing is how someone finds the run that needs fixing.

    Nothing cairn writes is half a file — a PUT is atomic and a local write goes
    through a temporary — but something else writing into the store must not be able
    to take a whole table off the screen.
    """
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    good = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="readable")
    _scored(st, "t", good, {"score": 0.5})
    st.write_bytes("evals/t/runs/e_BROKEN/result/row.json", b"{ not json")

    from cairn.web import app as web_app

    page = _client(tmp_path, monkeypatch, web_app).get("/evals/t")

    assert [r["eval_id"] for r in evals.list_runs(st, "t")] == [good]
    assert evals.count_runs(st, "t") == 1
    assert page.status_code == 200 and "readable" in page.text


def test_a_file_that_is_not_json_is_named(tmp_path):
    """Where the content is the answer, a bad file stops the read and says which one.

    Dropping it quietly would change the rows a run is scored on, and with them the
    snapshot two runs are compared by.
    """
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    st.write_bytes("datasets/d/rows/BAD.json", b"{ not json")

    with pytest.raises(ValueError, match="datasets/d/rows/BAD.json is not readable JSON"):
        ds.count_rows(st, "d")


def test_a_listing_shows_what_it_can_read(tmp_path):
    """The Running panel and the ingest list keep going past a file they cannot read."""
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 1}])
    evals.create_eval_table(st, TABLE)
    started = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="running")
    st.write_bytes("evals/t/runs/e_BAD/_meta.json", b"{ not json")
    st.write_bytes("datasets/d/ingests/ing_BAD.json", b"{ not json")

    assert [m["eval_id"] for m in evals.pending_runs(st, "t")] == [started]
    assert len(ds.list_ingests(st, "d")) == 1


class _NeverCalled:
    """An evaluator the failure above must reach before asking anything of it."""

    def score(self, ctx):                     # pragma: no cover - never reached
        raise AssertionError("scoring should not have started")


def test_a_tombstone_after_a_snapshot_does_not_reach_a_rescore(tmp_path):
    """A run is pinned to a set of files, and a delete is a new file outside that set.

    `rescore` reads the same predictions a different way, so it carries the original
    snapshot over. If a later tombstone reached it, the same `snapshot_id` would report
    two different numbers and comparability would mean nothing.
    """
    st = _st(tmp_path)
    ds.create_dataset(st, DS)
    ds.append(st, "d", [{"id": "a", "v": 10}, {"id": "b", "v": 20}, {"id": "c", "v": 60}])
    evals.create_eval_table(st, "name: t\ncolumns: [{name: n_rows, type: float}]\n")
    eid = evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="before the delete")
    for sid in evals.target_ids(st, "t", eid):
        evals.write_prediction(st, "t", eid, sid, b"{}", "json")
    first = evals.score_run(st, "t", eid, _CountsRows())

    ds.delete_rows(st, "d", ["c"])
    again = evals.get_run(st, "t", evals.rescore(st, "t", eid, _CountsRows(),
                                                 evaluator_version="v2"))["row"]

    assert first["n_rows"] == again["n_rows"] == 3.0
    assert first["snapshot_id"] == again["snapshot_id"]          # same data, same id
    # A run started after the delete sees the delete, and says so with a different id.
    later = evals.get_snapshot(st, "t", evals.create_run(
        st, "t", dataset="d", evaluator_version="v1", title="after the delete"))
    assert later["row_count"] == 2 and later["snapshot_id"] != first["snapshot_id"]


class _CountsRows:
    """Reports how many rows the run was pinned to, so a stray tombstone would show."""

    def score(self, ctx):
        from cairn.core.records import EvalResult

        return EvalResult(row={"n_rows": float(len(ctx.dataset.rows()))}, metrics=[], report_md="")


def test_a_snapshot_id_is_the_content_and_not_the_history(tmp_path):
    """Rows that were deleted and rows that never existed leave the same content.

    Folding the deleted keys into the hash would make a delete-then-reingest differ
    from a dataset that simply holds those rows, which is the comparison the two axes
    exist to make.
    """
    made = itertools.count()

    def rows_of(*ops) -> str:
        st = _st(tmp_path / f"store{next(made)}")
        ds.create_dataset(st, DS)
        for op in ops:
            op(st)
        return ds.snapshot(st, "d")["snapshot_id"]

    add_ab = lambda st: ds.append(st, "d", [{"id": "a", "v": 1}, {"id": "b", "v": 2}])
    add_c = lambda st: ds.append(st, "d", [{"id": "c", "v": 3}])
    drop_c = lambda st: ds.delete_rows(st, "d", ["c"])

    assert rows_of(add_ab) == rows_of(add_ab, add_c, drop_c)      # never had it == deleted it
    assert rows_of(add_ab, add_c) == rows_of(add_ab, add_c, drop_c, add_c)   # and back again
    assert rows_of(add_ab) != rows_of(add_ab, add_c)              # a delete does move the id
