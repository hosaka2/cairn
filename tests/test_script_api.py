"""What a script is handed: `cairn.io` and the two contexts.

Ingest and eval scripts never touch storage directly; they read through `ctx`, which
is the same file API for `file://`, `s3://` and `http(s)://`. These tests use `file://`
and plain paths, the two forms a local run actually produces.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from cairn import io
from cairn.interfaces.evaluator import DatasetView, EvalContext, PredictionView
from cairn.interfaces.ingestor import FileRef, IngestContext, S3Path

# --- cairn.io ---------------------------------------------------------------

def test_bytes_written_come_back_unchanged(tmp_path):
    uri = f"file://{tmp_path}/blob.bin"
    io.write_bytes(uri, b"\x00\xffbytes")
    assert io.read_bytes(uri) == b"\x00\xffbytes"


def test_text_is_decoded_with_the_encoding_asked_for(tmp_path):
    uri = f"file://{tmp_path}/t.txt"
    io.write_bytes(uri, "石".encode("shift_jis"))
    assert io.read_text(uri, encoding="shift_jis") == "石"


def test_open_is_a_context_manager(tmp_path):
    p = tmp_path / "o.txt"
    p.write_text("hello", encoding="utf-8")
    with io.open(f"file://{p}", "rt") as f:
        assert f.read() == "hello"


# --- IngestContext ----------------------------------------------------------

def _ingest_ctx(tmp_path) -> IngestContext:
    return IngestContext(dataset="d", tmpdir=str(tmp_path), created_by="tester")


def test_ingest_context_reads_a_source_file(tmp_path):
    (tmp_path / "src.jsonl").write_bytes(b'{"id": "a"}\n')
    ctx = _ingest_ctx(tmp_path)
    uri = f"file://{tmp_path}/src.jsonl"

    assert ctx.read_bytes(uri) == b'{"id": "a"}\n'
    assert ctx.read_text(uri).strip() == '{"id": "a"}'
    with ctx.open(uri) as f:
        assert f.read().startswith(b"{")


def test_file_ref_and_s3_path_are_strings_a_pydantic_input_can_declare(tmp_path):
    class Input(BaseModel):
        upload: FileRef = FileRef("")
        source: S3Path = S3Path("")

    # The values arrive as plain strings from the form; the field types convert them.
    inp = Input.model_validate({"upload": str(tmp_path / "up.csv"), "source": "s3://bucket/key"})

    assert isinstance(inp.upload, FileRef) and isinstance(inp.source, S3Path)
    assert inp.upload.as_path() == Path(tmp_path / "up.csv")
    assert inp.source == "s3://bucket/key"


# --- EvalContext ------------------------------------------------------------

ROWS = [{"id": "a", "gt": 1}, {"id": "b", "gt": 0}]


class _Config(BaseModel):
    """Stands in for the evaluator's own Config."""


def _eval_ctx(frames_loader=None) -> EvalContext:
    return EvalContext(
        dataset=DatasetView(ROWS, "id", frames_loader=frames_loader),
        predictions=PredictionView(lambda: iter([("a", b"1")])),
        config=_Config(), expected_n=2, actual_n=1,
    )


def test_dataset_view_reads_rows_by_key():
    view = _eval_ctx().dataset
    assert view.targets() == ["a", "b"]
    assert view.rows() == ROWS
    assert view.row("b") == {"id": "b", "gt": 0}


def test_an_unknown_sample_id_is_none_rather_than_an_error():
    """An evaluator asking about a sample the snapshot lacks gets an answer, not a crash."""
    assert _eval_ctx().dataset.row("nope") is None


def test_frames_are_empty_when_the_schema_declares_no_nested_table():
    assert _eval_ctx().dataset.frames("a") == []


def test_frames_pass_the_field_through_to_the_loader():
    seen: list[tuple[str, str | None]] = []

    def loader(sample_id: str, field: str | None):
        seen.append((sample_id, field))
        return [{"t": 0.0}]

    assert _eval_ctx(loader).dataset.frames("a", "readings") == [{"t": 0.0}]
    assert seen == [("a", "readings")]


def test_predictions_are_iterated_from_the_loader():
    assert list(_eval_ctx().predictions.iter()) == [("a", b"1")]


def test_eval_context_reads_a_referenced_asset(tmp_path):
    """Heavy assets stay out of the ledger: a column holds the URL and the evaluator reads it."""
    (tmp_path / "mask.png").write_bytes(b"\x89PNG")
    ctx, uri = _eval_ctx(), f"file://{tmp_path}/mask.png"

    assert ctx.read_bytes(uri) == b"\x89PNG"
    assert ctx.read_text(f"file://{tmp_path}/mask.png", encoding="latin-1").startswith("\x89")
    with ctx.open(uri) as f:
        assert f.read() == b"\x89PNG"
