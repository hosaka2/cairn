"""End-to-end smoke test of the core: a lock-free append-only log plus snapshots.

Covers:
  - dataset append (upsert correction, tombstone delete) and DuckDB paging
  - snapshot reproducibility: a run pins the dataset key set at creation time
  - prediction and scoring are separate: an adapter writes predictions, an evaluator scores them
  - vacuum reclaims superseded checkpoints
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from cairn.adapters.local import InlineAdapter
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.records import EvalResult, Metric, RunSpec
from cairn.core.storage import Storage

DATASET_SCHEMA = """
name: demo
description: smoke test fixture
key: id
columns:
  - {name: id, type: str, required: true}
  - {name: label, type: int, required: true}
"""

TABLE_SCHEMA = """
name: demo-acc
description: accuracy
columns:
  - {name: accuracy, type: float}
  - {name: coverage, type: float}
"""


class DemoEval:
    class Config(BaseModel):
        pass

    def score(self, ctx) -> EvalResult:
        gt = {r["id"]: r["label"] for r in ctx.dataset.rows()}
        correct = n = 0
        for sid, data in ctx.predictions.iter():
            pred = json.loads(data)["pred"]
            n += 1
            correct += int(pred == gt.get(sid))
        acc = correct / n if n else 0.0
        cov = ctx.actual_n / ctx.expected_n if ctx.expected_n else 0.0
        return EvalResult(
            row={"accuracy": round(acc, 3), "coverage": round(cov, 3)},
            metrics=[Metric(name="accuracy", value=acc), Metric(name="coverage", value=cov)],
            report_md=f"# demo\naccuracy={acc:.3f}",
        )


def test_end_to_end(tmp_path):
    st = Storage(f"file://{tmp_path}/cairn")

    # --- dataset: append (one row fails the schema), upsert correction, tombstone delete ---
    ds.create_dataset(st, DATASET_SCHEMA, created_by="tester")
    rec = ds.append(
        st, "demo",
        [{"id": "a", "label": 1}, {"id": "b", "label": 0}, {"id": "bad", "label": "x"}],
        created_by="tester",
    )
    assert rec["n_ops"] == 2 and rec["n_rejected"] == 1
    ds.append(st, "demo", [{"id": "c", "label": 1}], created_by="tester")
    ds.append(st, "demo", [{"id": "a", "label": 0}], created_by="tester")   # upsert correction
    assert ds.count_rows(st, "demo") == 3
    assert {r["id"] for r in ds.read_rows(st, "demo")} == {"a", "b", "c"}
    assert ds.merge(st, "demo")["a"]["label"] == 0                          # last write wins

    # DuckDB paging (LIMIT/OFFSET over manifest.parquet)
    page = ds.rows_page(st, "demo", limit=2, offset=0)
    assert len(page) == 2

    # tombstone delete
    ds.delete_rows(st, "demo", ["b"], created_by="tester")
    assert ds.count_rows(st, "demo") == 2

    # --- eval: pin a snapshot, write predictions, score ---
    evals.create_eval_table(st, TABLE_SCHEMA)
    eid = evals.create_run(
        st, "demo-acc", dataset="demo", evaluator_version="v1",
        title="smoke", created_by="tester", config={},
    )
    snap = evals.get_snapshot(st, "demo-acc", eid)
    assert snap["row_count"] == 2 and len(snap["snapshot_id"]) == 12

    preds = {"a": 0, "c": 1}   # both correct, so accuracy = 1.0
    adapter = InlineAdapter(st, "demo-acc", eid,
                            jobs={"infer": lambda sid, cfg: json.dumps({"pred": preds[sid]}).encode()})
    adapter.submit(RunSpec(sample_ids=["a", "c"], job_name="infer"), tags={})
    assert evals.count_predictions(st, "demo-acc", eid) == 2

    row = evals.score_run(st, "demo-acc", eid, DemoEval())
    assert row["accuracy"] == 1.0
    assert row["evaluator_version"] == "v1"
    assert row["snapshot_id"] == snap["snapshot_id"]
    assert row["status"] == "scored"

    # listing via DuckDB glob, with no index to maintain
    listed = evals.list_runs(st, "demo-acc")
    assert len(listed) == 1 and listed[0]["eval_id"] == eid
    assert evals.count_runs(st, "demo-acc") == 1

    # vacuum reclaims all but the widest checkpoint, and never touches the data.
    for i in range(ds.CHECKPOINT_EVERY * 2):
        ds.append(st, "demo", [{"id": f"x{i}", "label": 1}], created_by="tester")
    assert len(ds._checkpoints(st, "demo")) >= 2
    assert ds.vacuum(st, "demo")["removed_checkpoints"] == 0      # nothing is old enough yet
    res = ds.vacuum(st, "demo", keep=1, min_age_days=0)
    assert res["removed_checkpoints"] >= 1
    assert len(ds._checkpoints(st, "demo")) == 1
    assert ds.count_rows(st, "demo") == 2 + ds.CHECKPOINT_EVERY * 2


def test_snapshot_is_content_hash(tmp_path):
    """snapshot_id hashes content: identical re-ingest keeps it, a changed value moves it."""
    st = Storage(f"file://{tmp_path}/cairn")
    ds.create_dataset(st, DATASET_SCHEMA, created_by="t")
    ds.append(st, "demo", [{"id": "a", "label": 1}, {"id": "b", "label": 0}], created_by="t")
    s1 = ds.snapshot(st, "demo")["snapshot_id"]

    # append the same rows again: more log files, same content, same snapshot_id
    ds.append(st, "demo", [{"id": "a", "label": 1}, {"id": "b", "label": 0}], created_by="t")
    assert ds.snapshot(st, "demo")["snapshot_id"] == s1

    # upsert a new value: different snapshot_id
    ds.append(st, "demo", [{"id": "a", "label": 0}], created_by="t")
    assert ds.snapshot(st, "demo")["snapshot_id"] != s1


def test_inline_adapter_skips_none(tmp_path):
    """A job that returns None writes no prediction, so partial coverage is allowed."""
    st = Storage(f"file://{tmp_path}/cairn")
    ds.create_dataset(st, DATASET_SCHEMA, created_by="t")
    ds.append(st, "demo", [{"id": "a", "label": 1}, {"id": "b", "label": 0}], created_by="t")
    evals.create_eval_table(st, TABLE_SCHEMA)
    eid = evals.create_run(st, "demo-acc", dataset="demo",
                           evaluator_version="v1", title="t1", created_by="t", config={})
    # "b" produces None
    adapter = InlineAdapter(st, "demo-acc", eid, jobs={
        "infer": lambda sid, cfg: None if sid == "b" else json.dumps({"pred": 1}).encode()})
    adapter.submit(RunSpec(sample_ids=["a", "b"], job_name="infer"), tags={})
    assert evals.count_predictions(st, "demo-acc", eid) == 1


NESTED_SCHEMA = """
name: clips
key: clip_id
columns:
  - {name: clip_id, type: str, required: true}
nested:
  steps:
    schema:
      - {name: frame, type: int}
      - {name: x, type: float}
"""


def test_nested_frames_delivered(tmp_path):
    """Nested columns are hidden from the scalar listing and read via ctx.dataset.frames()."""
    st = Storage(f"file://{tmp_path}/cairn")
    ds.create_dataset(st, NESTED_SCHEMA, created_by="t")
    ds.append(st, "clips", [{"clip_id": "f1", "steps": [{"frame": 0, "x": 1.0}, {"frame": 1, "x": 2.0}]}],
              created_by="t")
    # the scalar listing omits steps
    page = ds.rows_page(st, "clips", limit=10)
    assert page and "steps" not in page[0]

    evals.create_eval_table(st, TABLE_SCHEMA.replace("demo-acc", "clip-acc"))
    eid = evals.create_run(st, "clip-acc", dataset="clips",
                           evaluator_version="v1", title="t1", created_by="t", config={})

    seen = {}

    class NestedEval:
        class Config(BaseModel):
            pass

        def score(self, ctx) -> EvalResult:
            seen["pts"] = ctx.dataset.frames("f1")
            return EvalResult(row={"accuracy": 1.0, "coverage": 1.0})

    evals.score_run(st, "clip-acc", eid, NestedEval())
    assert len(seen["pts"]) == 2 and seen["pts"][0]["frame"] == 0


def test_the_demo_runs_end_to_end_and_is_idempotent(tmp_path):
    """`cairn demo-seed` on a store that already has the demo must not double it."""
    from cairn.demo import pipeline

    st = Storage(f"file://{tmp_path}/cairn")
    pipeline.seed(st)
    counts = {t: evals.count_runs(st, t) for t in evals.list_eval_tables(st)}
    rows = {d: ds.count_rows(st, d) for d in ds.list_datasets(st)}

    pipeline.seed(st)

    assert {t: evals.count_runs(st, t) for t in evals.list_eval_tables(st)} == counts
    assert {d: ds.count_rows(st, d) for d in ds.list_datasets(st)} == rows


def test_the_demo_evaluator_leaves_unpredicted_samples_out_of_the_metrics(tmp_path):
    """Coverage says how much was predicted; the metrics describe only what was."""
    from cairn.demo.evaluate import AnomalyEvalV2
    from cairn.demo.ingest import SyntheticIngest
    from cairn.demo.pipeline import dataset_schema
    from cairn.interfaces.ingestor import IngestContext

    st = Storage(f"file://{tmp_path}/cairn")
    ds.create_dataset(st, dataset_schema("scans", "A"), created_by="t")
    ctx = IngestContext(dataset="scans", tmpdir=str(tmp_path), created_by="t")
    rows = list(SyntheticIngest().ingest(SyntheticIngest.Input(n=6, seed=3, site="A"), ctx))
    ds.append(st, "scans", rows, created_by="t", input={})

    evals.create_eval_table(st, "name: t\ncolumns: [{name: f1, type: float}, "
                                "{name: precision, type: float}, {name: recall, type: float}, "
                                "{name: accuracy, type: float}]\n")
    eid = evals.create_run(st, "t", dataset="scans", evaluator_version="v2", title="partial")
    for row in rows[:2]:                       # only two of the six get a prediction
        evals.write_prediction(st, "t", eid, row["sample_id"], json.dumps({"pred": 1}).encode())

    evals.score_run(st, "t", eid, AnomalyEvalV2())

    assert "coverage 33%" in evals.get_run(st, "t", eid)["report_md"]
    assert len([m for m in evals.read_metrics(st, "t", eid) if m["name"] == "correct"]) == 2
