"""Drives the whole demo, going through every interface for real (nothing faked).

  Ingestor(SyntheticIngest) -> dataset.append/upsert/delete/commit
  Runner(ChunkRunner).plan -> OrchestratorAdapter(InlineAdapter).submit(process_one)
  -> Evaluator(v1/v2).score -> row/metrics/report/assets

`cairn demo-seed` calls this. The metrics really move with the config, and v1/v2 compute
them differently.
"""

from __future__ import annotations

from typing import Any

from cairn.adapters.local import InlineAdapter
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.storage import Storage
from cairn.demo.evaluate import AnomalyEvalV1, AnomalyEvalV2
from cairn.demo.ingest import SyntheticIngest
from cairn.demo.model import ChunkRunner, make_process_one
from cairn.interfaces.ingestor import IngestContext

# The demo's own definitions. `cairn demo-init` writes these same strings out as
# convention files, so the generated scripts and the seeded store cannot drift apart.
SCHEMA = """name: {name}
kind: sensor
description: {desc}
key: sample_id
columns:
  - {{name: sample_id, type: str, required: true}}
  - {{name: site, type: str}}
  - {{name: temp, type: float}}
  - {{name: vibration, type: float}}
  - {{name: pressure, type: float}}
  - {{name: calibrated, type: bool}}
  - {{name: gt, type: int, required: true}}
"""

TABLE = """name: {name}
description: {desc}
columns:
  - {{name: f1, type: float, display: "F1", primary: true, direction: higher, scale: [0, 1]}}
  - {{name: precision, type: float, display: "precision"}}
  - {{name: recall, type: float, display: "recall"}}
  - {{name: accuracy, type: float, display: "accuracy"}}
default_sort: created_at desc
"""

# (dataset, site, rows, seed) and (eval table, description, dataset it targets).
DATASETS = (("sensor-anomaly-A", "A", 48, 7), ("sensor-anomaly-B", "B", 36, 11))
TABLES = (("anomaly", "Sensor anomaly detection at site A. Precision, recall, F1.", "sensor-anomaly-A"),
          ("anomaly-siteB", "Sensor anomaly detection at site B.", "sensor-anomaly-B"))


def dataset_schema(name: str, site: str) -> str:
    return SCHEMA.format(name=name, desc=f"Sensor scans at site {site}.")


def eval_table(name: str, desc: str) -> str:
    return TABLE.format(name=name, desc=desc)


def seed(st: Storage) -> None:
    """Builds the demo data through the real pipeline. Existing datasets and tables are skipped."""
    for name, site, n, seed_ in DATASETS:
        _build_dataset(st, name, site=site, n=n, seed=seed_)
    for table, desc, dataset in TABLES:
        _build_table(st, table, desc, dataset)


def _build_dataset(st: Storage, name: str, *, site: str, n: int, seed: int) -> str:
    if name in ds.list_datasets(st):
        return name
    ds.create_dataset(st, dataset_schema(name, site), created_by="seed")

    # Rows come from the real Ingestor, appended in two batches to leave an ingest history.
    ing = SyntheticIngest()
    ctx = IngestContext(dataset=name, tmpdir="/tmp", created_by="seed")
    rows = list(ing.ingest(SyntheticIngest.Input(n=n, seed=seed, site=site), ctx))
    ds.append(st, name, rows[:-6], created_by="seed", input={"ingestor": "SyntheticIngest", "n": n - 6})
    ds.append(st, name, rows[-6:], created_by="seed", input={"ingestor": "SyntheticIngest", "n": 6})

    # Append-only edits: a label fix (upsert) and a bad sample removal (tombstone).
    fixed = dict(rows[0])
    fixed["gt"] = 1 - fixed["gt"]
    ds.append(st, name, [fixed], created_by="seed", input={"note": "label fix"})
    ds.delete_rows(st, name, [rows[1]["sample_id"]], created_by="seed")
    return name


def _build_table(st: Storage, table: str, desc: str, dataset: str) -> None:
    if table in evals.list_eval_tables(st):
        return
    evals.create_eval_table(st, eval_table(table, desc))

    rows = ds.read_rows(st, dataset)
    rows_by_id = {r["sample_id"]: r for r in rows}
    sample_ids = list(rows_by_id)

    # (evaluator_version, model config, title, comment) - oldest first, newest ends up on top.
    plan: list[tuple[str, dict[str, Any], str, str]] = [
        ("v1", {"threshold": 1.1, "w_vib": 1.0}, "First baseline", "High threshold to start"),
        ("v1", {"threshold": 0.7, "w_vib": 1.2}, "Lower the threshold", ""),
        ("v2", {"threshold": 0.7, "w_vib": 1.2}, "Switch to v2", "Proper F1; v1 not comparable"),
        ("v2", {"threshold": 0.5, "w_vib": 1.3}, "More sensitive", ""),
        ("v2", {"threshold": 0.55, "w_vib": 1.3, "w_temp": 0.03}, "Add temperature weight",
         "Check whether false positives drop"),
    ]
    for ev, model_cfg, title, comment in plan:
        _run_and_score(st, table, dataset, rows_by_id, sample_ids, ev, model_cfg, title, comment)


def _run_and_score(st, table, dataset, rows_by_id, sample_ids, ev, model_cfg, title, comment) -> None:
    eid = evals.create_run(
        st, table, dataset=dataset, evaluator_version=ev,
        title=title, comment=comment, created_by="seed", config=model_cfg,
    )
    # The Runner plans the chunks; InlineAdapter runs process_one in place and writes predictions.
    runner = ChunkRunner()
    specs = runner.plan(sample_ids, ChunkRunner.Config(chunk_size=16, model=model_cfg))
    adapter = InlineAdapter(st, table, eid, jobs={"infer": make_process_one(rows_by_id)})
    for spec in specs:
        adapter.submit(spec, tags={"cairn/eval_id": eid})
    # The scoring step: predictions are aggregated into metrics.
    evaluator = AnomalyEvalV1() if ev == "v1" else AnomalyEvalV2()
    evals.score_run(st, table, eid, evaluator)
