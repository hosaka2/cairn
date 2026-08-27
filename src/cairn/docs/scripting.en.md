# Writing scripts

All cairn fixes is the **location, the export names, and a minimal shape**. The ingest and evaluation logic is up to you.
Once written, `cairn check` inspects the contract (it fails before registration, so you are not left guessing at the cause).

```bash
cairn init                        # create datasets/ and evals/
cairn new dataset sensor-scans    # schema.yaml + ingest.py
cairn new eval scan-anomaly       # table.yaml + run.py + v1.py
cairn check                       # inspect the contract (do this first)
```

## Directory convention

```
datasets/<name>/
  schema.yaml     column definitions (the dataset name must match <name>)
  input.py        class Input(BaseModel) — the form's fields. **Read, never imported**
  ingest.py       INGESTOR = <class>
evals/<name>/
  table.yaml      definitions of the columns shown in the list view
  config.py       class Config(BaseModel) — the run form's fields, read the same way
  run.py          RUNNER / PROCESS_FACTORY / CONFIG
  v1.py, v2.py …  EVALUATOR = <class> (file name = version of the evaluation method)
```

Heavy imports belong in `ingest.py` / `run.py`. The web draws a form from `input.py` / `config.py` without importing anything — not by convention, by there being no code path that runs them. Whether the two name the same fields is what `cairn check` looks at.

These two directories are scanned and registered automatically at startup (`CAIRN_SCRIPTS`, defaults to the current directory).
Only files named **`v` + digits** (`v1.py`, `v2.py`) are read as eval versions.

---

## schema.yaml (dataset columns)

```yaml
name: sensor-scans        # must match datasets/<name>/
kind: sensor              # display label for the kind (optional)
description: description
key: sample_id            # row identity. Re-appending the same key overwrites (last write wins)
columns:
  - {name: sample_id,    type: str, required: true}
  - {name: site,         type: str}
  - {name: calibrated,   type: bool}
  - {name: waveform_url, type: str}     # hold only a reference for heavy assets
  - {name: gt,           type: int, required: true}
nested:                               # child tables not shown in the list view (a time series, etc.)
  readings:
    schema:
      - {name: t,     type: float}
      - {name: value, type: float}
```

- `type`: `str` / `int` / `float` / `bool` / `datetime` / `s3path` / `list[float]` / `list[int]` /
  `list[str]` / `json`
- A row missing a column marked `required: true` is rejected at ingest.
- **Do not put heavy things (raw waveforms, images, masks, point clouds) in the ledger.** Give the column a URL/path and read it
  on the evaluation side with `ctx.read_bytes(url)`. As a principle, **do not hold values that can be derived** from the columns.
- `nested` goes directly inside the row and is retrieved at evaluation time with `ctx.dataset.frames(sample_id)`.

## table.yaml (eval list columns)

```yaml
name: scan-anomaly
dataset: sensor-scans   # the default target (preset in the run form, changeable at run time)
columns:
  - {name: f1,        type: float, display: "F1", primary: true, direction: higher, scale: [0, 1]}
  - {name: miss_rate, type: float, display: "Missed%", direction: lower}
  - {name: coverage,  type: float, display: "Coverage%"}
default_sort: created_at desc
```

- The column with `primary: true` is the **primary metric** = the stone width (exactly one column; if omitted, the first column).
- `direction`: `higher` (bigger is better) / `lower` (smaller is better, e.g. error). Sets the comparison color and the direction of the stone width.
- `scale: [min, max]`: the absolute range of the stone width. Setting it keeps small differences from being exaggerated.
- **The `row` the evaluator returns must match these columns exactly** (extra columns are an error). Common metadata such as
  `eval_id` / `snapshot_id` / `evaluator_version` is added automatically by cairn.

---

## Ingestor (ingest)

```python
# datasets/sensor-scans/ingest.py
from pydantic import BaseModel
from cairn.interfaces.ingestor import IngestContext

class Ingest:
    class Input(BaseModel):        # ← the UI input fields are generated from this
        source: str = ""

    execution = "inline"

    def ingest(self, inp: "Ingest.Input", ctx: IngestContext):
        raw = ctx.read_text(inp.source)          # s3:// gs:// file:// http(s):// through the same API
        for line in raw.splitlines():
            yield {"sample_id": ..., "site": ...}      # yield dicts conforming to schema.yaml

INGESTOR = Ingest
```

**Contract**: assign a class to `INGESTOR` / `Input` must be a pydantic `BaseModel` /
`ingest(self, inp, ctx)` must yield dicts conforming to the schema.

Available on `ctx`: `read_text` `read_bytes` `open` (backend-independent; no boto3 needed),
`dataset` `tmpdir` `created_by`. External connections such as Postgres are made by the script with its own dependencies.

## Runner / process_one (inference)

```python
# evals/scan-anomaly/run.py
import json
from pydantic import BaseModel
from cairn.core.records import RunSpec

class Config(BaseModel):           # ← the "config" field of the run form. Changeable at run time
    threshold: float = 0.5
    model_config = {"extra": "ignore"}

def make_process_one(rows_by_id):
    def process_one(sample_id: str, run_config: dict) -> bytes | None:
        cfg = Config(**run_config)
        row = rows_by_id[sample_id]
        if is_out_of_scope(row):
            return None                       # ← no prediction = outside the evaluation (not a failure)
        return json.dumps({"pred": ...}).encode()
    return process_one

class Runner:
    class Config(BaseModel):
        chunk_size: int = 16
        model: dict = {}

    def plan(self, sample_ids: list[str], cfg: "Runner.Config") -> list[RunSpec]:
        return [RunSpec(sample_ids=sample_ids[i:i + cfg.chunk_size], job_name="infer",
                        run_config=cfg.model)
                for i in range(0, len(sample_ids), cfg.chunk_size)]

RUNNER = Runner
PROCESS_FACTORY = make_process_one
CONFIG = Config
```

**Contract**: `RUNNER` has `Config` (pydantic) and `plan(sample_ids, cfg) -> list[RunSpec]` /
`PROCESS_FACTORY` is a function that returns `process_one(sample_id, run_config) -> bytes | None` /
`CONFIG` is a pydantic model.

- When running from the web, `RunSpec.job_name` must be **`"infer"`**.
- The contents of a prediction are **opaque bytes**. The format is free (JSON, parquet, anything). Only the evaluator reads it.
- **Returning `None` writes no prediction.** This expresses "that target has no artifact" without failing.

## Evaluator (evaluation)

```python
# evals/scan-anomaly/v1.py
from cairn.core.records import EvalResult, Metric

class Eval:
    def score(self, ctx) -> EvalResult:
        gt = {r["sample_id"]: r for r in ctx.dataset.rows()}
        total = 0.0
        for sample_id, data in ctx.predictions.iter():     # (id, bytes)
            ...
        return EvalResult(
            row={"f1": 0.882, "miss_rate": 6.3, "coverage": 98.3},      # matches table.yaml
            metrics=[Metric(name="f1", value=0.882),
                     Metric(name="f1", value=0.0, sample_id="A_014")],
            report_md="## Results\n\n…",           # the details shown on screen
            assets={"plot.svg": b"<svg .../>"},    # referenced from report.md as ![](assets/plot.svg)
            metadata={"by_site": {"A": 0.88}},     # free-form JSON, not turned into columns
        )

EVALUATOR = Eval
```

**Contract**: assign a class to `EVALUATOR` / `score(self, ctx) -> EvalResult` /
`Config` (pydantic) is optional (if present, the run's `config` is populated into it and passed).

What `ctx` holds:

| | Meaning |
|---|---|
| `ctx.dataset.rows()` | all rows as of the snapshot (GT) |
| `ctx.dataset.row(id)` / `targets()` | one row / the list of ids |
| `ctx.dataset.frames(id)` | the `nested` child tables (a time series, etc.) |
| `ctx.predictions.iter()` | `(sample_id, bytes)` in order |
| `ctx.config` | the run-time config (`Config` type) |
| `ctx.expected_n` / `ctx.actual_n` | number of targets / number of predictions written (coverage) |
| `ctx.read_bytes(url)` etc. | reading referenced assets (raw waveforms, images, etc.) |

Rules to follow:

- **The only input is ctx.** Do not hold hidden mutable state or global side effects (the same input must give the same result).
  Reading referenced assets through `ctx` is a legitimate input path.
- **Aggregate metrics that cannot be decomposed per sample at evaluation time.** mAP / AUC / weighted averages are only
  determined once all predictions are in, so do not collapse them earlier on the inference side; compute them here.
- **Do not edit a version file that has been merged.** To fix a formula, cut a new version with
  `cairn new eval-version <name> v2` (to preserve the meaning of past runs; runs with different versions are not compared side by side).

---

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| The run form shows "not registered" | misspelled export name, import error | see the reason with `cairn check` |
| No input fields appear | `Input` is not pydantic | make it `class Input(BaseModel)` |
| Evaluation fails with `row violates table.yaml` | columns added/removed, type mismatch | make `table.yaml` and `row` match |
| The stone width is the opposite of what you expect | `direction` not set (defaults to higher) | put `direction: lower` on error-type columns |
| Experiments cannot be compared | `snapshot_id` differs | freeze the ledger and only add runs (changing or deleting values changes the snapshot) |
| `v2.py` is not read | the file name is not `v` + digits | name it like `v2.py` |
