"""Generates the convention directory for user scripts (`cairn init` / `cairn new ...`).

Convention:
  datasets/<name>/schema.yaml     column definitions
  datasets/<name>/ingest.py       ingest; assign the Ingestor to the module variable INGESTOR
  evals/<name>/table.yaml         column definitions of the eval table
  evals/<name>/run.py             inference unit; assign RUNNER / PROCESS_FACTORY / CONFIG
  evals/<name>/v1.py, v2.py ...   evaluator; assign EVALUATOR. Frozen once merged (cut a version)

Generating the files keeps locations and export names on the convention, which is what
discovery (registry.discover) relies on to register them automatically.
"""

from __future__ import annotations

import re
from pathlib import Path

# --- Templates --------------------------------------------------------------

_SCHEMA_YAML = """name: {name}
kind: {kind}
description: TODO description
key: id
columns:
  - {{name: id, type: str, required: true}}
  - {{name: gt, type: int, required: true}}
"""

_INGEST_PY = '''"""Ingest for {name}. Yields dicts matching schema.yaml from the input.

cairn discovers the class assigned to `INGESTOR`. The UI form is generated from `Input`.
"""
from pydantic import BaseModel

from cairn.interfaces.ingestor import IngestContext


class Ingest:
    class Input(BaseModel):
        source: str = ""          # e.g. an input file or path. Becomes a field in the UI


    def ingest(self, inp: "Ingest.Input", ctx: IngestContext):
        # ctx.read_text / read_bytes / open read s3:// / gs:// / file:// backend-agnostically.
        # e.g. raw = ctx.read_text(inp.source)
        # External connections such as Postgres are this script's own dependency; cairn passes none.
        # TODO: read inp and yield dicts matching schema.yaml
        yield {{"id": "sample-001", "gt": 0}}


INGESTOR = Ingest
'''

_TABLE_YAML = """name: {name}
description: TODO description
columns:
  # The column with primary: true is the primary metric and sets the stone width (exactly one).
  # direction is the better direction, scale is the absolute range [min, max].
  - {{name: score, type: float, display: "score", primary: true, direction: higher, scale: [0, 1]}}
  - {{name: coverage, type: float, display: "coverage"}}
default_sort: created_at desc
"""

_RUN_PY = '''"""Inference unit for {name}. The Runner only decides the chunking.

Runs the same process_one as production. cairn discovers `RUNNER` / `PROCESS_FACTORY` / `CONFIG`.
"""
import json

from pydantic import BaseModel

from cairn.core.records import RunSpec


class Config(BaseModel):
    """Model settings (becomes the config form in the UI)."""

    threshold: float = 0.5
    model_config = {{"extra": "ignore"}}


def make_process_one(rows_by_id):
    def process_one(sample_id: str, run_config: dict) -> bytes:
        cfg = Config(**run_config)
        row = rows_by_id[sample_id]
        # TODO: the real inference. Produce a prediction from row and cfg
        return json.dumps({{"pred": 0}}).encode()

    return process_one


class Runner:
    class Config(BaseModel):
        chunk_size: int = 16
        model: dict = {{}}

    def plan(self, sample_ids: list[str], cfg: "Runner.Config") -> list[RunSpec]:
        return [
            RunSpec(sample_ids=sample_ids[i:i + cfg.chunk_size], job_name="infer", run_config=cfg.model)
            for i in range(0, len(sample_ids), cfg.chunk_size)
        ]


RUNNER = Runner
PROCESS_FACTORY = make_process_one
CONFIG = Config
'''

_EVAL_PY = '''"""Eval {version} for {name}. Predictions + GT -> a table.yaml row + a report.

Frozen once merged: cut a new version file (v{next}.py) even for a bug fix.
cairn discovers `EVALUATOR`. Metrics that do not decompose per sample, such as mAP or AUC,
are aggregated in score.
"""
from pydantic import BaseModel

from cairn.core.records import EvalResult, Metric


class Eval:
    class Config(BaseModel):
        model_config = {{"extra": "ignore"}}

    def score(self, ctx) -> EvalResult:
        gt = {{r["id"]: r["gt"] for r in ctx.dataset.rows()}}
        # TODO: aggregate metrics from ctx.predictions.iter() and gt
        score = 0.0
        return EvalResult(
            row={{"score": round(score, 3), "coverage": ctx.actual_n / max(ctx.expected_n, 1)}},
            metrics=[Metric(name="score", value=score)],
            report_md="## {name} {version}\\n\\nTODO report.",
        )


EVALUATOR = Eval
'''

_README = """# cairn scripts (convention)

- `datasets/<name>/` - schema.yaml + ingest.py (`INGESTOR`)
- `evals/<name>/` - table.yaml + run.py (`RUNNER`/`PROCESS_FACTORY`/`CONFIG`) + vN.py (`EVALUATOR`)

Generate with `cairn new dataset <name>` / `cairn new eval <name>`.
cairn scans these two directories at startup and registers what it finds
(CAIRN_SCRIPTS defaults to the current directory).
`vN.py` is frozen once merged so past results keep their meaning; cut a new version to fix one.
"""


# --- Generation -------------------------------------------------------------

def _write(path: Path, content: str, *, force: bool) -> str:
    if path.exists() and not force:
        return f"skip (exists) {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"write {path}"


def init(root: str | Path) -> list[str]:
    root = Path(root)
    out = []
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "evals").mkdir(parents=True, exist_ok=True)
    out.append(_write(root / "README.md", _README, force=False))
    return out


def new_dataset(root: str | Path, name: str, *, kind: str = "tabular", force: bool = False) -> list[str]:
    d = Path(root) / "datasets" / name
    return [
        _write(d / "schema.yaml", _SCHEMA_YAML.format(name=name, kind=kind), force=force),
        _write(d / "ingest.py", _INGEST_PY.format(name=name), force=force),
    ]


def new_eval(root: str | Path, name: str, *, force: bool = False) -> list[str]:
    d = Path(root) / "evals" / name
    return [
        _write(d / "table.yaml", _TABLE_YAML.format(name=name), force=force),
        _write(d / "run.py", _RUN_PY.format(name=name), force=force),
        _write(d / "v1.py", _EVAL_PY.format(name=name, version="v1", next="2"), force=force),
    ]


VERSION_RE = re.compile(r"^v(\d+)$")
"""Eval versions are discovered as `v` + digits; anything else would never be read."""


def new_eval_version(root: str | Path, name: str, version: str, *, force: bool = False) -> list[str]:
    m = VERSION_RE.match(version)
    if not m:
        raise ValueError(f"'{version}' is not a version: use v1, v2, … so it can be discovered")
    d = Path(root) / "evals" / name
    nxt = str(int(m.group(1)) + 1)
    return [_write(d / f"{version}.py", _EVAL_PY.format(name=name, version=version, next=nxt), force=force)]


# --- Wire the demo up as convention files -----------------------------------

DOC_IN = '''"""Fields of the ingest form (demo).

cairn reads this file and never imports it, so nothing here can reach the process that
draws the form. Keep it to the declaration; the work goes in ingest.py.
"""
'''

DOC_CFG = '''"""Fields of the run form (demo). Read, never imported — see input.py."""
'''


def _demo_input() -> str:
    return (
        DOC_IN
        + "from pydantic import BaseModel\n\n\n"
        + "class Input(BaseModel):\n"
        + "    n: int = 40\n"
        + "    seed: int = 1\n"
        + '    site: str = "A"\n'
        + "    anomaly_rate: float = 0.3\n"
    )


def _demo_config() -> str:
    return (
        DOC_CFG
        + "from pydantic import BaseModel\n\n\n"
        + "class Config(BaseModel):\n"
        + "    threshold: float = 0.6\n"
        + "    w_temp: float = 0.02\n"
        + "    w_vib: float = 1.2\n"
        + "    w_pressure: float = 0.05\n"
    )


def _demo_ingest(dataset: str) -> str:
    return (f'"""Ingest for {dataset} (demo). Wires up SyntheticIngest."""\n'
            "from cairn.demo.ingest import SyntheticIngest\n\n"
            "INGESTOR = SyntheticIngest\n")


def _demo_run() -> str:
    return ('"""Inference unit (demo). Wires up ChunkRunner + process_one + ModelConfig."""\n'
            "from cairn.demo.model import ChunkRunner, ModelConfig, make_process_one\n\n"
            "RUNNER = ChunkRunner\n"
            "PROCESS_FACTORY = make_process_one\n"
            "CONFIG = ModelConfig\n")


def _demo_eval(cls: str) -> str:
    return (f'"""Eval (demo). Wires up {cls}. Frozen once merged."""\n'
            f"from cairn.demo.evaluate import {cls}\n\n"
            f"EVALUATOR = {cls}\n")


def demo_init(root: str | Path, *, force: bool = False) -> list[str]:
    """Writes the demo out as a convention directory: a working reference wiring.

    The YAML comes from `cairn.demo.pipeline`, the same strings `cairn demo-seed` uses,
    so the generated files describe exactly the data the demo builds.
    """
    from cairn.demo import pipeline as demo

    root = Path(root)
    out = init(root)
    for name, site, _n, _seed in demo.DATASETS:
        d = root / "datasets" / name
        out.append(_write(d / "schema.yaml", demo.dataset_schema(name, site), force=force))
        out.append(_write(d / "input.py", _demo_input(), force=force))
        out.append(_write(d / "ingest.py", _demo_ingest(name), force=force))
    for name, desc, _dataset in demo.TABLES:
        d = root / "evals" / name
        out.append(_write(d / "table.yaml", demo.eval_table(name, desc), force=force))
        out.append(_write(d / "config.py", _demo_config(), force=force))
        out.append(_write(d / "run.py", _demo_run(), force=force))
        out.append(_write(d / "v1.py", _demo_eval("AnomalyEvalV1"), force=force))
        out.append(_write(d / "v2.py", _demo_eval("AnomalyEvalV2"), force=force))
    return out
