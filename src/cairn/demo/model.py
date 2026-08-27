"""Example inference unit (process_one) and Runner.

`process_one(sample_id, run_config)` is meant to be the same per-sample function used in
production; the Runner only decides how many samples go into one chunk. During evaluation
the InlineAdapter runs it in place.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from cairn.core.records import RunSpec


class ModelConfig(BaseModel):
    """Threshold linear model settings. Extra keys are ignored so the eval config can be shared."""

    threshold: float = 0.6
    w_temp: float = 0.02
    w_vib: float = 1.2
    w_pressure: float = 0.05

    model_config = {"extra": "ignore"}


def anomaly_score(row: dict[str, Any], cfg: ModelConfig) -> float:
    return (cfg.w_temp * (row["temp"] - 60)
            + cfg.w_vib * (row["vibration"] - 0.40)
            + cfg.w_pressure * (row["pressure"] - 101))


def make_process_one(rows_by_id: dict[str, dict[str, Any]]) -> Callable[[str, dict[str, Any]], bytes]:
    """Returns a process_one mapping sample_id to prediction bytes, closing over the rows."""

    def process_one(sample_id: str, run_config: dict[str, Any]) -> bytes:
        cfg = ModelConfig(**run_config)
        s = anomaly_score(rows_by_id[sample_id], cfg)
        return json.dumps({"pred": int(s > cfg.threshold), "score": round(s, 4)}).encode()

    return process_one


class ChunkRunner:
    """Runner that groups samples into chunks and builds RunSpecs. It does not execute them."""

    class Config(BaseModel):
        chunk_size: int = 16
        model: dict = {}

    def plan(self, sample_ids: list[str], cfg: ChunkRunner.Config) -> list[RunSpec]:
        return [
            RunSpec(sample_ids=sample_ids[i:i + cfg.chunk_size], job_name="infer", run_config=cfg.model)
            for i in range(0, len(sample_ids), cfg.chunk_size)
        ]
