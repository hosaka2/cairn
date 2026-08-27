"""Example Ingestor: generates synthetic sensor data (deterministic per seed, no external deps).

The normal and anomalous distributions overlap, so a threshold model cannot separate them
perfectly and the metrics stay below 1.0.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from cairn.interfaces.ingestor import IngestContext


class SyntheticIngest:
    """Ingestor turning input (row count, seed, site) into rows that match the schema."""

    class Input(BaseModel):
        n: int = 40
        seed: int = 1
        site: str = "A"
        anomaly_rate: float = 0.3

    execution = "inline"

    def ingest(self, inp: SyntheticIngest.Input, ctx: IngestContext) -> Iterable[dict[str, Any]]:
        rng = random.Random(inp.seed)
        # Separate stream so adding this column does not shift the sensor values.
        cal_rng = random.Random(inp.seed + 1000)
        for i in range(inp.n):
            is_anom = rng.random() < inp.anomaly_rate
            b = 1.0 if is_anom else 0.0
            yield {
                "sample_id": f"{inp.site}_{i:03d}",
                "site": inp.site,
                "temp": round(rng.gauss(60 + 15 * b, 6.5), 2),
                "vibration": round(rng.gauss(0.40 + 0.50 * b, 0.20), 3),
                "pressure": round(rng.gauss(101 + 4 * b, 2.6), 2),
                "calibrated": cal_rng.random() > 0.3,
                "gt": int(is_anom),
            }
