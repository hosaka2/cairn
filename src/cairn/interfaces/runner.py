"""Runner: decides how inference work is split into units.

It plans only. The work itself is done by `process_one(sample_id, run_config)`, which
is the same function production uses. Grouping samples into chunks keeps the number of
submitted jobs manageable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from cairn.core.records import RunSpec


@runtime_checkable
class Runner(Protocol):
    """Plans the units of work for a set of samples."""

    class Config(BaseModel):
        ...

    def plan(self, sample_ids: list[str], cfg: Runner.Config) -> list[RunSpec]:
        ...
