"""The protocol an orchestrator adapter implements.

cairn itself knows nothing about Dagster, Airflow or anything else: implement these two
methods to submit work wherever you run it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cairn.core.records import RunSpec, RunStatus


@runtime_checkable
class OrchestratorAdapter(Protocol):
    def submit(self, spec: RunSpec, tags: dict[str, str]) -> str:
        """Submit one unit of work and return its id."""
        ...

    def status(self, run_ids: list[str]) -> dict[str, RunStatus]:
        """State of each submitted unit."""
        ...
