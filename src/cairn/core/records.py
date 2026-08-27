"""Types exchanged between cairn and the scripts it runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Metric(BaseModel):
    """One measurement. `sample_id=None` means it covers the whole run."""

    name: str
    value: float
    sample_id: str | None = None
    sub_id: str | None = None


class EvalResult(BaseModel):
    """What `Evaluator.score` returns."""

    row: dict[str, Any]
    """One row for the list view. Its keys must match table.yaml exactly."""

    metrics: list[Metric] = Field(default_factory=list)
    report_md: str = ""
    assets: dict[str, bytes] = Field(default_factory=dict)
    """Files referenced by the report, e.g. `{"plot.svg": b"<svg .../>"}`."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Free-form JSON with no column definition, shown next to the run config.
    Use it for anything that explains the run later: inputs, environment, counts."""


class RunSpec(BaseModel):
    """One unit of inference work, as planned by `Runner.plan`."""

    sample_ids: list[str]
    job_name: str
    run_config: dict[str, Any] = Field(default_factory=dict)


class RunStatus(BaseModel):
    """State of a submitted unit of work."""

    run_id: str
    state: str  # queued / running / success / failure / unknown
