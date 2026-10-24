"""Pipeline: hands a started run to wherever its inference runs, and gathers what came back.

Optional, one per table (`evals/<name>/pipeline.py`). With it, starting a run submits the
work, and Evaluate gathers the predictions before scoring them, so a table goes from
"New run" to a result without anyone writing predictions by hand. cairn knows nothing about
the place the work runs: Dagster, Airflow, a batch queue, a shell script — the class does.

`submit` returns as soon as the work is handed over; nobody waits for it here. Whoever
watches the work decides when it is done and presses Evaluate. `collect` is optional: a
pipeline that writes its predictions itself (`cairn eval put-prediction`) leaves it out.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


class NotReady(Exception):
    """The work has not finished. Raised by `collect`; the run stays pending and nothing is scored."""


@runtime_checkable
class Pipeline(Protocol):
    def submit(self, eval_id: str, sample_ids: list[str], config: dict[str, Any]) -> None:
        """Hand the run over. Tag the work with `eval_id` to trace it back."""
        ...


@runtime_checkable
class CollectingPipeline(Pipeline, Protocol):
    def collect(self, eval_id: str, sample_ids: list[str], config: dict[str, Any]) -> Iterable[tuple[str, bytes]]:
        """The predictions as `(sample_id, bytes)`, or `NotReady` if the work is still going."""
        ...
