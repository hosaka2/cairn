"""Runs inference in this process, without an orchestrator.

Jobs are registered by name, executed as soon as they are submitted, and their output is
written to the run's predictions. Submitting the same RunSpec to another adapter would
run the identical work elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cairn.core import evals
from cairn.core.records import RunSpec, RunStatus
from cairn.core.storage import Storage

# process_one(sample_id, run_config) -> bytes | None
#   bytes: the prediction for that sample. None: this run produced nothing for it, which
#   is normal, and leaves the sample out of the evaluation instead of failing the job.
ProcessFn = Callable[[str, dict[str, Any]], "bytes | None"]


class InlineAdapter:
    """Runs a registered job immediately and stores what it returns."""

    def __init__(self, st: Storage, table: str, eval_id: str, jobs: dict[str, ProcessFn], ext: str = "bin") -> None:
        self._st = st
        self._table = table
        self._eval_id = eval_id
        self._jobs = jobs
        self._ext = ext
        self._runs: dict[str, RunStatus] = {}

    def submit(self, spec: RunSpec, tags: dict[str, str]) -> str:
        fn = self._jobs.get(spec.job_name)
        if fn is None:
            raise KeyError(f"job '{spec.job_name}' is not registered")
        run_id = f"inline:{spec.job_name}:{spec.sample_ids[0] if spec.sample_ids else '-'}"
        try:
            # The chunk is inferred first and written as a chunk: one round trip per
            # sample is what a remote root would charge for writing them one by one.
            done = [(sid, data) for sid in spec.sample_ids
                    if (data := fn(sid, spec.run_config)) is not None]
            evals.write_predictions(self._st, self._table, self._eval_id, done, self._ext)
            self._runs[run_id] = RunStatus(run_id=run_id, state="success")
        except Exception:
            self._runs[run_id] = RunStatus(run_id=run_id, state="failure")
            raise
        return run_id

    def status(self, run_ids: list[str]) -> dict[str, RunStatus]:
        return {r: self._runs.get(r, RunStatus(run_id=r, state="unknown")) for r in run_ids}
