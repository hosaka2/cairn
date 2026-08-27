"""Performing a started run in this process.

A real pipeline runs its inference wherever it runs, and writes predictions against the
`eval_id` cairn handed it. This is the other case: a script registered for the table, small
enough to run right here — the demo, or a script you are trying out.

Reached only from `cairn eval run`: the web never performs a run.
"""

from __future__ import annotations

from typing import Any

from cairn.adapters.local import InlineAdapter
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.storage import Storage


def run_inference(st: Storage, table: str, eval_id: str, inference: dict[str, Any],
                  chunk_size: int = 16) -> int:
    """Write the predictions of a started run with its registered script.

    Reads the snapshot the scoring will read, so inference and evaluation see the same
    rows. Returns how many predictions the run holds afterwards.
    """
    evals.ensure_writable(st, table, eval_id)
    meta = evals.get_meta(st, table, eval_id)
    dataset, config = meta["dataset"], meta.get("config") or {}

    rows = ds.read_rows(st, dataset, evals.get_snapshot(st, table, eval_id)["keys"])
    rows_by_id = {r[ds.load_schema(st, dataset).key]: r for r in rows}

    runner = inference["runner"]()
    specs = runner.plan(list(rows_by_id), runner.Config(chunk_size=chunk_size, model=config))
    adapter = InlineAdapter(st, table, eval_id, jobs={"infer": inference["process_factory"](rows_by_id)})
    for spec in specs:
        adapter.submit(spec, tags={"cairn/eval_id": eval_id})
    return evals.count_predictions(st, table, eval_id)
