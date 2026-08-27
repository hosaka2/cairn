"""Doing independent lookups at the same time.

Against a bucket, a page or a listing is mostly waiting on the network. The work items
here never touch each other — one table, one dataset, one run — so the only thing that
grows with their number is the waiting.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

WORKERS = 8
"""Enough to hide the latency, few enough to stay polite to the store."""


def each(items: list[T], build: Callable[[T], R]) -> list[R]:
    """Apply `build` to every item, at the same time, keeping the order."""
    if len(items) < 2:
        return [build(i) for i in items]
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(items))) as pool:
        return list(pool.map(build, items))


def both(*fns: Callable[[], Any]) -> list[Any]:
    """Run a handful of unrelated lookups at the same time and keep their results."""
    with ThreadPoolExecutor(max_workers=len(fns)) as pool:
        return [f.result() for f in [pool.submit(fn) for fn in fns]]
