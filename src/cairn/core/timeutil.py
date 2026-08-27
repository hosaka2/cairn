"""Timestamps. Everything is recorded as a UTC ISO 8601 string."""

from __future__ import annotations

import datetime as _dt


def now_iso() -> str:
    """Current time, UTC ISO 8601, second precision."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
