"""ULID generation.

A ULID starts with a timestamp, so sorting keys lexicographically sorts them by
creation time. Dataset rows are merged in that order, which is what makes
"last write wins" well defined without a lock.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone

# Crockford's base32 (no I, L, O, U).
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


_LOCK = threading.Lock()
_LAST_MS = 0
_LAST_TAIL = 0


def ulid() -> str:
    """26 characters: 48 bits of milliseconds, then 80 bits that only ever go up.

    Sorting ULIDs has to sort them by the order they were made, because that is what
    makes "last write wins" mean anything. A millisecond is not fine enough to tell two
    writes apart, so inside one the tail is incremented rather than drawn again — twice
    out of every four consecutive draws came out in the wrong order otherwise. A clock
    that steps backwards is held at the last millisecond for the same reason.

    Two processes writing in the same millisecond are still in no particular order, but
    then neither is what they did: nothing here can order writes nobody ordered.
    """
    global _LAST_MS, _LAST_TAIL

    with _LOCK:
        ms = int(time.time() * 1000) & ((1 << 48) - 1)
        if ms > _LAST_MS:
            # A fresh tail, with the top bit clear so incrementing cannot carry into
            # the timestamp however many are made in one millisecond.
            _LAST_MS, _LAST_TAIL = ms, int.from_bytes(os.urandom(10), "big") >> 1
        else:
            _LAST_TAIL += 1
        return _encode(_LAST_MS, 10) + _encode(_LAST_TAIL, 16)


def ulid_time(value: str) -> datetime:
    """When a ULID was made. Its first ten characters are the milliseconds."""
    ms = 0
    for ch in value[:10]:
        ms = ms * 32 + _CROCKFORD.index(ch)
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def eval_id() -> str:
    """Id of an evaluation run."""
    return "e_" + ulid()


def ingest_id() -> str:
    """Id of an ingest."""
    return "ing_" + ulid()
