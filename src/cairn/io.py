"""File access for scripts, backend independent.

Lets ingest and eval scripts read `s3://`, `gs://`, `file://` and `http(s)://` through
one API, with the same credentials as CAIRN_ROOT. Databases are out of scope: a script
that needs one brings its own dependency.
"""

from __future__ import annotations

from typing import IO, Any, cast

import fsspec
from fsspec.core import OpenFile


def open(uri: str, mode: str = "rb", **kwargs: Any) -> OpenFile:
    """Open a URI; use it as a context manager."""
    return cast("OpenFile", fsspec.open(uri, mode, **kwargs))


def read_bytes(uri: str) -> bytes:
    with open(uri, "rb") as f:
        return cast("IO[bytes]", f).read()


def read_text(uri: str, encoding: str = "utf-8") -> str:
    return read_bytes(uri).decode(encoding)


def write_bytes(uri: str, data: bytes) -> None:
    with open(uri, "wb") as f:
        cast("IO[bytes]", f).write(data)
