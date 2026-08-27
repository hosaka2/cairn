"""Ingestor: turns any input into rows that conform to schema.yaml.

A script only yields dicts; writing them to storage is cairn's job. The `Input` model
also defines the form shown in the web UI.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from fsspec.core import OpenFile


class IngestContext(BaseModel):
    """What `ingest` is given.

    `open` / `read_bytes` / `read_text` read a source file, local or remote. Anything
    else a script needs, such as a database, it brings itself.
    """

    dataset: str
    tmpdir: str
    created_by: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def open(self, uri: str, mode: str = "rb", **kwargs: Any) -> OpenFile:
        from cairn import io

        return io.open(uri, mode, **kwargs)

    def read_bytes(self, uri: str) -> bytes:
        from cairn import io

        return io.read_bytes(uri)

    def read_text(self, uri: str, encoding: str = "utf-8") -> str:
        from cairn import io

        return io.read_text(uri, encoding)


@runtime_checkable
class Ingestor(Protocol):
    """Yields rows that conform to the dataset schema."""

    class Input(BaseModel):
        """Fields the caller fills in; also rendered as the form in the web UI."""

    execution: str

    def ingest(self, inp: Ingestor.Input, ctx: IngestContext) -> Iterable[dict[str, Any]]:
        ...


class FileRef(str):
    """Reference to an uploaded file; rendered as a file field in the web UI."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())

    def as_path(self) -> Path:
        return Path(self)


class S3Path(str):
    """Path or URL to an object in storage; rendered as a text field in the web UI."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
