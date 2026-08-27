"""Evaluator: turns dataset rows and predictions into an EvalResult.

Everything an evaluator needs arrives through `ctx`: the dataset, the predictions, and
any referenced asset it reads with `ctx.read_bytes`. Given the same inputs it must
produce the same result, so it keeps no hidden state of its own.

Metrics are computed here rather than during inference, because measures like mAP or
AUC only exist once every prediction is available.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from fsspec.core import OpenFile

from cairn.core.records import EvalResult


class DatasetView:
    """Read access to the dataset rows a run was pinned to."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        key: str,
        frames_loader: Callable[[str, str | None], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._rows = rows
        self._key = key
        self._frames_loader = frames_loader

    def targets(self) -> list[str]:
        """Every sample id, i.e. the values of the key column."""
        return [r[self._key] for r in self._rows]

    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    def row(self, sample_id: str) -> dict[str, Any] | None:
        for r in self._rows:
            if r.get(self._key) == sample_id:
                return r
        return None

    def frames(self, sample_id: str, field: str | None = None) -> list[dict[str, Any]]:
        """Rows of a nested table, read only when asked for.

        `field` names the nested table; it can be omitted when the schema declares one.
        """
        if self._frames_loader is None:
            return []
        return self._frames_loader(sample_id, field)


class PredictionView:
    """Read access to the predictions of a run."""

    def __init__(self, loader: Callable[[], Iterator[tuple[str, bytes]]]) -> None:
        self._loader = loader

    def iter(self) -> Iterator[tuple[str, bytes]]:
        """Yield `(sample_id, payload)` for each prediction."""
        return self._loader()


class EvalContext(BaseModel):
    """What `score` is given.

    `expected_n` is how many samples the snapshot holds and `actual_n` how many
    predictions were written; how to treat the difference is the evaluator's decision.
    `open` / `read_bytes` / `read_text` read any referenced asset, local or remote.
    """

    dataset: DatasetView
    predictions: PredictionView
    config: BaseModel
    expected_n: int
    actual_n: int

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
class Evaluator(Protocol):
    """Scores one run. Takes everything from `ctx` and keeps no state between runs.

    A nested `Config` (pydantic) is optional: declare one to receive the run's config.
    """

    def score(self, ctx: EvalContext) -> EvalResult:
        ...
