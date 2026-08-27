"""Parsing and validation of schema.yaml (dataset columns) and table.yaml (eval columns).

Columns shown in a list view are always scalars; anything structured goes under `nested`.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
"""Dataset and eval table names become storage paths, so they stay simple."""


def _check_name(name: str) -> None:
    if not NAME_RE.match(name or ""):
        raise ValueError(f"invalid name '{name}': use letters, digits, dot, dash or underscore (max 64)")


# --- type checks --------------------------------------------------------------

def _is_datetime(v: Any) -> bool:
    if isinstance(v, (_dt.datetime, _dt.date)):
        return True
    if isinstance(v, str):
        try:
            _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False
    return False


def _is_list_of(v: Any, inner: type | tuple[type, ...]) -> bool:
    return isinstance(v, list) and all(isinstance(x, inner) for x in v)


# Type name -> predicate. `json` accepts anything.
_CHECKS = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "datetime": _is_datetime,
    "s3path": lambda v: isinstance(v, str),
    "list[float]": lambda v: _is_list_of(v, (int, float)),
    "list[int]": lambda v: _is_list_of(v, int),
    "list[str]": lambda v: _is_list_of(v, str),
    "json": lambda v: True,
}


# Added to every result row by cairn; scripts never set these.
# A model is identified by run.py plus the run config and code_commit, so there is no
# separate model version field.
COMMON_EVAL_COLUMNS = [
    "eval_id",
    "dataset",
    "snapshot_id",
    "row_count",
    "evaluator_version",
    "code_commit",
    "title",
    "comment",
    "created_by",
    "created_at",
    "status",
]


class Column(BaseModel):
    name: str
    type: str = "str"
    required: bool = False
    derived: bool = False  # filled in by ingest rather than supplied
    display: str | None = None


class NestedField(BaseModel):
    name: str
    type: str = "json"


class Nested(BaseModel):
    """A structured child table, such as a sequence of points.

    Values are stored inline in the row, kept out of the list view, and read at eval time
    with `ctx.dataset.frames(sample_id)`.

    Heavy assets (images, masks, point clouds) do not belong here: put a URL or path in a
    scalar column and read it with `ctx.read_bytes(...)`.
    """

    schema_: list[NestedField] = Field(default_factory=list, alias="schema")

    model_config = {"populate_by_name": True}


class DatasetSchema(BaseModel):
    """A dataset schema, as written in schema.yaml."""

    name: str
    description: str = ""
    kind: str = ""  # optional label, e.g. detection / sfm / llm / tabular
    key: str
    columns: list[Column]
    nested: dict[str, Nested] = Field(default_factory=dict)

    def model_post_init(self, _ctx: Any) -> None:
        _check_name(self.name)
        if not self.columns:
            raise ValueError("a dataset needs at least one column")
        if self.key not in {c.name for c in self.columns}:
            raise ValueError(f"key '{self.key}' must be one of the columns")

    def scalar_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def validate_row(self, row: dict[str, Any]) -> list[str]:
        """Validate one row; an empty list means it is valid."""
        errs: list[str] = []
        if self.key not in row:
            errs.append(f"key '{self.key}' is missing")
        for col in self.columns:
            present = col.name in row and row[col.name] is not None
            if not present:
                if col.required:
                    errs.append(f"'{col.name}' is required")
                continue
            check = _CHECKS.get(col.type)
            if check and not check(row[col.name]):
                errs.append(f"'{col.name}' must be of type {col.type} (got {row[col.name]!r})")
        return errs


class EvalColumn(BaseModel):
    name: str
    type: str = "float"
    display: str | None = None
    primary: bool = False
    """Primary metric: the column drawn as the stone width. Set it on one column only."""
    direction: Literal["higher", "lower"] = "higher"
    """Which direction is better; decides delta colors and how stone width is drawn."""
    scale: list[float] | None = None
    """Absolute range `[min, max]` for the stone width. Without it the width is scaled
    relative to the runs currently shown, which exaggerates small differences."""


class EvalTableSchema(BaseModel):
    """An eval table, as written in table.yaml."""

    name: str
    description: str = ""
    dataset: str | None = None  # default target, preselected in the run form
    columns: list[EvalColumn]
    default_sort: str = "created_at desc"

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def primary_column(self) -> EvalColumn | None:
        """The primary column: the one marked `primary`, or the first one."""
        for c in self.columns:
            if c.primary:
                return c
        return self.columns[0] if self.columns else None

    def model_post_init(self, _ctx: Any) -> None:
        _check_name(self.name)
        if sum(1 for c in self.columns if c.primary) > 1:
            raise ValueError("primary: true must be set on exactly one column")
        # Common metadata is written by cairn; a column of the same name would let a
        # script overwrite snapshot_id or created_at in the listing.
        if clash := sorted(set(self.column_names()) & set(COMMON_EVAL_COLUMNS)):
            raise ValueError(f"these column names are reserved: {clash}")

    def validate_result_row(self, row: dict[str, Any]) -> list[str]:
        """Validate the row an evaluator returned; columns must match exactly."""
        errs: list[str] = []
        allowed = set(self.column_names())
        extra = set(row) - allowed
        if extra:
            errs.append(f"columns not in table.yaml: {sorted(extra)}")
        for col in self.columns:
            if col.name in row and row[col.name] is not None:
                check = _CHECKS.get(col.type)
                if check and not check(row[col.name]):
                    errs.append(f"'{col.name}' must be of type {col.type}")
        return errs




def parse_dataset_schema(text: str) -> DatasetSchema:
    return DatasetSchema.model_validate(yaml.safe_load(text))


def parse_eval_table(text: str) -> EvalTableSchema:
    return EvalTableSchema.model_validate(yaml.safe_load(text))
