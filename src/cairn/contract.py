"""Validation of the script contract.

Scripts are discovered by location and export name, and are checked here as soon as
they are found, so a mistake surfaces at discovery with a message that says what is
wrong — instead of being skipped silently or failing deep inside a run.
"""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel


def _is_model(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _has_method(obj: Any, name: str, min_args: int) -> str | None:
    """Check that `name` is callable and takes at least `min_args` arguments besides self."""
    fn = getattr(obj, name, None)
    if fn is None:
        return f"`{name}` method is missing"
    if not callable(fn):
        return f"`{name}` is not callable (define it as a method)"
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return None  # accept implementations whose signature cannot be read
    if params and params[0].name == "self":
        params = params[1:]
    n = sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    has_var = any(p.kind is p.VAR_POSITIONAL for p in params)
    if n < min_args and not has_var:
        return f"`{name}` must accept {min_args} argument(s) (it takes {n})"
    return None


def check_ingestor(obj: Any) -> list[str]:
    """INGESTOR must be a class with a pydantic `Input` and `ingest(inp, ctx)`."""
    if not isinstance(obj, type):
        return ["Assign a class to INGESTOR (not an instance or a function)"]
    errs: list[str] = []
    inp = getattr(obj, "Input", None)
    if inp is None:
        errs.append("Nested class `Input` is missing (the UI form is generated from it)")
    elif not _is_model(inp):
        errs.append("`Input` must subclass pydantic BaseModel")
    if e := _has_method(obj, "ingest", 2):
        errs.append(e + " — `def ingest(self, inp, ctx)`")
    return errs


def check_evaluator(obj: Any) -> list[str]:
    """EVALUATOR must be a class with `score(ctx)`. `Config` is optional."""
    if not isinstance(obj, type):
        return ["Assign a class to EVALUATOR (not an instance or a function)"]
    errs: list[str] = []
    if e := _has_method(obj, "score", 1):
        errs.append(e + " — `def score(self, ctx) -> EvalResult`")
    cfg = getattr(obj, "Config", None)
    if cfg is not None and not _is_model(cfg):
        errs.append("`Config` must subclass pydantic BaseModel (omit it if unused)")
    return errs


def check_runner(obj: Any) -> list[str]:
    """RUNNER must be a class with a pydantic `Config` and `plan(sample_ids, cfg)`."""
    if not isinstance(obj, type):
        return ["Assign a class to RUNNER (not an instance or a function)"]
    errs: list[str] = []
    cfg = getattr(obj, "Config", None)
    if cfg is None:
        errs.append("Nested class `Config` is missing (chunking settings)")
    elif not _is_model(cfg):
        errs.append("`Config` must subclass pydantic BaseModel")
    if e := _has_method(obj, "plan", 2):
        errs.append(e + " — `def plan(self, sample_ids, cfg) -> list[RunSpec]`")
    return errs


def check_process_factory(obj: Any) -> list[str]:
    if not callable(obj):
        return ["PROCESS_FACTORY must be a function (returning `process_one`)"]
    return []


def check_config(obj: Any) -> list[str]:
    if not _is_model(obj):
        return ["CONFIG must be a class subclassing pydantic BaseModel"]
    return []


# Export name -> (check, where it belongs). Shared by discovery and `cairn check`.
EXPORTS = {
    "INGESTOR": (check_ingestor, "datasets/<name>/ingest.py"),
    "EVALUATOR": (check_evaluator, "evals/<name>/v1.py …"),
    "RUNNER": (check_runner, "evals/<name>/run.py"),
    "PROCESS_FACTORY": (check_process_factory, "evals/<name>/run.py"),
    "CONFIG": (check_config, "evals/<name>/run.py"),
}


def check_export(name: str, obj: Any) -> list[str]:
    """Validate one export against its contract; unknown names pass."""
    entry = EXPORTS.get(name)
    return entry[0](obj) if entry else []
