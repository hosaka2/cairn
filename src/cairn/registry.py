"""In-process registry of the scripts cairn can run.

Scripts are found by scanning the convention directories (`CAIRN_SCRIPTS`, current
directory by default). A module named by `CAIRN_REGISTRY` may register more of them
programmatically; the bundled demo uses that.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cairn import scripts

_INGESTORS: dict[str, Any] = {}                 # dataset -> Ingestor class
_EVALUATORS: dict[str, dict[str, Any]] = {}      # table -> {version: Evaluator class}
_INFERENCE: dict[str, dict[str, Any]] = {}       # table -> {runner, process_factory, config}
_LOADED: set[str] = set()
_SCANNED: dict[str, tuple] = {}     # script root -> fingerprint of the last scan
_PROBLEMS: list[dict[str, str]] = []             # contract violations found while scanning


def problems() -> list[dict[str, str]]:
    """Problems found while scanning, as `{where, what}`."""
    return list(_PROBLEMS)


def _problem(where: str, what: str) -> None:
    _PROBLEMS.append({"where": where, "what": what})


def register_ingestor(dataset: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _INGESTORS[dataset] = cls
        return cls
    return deco


def register_evaluator(table: str, version: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _EVALUATORS.setdefault(table, {})[version] = cls
        return cls
    return deco


def register_inference(table: str, *, runner: type, process_factory: Callable, config: type) -> None:
    _INFERENCE[table] = {"runner": runner, "process_factory": process_factory, "config": config}


def get_ingestor(dataset: str) -> Any | None:
    return _INGESTORS.get(dataset)


def get_evaluators(table: str) -> dict[str, Any]:
    return _EVALUATORS.get(table, {})


def get_inference(table: str) -> dict[str, Any] | None:
    return _INFERENCE.get(table)


def module_registrations() -> None:
    """Import `CAIRN_REGISTRY` once, and nothing from a script directory.

    That module is named by whoever runs cairn — the bundled demo by default — so this
    imports only what cairn itself ships. A listing may call it; scanning the script
    directories is a different thing and stays behind `ensure_loaded`.
    """
    _load_registry_module(reload=False)


def _load_registry_module(*, reload: bool = True) -> None:
    """Import `CAIRN_REGISTRY`, whose import registers components in process.

    Re-imported after every rescan, because `_forget` drops what it registered too.
    """
    mod = os.environ.get("CAIRN_REGISTRY", "cairn.demo.registry")
    if not mod or (mod in _LOADED and not reload):
        return
    try:
        imported = importlib.import_module(mod)
        if mod in _LOADED:                  # already imported: run its registrations again
            importlib.reload(imported)
    except Exception:  # noqa: BLE001 - the app still works without it
        pass
    _LOADED.add(mod)


_LAZY: dict[str, int] = {}      # script file -> mtime of the last import of it


def _load_one(path: Path, exports: tuple[str, ...]) -> list[Any] | None:
    """Import one script file and take its exports, if it is new or has changed.

    None means there is nothing to register from this call — the file is not there, it
    is already loaded and unchanged, or it did not satisfy the contract. The caller
    answers from what is registered either way.
    """
    if not path.exists():
        return None
    mtime = path.stat().st_mtime_ns
    mod = _mod_name(path)
    if _LAZY.get(str(path)) != mtime:
        _PROBLEMS[:] = [p for p in _PROBLEMS if p["where"] != str(path)]
        _LAZY[str(path)] = mtime
        got = [_load_export(path, mod, name) for name in exports]
        return got if all(o is not None for o in got) else None
    return None


def registered_ingestor(dataset: str) -> Any | None:
    """What `CAIRN_REGISTRY` registered for this dataset, without touching a script file.

    The bundled demo registers this way, so its forms work without a script directory.
    A script on disk is never imported to draw a form; its fields come from `input.py`.
    """
    module_registrations()
    return get_ingestor(dataset)


def registered_inference(table: str) -> dict[str, Any] | None:
    """As `registered_ingestor`, for a table's inference unit."""
    module_registrations()
    return get_inference(table)


def load_ingestor(dataset: str) -> Any | None:
    """The ingestor for one dataset, importing that script and no other.

    A form needs the one script it is a form for. Importing the rest would pull in what
    every other script depends on, to draw a page about this one.
    """
    module_registrations()
    path = Path(scripts.root()) / "datasets" / dataset / "ingest.py"
    if got := _load_one(path, ("INGESTOR",)):
        register_ingestor(dataset)(got[0])
    return get_ingestor(dataset)


def load_inference(table: str) -> dict[str, Any] | None:
    """The inference unit for one table, importing that script and no other."""
    module_registrations()
    path = Path(scripts.root()) / "evals" / table / "run.py"
    if got := _load_one(path, ("RUNNER", "PROCESS_FACTORY", "CONFIG")):
        register_inference(table, runner=got[0], process_factory=got[1], config=got[2])
    return get_inference(table)


def load_evaluator(table: str, version: str) -> Any | None:
    """The evaluator of one version, importing that file and no other."""
    module_registrations()
    if scripts.VERSION_FILE.match(version):
        path = Path(scripts.root()) / "evals" / table / f"{version}.py"
        if got := _load_one(path, ("EVALUATOR",)):
            register_evaluator(table, version)(got[0])
    return get_evaluators(table).get(version)


def declaration_mismatches() -> list[dict[str, str]]:
    """Where a declared form and the script it belongs to have drifted apart.

    A form comes from `input.py` / `config.py`, which are read and never imported; the
    submitted values are validated by the script's own model. The two have to name the
    same fields, and only something that imports can say whether they do — so this is
    part of `cairn check`, not of drawing a page.
    """
    out: list[dict[str, str]] = []
    checks = (("datasets", scripts.datasets_with_ingest(), get_ingestor, "Input"),
              ("evals", [d.name for d in (Path(scripts.root()) / "evals").glob("*") if d.is_dir()],
               lambda t: (get_inference(t) or {}).get("config"), "Config"))
    for kind, names, resolve, cls in checks:
        for name in names:
            declared = scripts.fields(kind, name)
            if declared is None:
                continue
            obj = resolve(name)
            model = getattr(obj, cls, obj) if obj is not None else None
            if model is None or not hasattr(model, "model_fields"):
                continue
            here, there = {f["name"] for f in declared}, set(model.model_fields)
            if here != there:
                fname = scripts.DECLARED[kind][0]
                out.append({"where": f"{scripts.root()}/{kind}/{name}/{fname}",
                            "what": f"declares {sorted(here)} but the script takes {sorted(there)}"})
    return out



def ensure_loaded() -> None:
    """Register what this process can run: the `CAIRN_REGISTRY` module, then the script dirs.

    Scripts on disk are scanned last, so a user's own script wins over a registration
    made in process for the same name.
    """
    root = os.environ.get("CAIRN_SCRIPTS", os.getcwd())
    fingerprint = _fingerprint(root)
    if _SCANNED.get(root) == fingerprint:
        return
    _SCANNED[root] = fingerprint
    _forget(root)
    _load_registry_module()
    try:
        discover(root)
    except Exception as e:  # noqa: BLE001 - keep serving, but record why
        _problem(root, f"scan failed: {type(e).__name__}: {e}")


def _fingerprint(root: str) -> tuple:
    """Cheap signature of the script files, so edits are picked up without a restart."""
    base = Path(root)
    files = sorted([*base.glob("datasets/*/ingest.py"), *base.glob("evals/*/*.py")])
    return tuple((str(f), f.stat().st_mtime_ns) for f in files)


def _forget(root: str) -> None:
    """Drop what a previous scan of this root registered."""
    _PROBLEMS.clear()
    _LAZY.clear()
    _INGESTORS.clear()
    _EVALUATORS.clear()
    _INFERENCE.clear()


def _import_file(path: Any, mod_name: str) -> Any:
    """Import a module from a file path."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} cannot be imported as a module")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: dataclasses and typing look the module up by name.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    return module


def _mod_name(path: Path) -> str:
    """Unique module name for a script file, so two directories cannot collide."""
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    return f"cairn_script_{digest}"


_VERSION_FILE = re.compile(r"^v\d+$")
"""Eval versions are `v1.py`, `v2.py`, … only — never `validate.py` and friends."""


def _load_export(path: Path, mod_name: str, name: str) -> Any | None:
    """Import a file, take one export and validate it. Records the problem and returns None on failure."""
    from cairn import contract

    try:
        m = _import_file(path, mod_name)
    except Exception as e:  # noqa: BLE001 - syntax errors and the like; report, do not skip
        _problem(str(path), f"failed to import: {type(e).__name__}: {e}")
        return None
    obj = getattr(m, name, None)
    if obj is None:
        _problem(str(path), f"`{name}` is missing (this file must define {name})")
        return None
    if errs := contract.check_export(name, obj):
        for e in errs:
            _problem(f"{path} :: {name}", e)
        return None
    return obj


def discover(root: str) -> None:
    """Scan `datasets/<name>/ingest.py` and `evals/<name>/{run,vN}.py` and register what is valid.

    Anything missing or off-contract is recorded in `problems()` rather than dropped.
    """
    base = Path(root)
    for d in sorted((base / "datasets").glob("*")):
        f = d / "ingest.py"
        if not (d.is_dir() and f.exists()):
            continue
        if ing := _load_export(f, _mod_name(f), "INGESTOR"):
            register_ingestor(d.name)(ing)
    for d in sorted((base / "evals").glob("*")):
        if not d.is_dir():
            continue
        run = d / "run.py"
        if run.exists():
            mod = _mod_name(run)
            runner = _load_export(run, mod, "RUNNER")
            factory = _load_export(run, mod, "PROCESS_FACTORY")
            config = _load_export(run, mod, "CONFIG")
            if runner is not None and factory is not None and config is not None:
                register_inference(d.name, runner=runner, process_factory=factory, config=config)
        for vf in sorted(d.glob("v*.py")):
            if not _VERSION_FILE.match(vf.stem):
                _problem(str(vf), "eval versions must be named v1.py, v2.py, … — this file is ignored")
                continue
            if (ev := _load_export(vf, _mod_name(vf), "EVALUATOR")):
                register_evaluator(d.name, vf.stem)(ev)


def form_fields(model: type) -> list[dict[str, Any]]:
    """Describe a pydantic model as form fields."""
    out: list[dict[str, Any]] = []
    for name, f in model.model_fields.items():
        ann = f.annotation
        widget, options = "text", None
        if ann is bool:
            widget = "bool"
        elif typing.get_origin(ann) is typing.Literal:
            widget, options = "select", [str(a) for a in typing.get_args(ann)]
        elif ann in (int, float):
            widget = "number"
        default = f.get_default(call_default_factory=True)
        if default is None or repr(default) == "PydanticUndefined":
            default = ""
        out.append({
            "name": name, "widget": widget, "options": options,
            "default": default, "required": f.is_required(),
            "type": getattr(ann, "__name__", str(ann)),
        })
    return out


def coerce(model: type, raw: dict[str, str]) -> dict[str, Any]:
    """Validate and convert submitted form values with a pydantic model."""
    parsed: dict[str, Any] = {}
    for name, f in model.model_fields.items():
        if name not in raw or raw[name] == "":
            continue
        ann = f.annotation
        v: Any = raw[name]
        if ann is bool:
            v = raw[name] in ("on", "true", "True", "1")
        elif ann is int:
            v = int(raw[name])
        elif ann is float:
            v = float(raw[name])
        parsed[name] = v
    return model(**parsed).model_dump()
