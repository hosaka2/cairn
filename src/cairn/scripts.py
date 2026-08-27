"""What the convention directories declare, read without running them.

A listing needs to know which scripts exist and which eval versions a table has. That
is answerable from the file names and one look at the source, and importing to find out
would pull whatever a script depends on — a detection model, a solver, a CUDA runtime —
into the process drawing the page, where a heavy import or a broken one has no business
being. Running a script is a different matter: `registry` imports, and only when
something is actually about to be run.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

VERSION_FILE = re.compile(r"^v\d+$")
"""Eval versions are `v1.py`, `v2.py`, … only — never `validate.py` and friends."""

EXPORTS = {"ingest.py": ("INGESTOR",), "run.py": ("RUNNER", "PROCESS_FACTORY", "CONFIG")}
"""What each kind of script file has to name at the top level."""


def root() -> str:
    """Where the convention directories live."""
    return os.environ.get("CAIRN_SCRIPTS", os.getcwd())


def _declares(path: Path, names: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Whether the file assigns each name at the top level, without importing it."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as e:
        return False, [f"cannot be read: {type(e).__name__}: {e}"]
    assigned = {t.id for node in tree.body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    assigned |= {node.target.id for node in tree.body
                 if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    missing = [n for n in names if n not in assigned]
    return not missing, [f"`{n}` is missing (this file must define {n})" for n in missing]


# --- the declaration files, which are read and never imported -----------------

DECLARED = {"datasets": ("input.py", "Input"), "evals": ("config.py", "Config")}
"""Kind -> (the file that declares a form, the class in it holding the fields).

These live apart from the script that does the work, and cairn only ever reads them.
That is the whole point: a heavy import in `ingest.py` cannot reach the process drawing
the form, because the file the form comes from is never executed — not by convention,
by there being no code path that runs it.
"""


def _field(node: ast.AnnAssign) -> dict[str, Any] | None:
    """One annotated attribute, described the way a form needs it."""
    if not isinstance(node.target, ast.Name):
        return None
    ann, widget, options = node.annotation, "text", None
    if isinstance(ann, ast.Name) and ann.id == "bool":
        widget = "bool"
    elif isinstance(ann, ast.Name) and ann.id in ("int", "float"):
        widget = "number"
    elif (isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name)
          and ann.value.id == "Literal"):
        args = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
        options = [str(a.value) for a in args if isinstance(a, ast.Constant)]
        widget = "select"
    default: Any = ""
    if node.value is not None:
        try:
            default = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):   # a call or a name: nothing to show
            default = ""
    return {"name": node.target.id, "widget": widget, "options": options,
            "default": "" if default is None else default,
            "required": node.value is None, "type": ast.unparse(ann)}


def fields(kind: str, name: str) -> list[dict[str, Any]] | None:
    """The form fields a script declares, read without importing anything.

    None means there is no declaration file, which is not an error: the dataset takes
    pasted JSONL instead of a generated form.
    """
    fname, cls = DECLARED[kind]
    path = Path(root()) / kind / name / fname
    if not path.exists():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return [f for b in node.body if isinstance(b, ast.AnnAssign) and (f := _field(b))]
    return None


def datasets_with_ingest() -> list[str]:
    """Datasets that have an ingest script."""
    return sorted(d.name for d in (Path(root()) / "datasets").glob("*") if (d / "ingest.py").exists())


def versions(table: str) -> list[str]:
    """Eval versions a table has: the file names, plus anything CAIRN_REGISTRY named."""
    from cairn import registry

    registry.module_registrations()
    d = Path(root()) / "evals" / table
    named = {f.stem for f in d.glob("v*.py") if VERSION_FILE.match(f.stem)}
    return sorted(named | set(registry.get_evaluators(table)))


def problems() -> list[dict[str, str]]:
    """What is visibly wrong with the scripts, short of running them.

    A file that does not parse, or does not name what its kind has to name, is caught
    here. Whether the objects it names honour the contract is `cairn check`, which
    imports them.
    """
    base = Path(root())
    out: list[dict[str, str]] = []
    for f in sorted(base.glob("datasets/*/ingest.py")):
        out += [{"where": str(f), "what": w} for w in _declares(f, EXPORTS["ingest.py"])[1]]
    for d in sorted(p for p in (base / "evals").glob("*") if p.is_dir()):
        if (run := d / "run.py").exists():
            out += [{"where": str(run), "what": w} for w in _declares(run, EXPORTS["run.py"])[1]]
        for vf in sorted(d.glob("v*.py")):
            if not VERSION_FILE.match(vf.stem):
                out.append({"where": str(vf),
                            "what": "eval versions must be named v1.py, v2.py, … — this file is ignored"})
                continue
            out += [{"where": str(vf), "what": w} for w in _declares(vf, ("EVALUATOR",))[1]]
    return out
