"""Finding the scripts this process can run, and turning their models into forms.

Two ways in: the convention directories (scanned) and a `CAIRN_REGISTRY` module
(imported). Both must survive scripts that are broken or absent, because the app has to
keep serving and say what is wrong instead of failing to start.
"""

from __future__ import annotations

import sys
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from cairn import registry, scripts

INGEST_PY = """from pydantic import BaseModel


class Ingest:
    class Input(BaseModel):
        n: int = 1

    execution = "inline"

    def ingest(self, inp, ctx):
        yield {"id": "a"}


INGESTOR = Ingest
"""


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    """An empty script root and an empty registry, restored for the next test."""
    monkeypatch.setenv("CAIRN_SCRIPTS", str(tmp_path))
    monkeypatch.setenv("CAIRN_REGISTRY", "")
    for store in (registry._PROBLEMS, registry._INGESTORS, registry._EVALUATORS,
                  registry._INFERENCE, registry._SCANNED, registry._LOADED):
        store.clear()
    yield
    for store in (registry._PROBLEMS, registry._INGESTORS, registry._EVALUATORS,
                  registry._INFERENCE, registry._SCANNED, registry._LOADED):
        store.clear()


# --- loading ----------------------------------------------------------------

def test_a_registry_module_that_cannot_be_imported_does_not_stop_the_app(monkeypatch):
    monkeypatch.setenv("CAIRN_REGISTRY", "cairn.no.such.module")

    registry.ensure_loaded()

    assert registry.get_ingestor("anything") is None


def test_the_registry_module_survives_a_rescan(monkeypatch, tmp_path):
    """A rescan forgets what was registered, so the module has to register again."""
    monkeypatch.setenv("CAIRN_REGISTRY", "cairn.demo.registry")
    sys.modules.pop("cairn.demo.registry", None)

    registry.ensure_loaded()
    assert registry.get_ingestor("sensor-anomaly-A") is not None

    (tmp_path / "datasets" / "later").mkdir(parents=True)     # changes the fingerprint
    (tmp_path / "datasets" / "later" / "ingest.py").write_text(INGEST_PY)
    registry.ensure_loaded()

    assert registry.get_ingestor("sensor-anomaly-A") is not None
    assert registry.get_ingestor("later") is not None


def test_a_scan_that_blows_up_is_recorded_rather_than_raised(monkeypatch):
    def boom(root: str) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr(registry, "discover", boom)
    registry.ensure_loaded()

    assert "scan failed: OSError: disk went away" in registry.problems()[0]["what"]


def test_a_file_where_an_eval_directory_was_expected_is_skipped(tmp_path):
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "notes.txt").write_text("not a script")

    registry.discover(str(tmp_path))

    assert registry.problems() == []
    assert registry.get_evaluators("notes.txt") == {}


# --- models become forms ----------------------------------------------------

class Cfg(BaseModel):
    threshold: float = 0.5
    chunk_size: int = 16
    label: str = "run"
    strict: bool = False
    mode: Literal["fast", "careful"] = "fast"
    required: str = Field(...)
    made_up: list[str] = Field(default_factory=list)


def _fields() -> dict[str, dict]:
    return {f["name"]: f for f in registry.form_fields(Cfg)}


def test_each_field_gets_the_widget_its_type_calls_for():
    fields = _fields()
    assert fields["threshold"]["widget"] == "number"
    assert fields["chunk_size"]["widget"] == "number"
    assert fields["label"]["widget"] == "text"
    assert fields["strict"]["widget"] == "bool"
    assert fields["mode"]["widget"] == "select"
    assert fields["mode"]["options"] == ["fast", "careful"]


def test_defaults_are_offered_and_a_field_without_one_starts_empty():
    fields = _fields()
    assert fields["threshold"]["default"] == 0.5
    assert fields["required"]["default"] == ""
    assert fields["made_up"]["default"] == []


@pytest.mark.parametrize("sent, strict", [("on", True), ("true", True), ("1", True), ("off", False)])
def test_a_checkbox_arrives_as_whatever_the_browser_sent(sent, strict):
    assert registry.coerce(Cfg, {"required": "x", "strict": sent})["strict"] is strict


def test_numbers_arrive_as_text_and_are_read_as_numbers():
    out = registry.coerce(Cfg, {"required": "x", "threshold": "0.75", "chunk_size": "32"})
    assert out["threshold"] == 0.75 and out["chunk_size"] == 32


def test_a_field_left_blank_falls_back_to_its_default():
    out = registry.coerce(Cfg, {"required": "x", "threshold": ""})
    assert out["threshold"] == 0.5


def test_fields_the_form_does_not_carry_are_ignored():
    """The form posts more than the model declares (mode switches, csrf, …)."""
    out = registry.coerce(Cfg, {"required": "x", "mode": "careful", "not_a_field": "1"})
    assert out["mode"] == "careful" and "not_a_field" not in out


def test_a_file_python_cannot_load_as_a_module_is_refused(tmp_path):
    """Discovery only reaches `.py` files; anything else must not be half-imported."""
    other = tmp_path / "notes.txt"
    other.write_text("hello")

    with pytest.raises(ImportError, match="cannot be imported"):
        registry._import_file(other, "notes")


# --- what a listing is allowed to load ---------------------------------------

_MARK = '''import pathlib
pathlib.Path(__file__).with_name("IMPORTED").write_text("yes")   # stands in for a heavy import
'''
INGEST = _MARK + "from cairn.demo.ingest import SyntheticIngest\nINGESTOR = SyntheticIngest\n"
V1 = _MARK + "from cairn.demo.evaluate import AnomalyEvalV1\nEVALUATOR = AnomalyEvalV1\n"
RUN = _MARK + ("from cairn.demo.model import ChunkRunner, ModelConfig, make_process_one\n"
               "RUNNER = ChunkRunner\nPROCESS_FACTORY = make_process_one\nCONFIG = ModelConfig\n")


def _scripts(tmp_path, monkeypatch):
    """A script directory whose files record the fact of being imported."""
    (tmp_path / "datasets" / "things").mkdir(parents=True)
    (tmp_path / "datasets" / "things" / "ingest.py").write_text(INGEST, encoding="utf-8")
    (tmp_path / "evals" / "acc").mkdir(parents=True)
    (tmp_path / "evals" / "acc" / "v1.py").write_text(V1, encoding="utf-8")
    monkeypatch.setenv("CAIRN_SCRIPTS", str(tmp_path))
    return tmp_path


def _imported(root) -> list[str]:
    return sorted(str(f.parent.name) for f in root.rglob("IMPORTED"))


def test_a_listing_reads_the_scripts_without_running_them(tmp_path, monkeypatch):
    """A page that says which scripts exist must not pull in what they depend on.

    An ingest script imports its model, its solver, its CUDA runtime; a listing that
    imports to find out what is there loads all of it into the process drawing the page.
    """
    root = _scripts(tmp_path, monkeypatch)

    assert scripts.datasets_with_ingest() == ["things"]
    assert scripts.versions("acc") == ["v1"]
    assert scripts.problems() == []
    assert _imported(root) == []


def test_opening_one_form_imports_that_script_and_no_other(tmp_path, monkeypatch):
    """The form for one dataset must not pull in every other script in the tree."""
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "other").mkdir()
    (root / "datasets" / "other" / "ingest.py").write_text(INGEST, encoding="utf-8")

    registry.load_ingestor("things")

    assert _imported(root) == ["things"]


def test_a_second_look_does_not_import_it_again(tmp_path, monkeypatch):
    """Editing the file is what makes it load again, not visiting the page again."""
    root = _scripts(tmp_path, monkeypatch)
    registry.load_ingestor("things")
    (root / "datasets" / "things" / "IMPORTED").unlink()

    registry.load_ingestor("things")

    assert _imported(root) == []


def test_a_form_loads_the_script_it_is_a_form_for(tmp_path, monkeypatch):
    """Each of the three kinds is reachable one at a time."""
    root = _scripts(tmp_path, monkeypatch)
    (root / "evals" / "acc" / "run.py").write_text(RUN, encoding="utf-8")

    assert registry.load_ingestor("things") is not None
    assert registry.load_inference("acc") is not None
    assert registry.load_evaluator("acc", "v1") is not None
    assert registry.load_evaluator("acc", "nonsense") is None      # not a version at all
    assert registry.load_ingestor("nothing-here") is None


def test_running_one_is_what_imports_it(tmp_path, monkeypatch):
    """The contract is checked where it matters: when something is about to be run."""
    root = _scripts(tmp_path, monkeypatch)

    registry.ensure_loaded()

    assert _imported(root) == ["acc", "things"]


def test_a_second_scan_of_unchanged_scripts_does_nothing(tmp_path, monkeypatch):
    _scripts(tmp_path, monkeypatch)
    registry.ensure_loaded()

    registry.ensure_loaded()      # same files: no rescan, and nothing forgotten

    assert registry.get_ingestor("things") is not None


def test_a_script_that_forgets_its_export_is_named_without_being_run(tmp_path, monkeypatch):
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "things" / "ingest.py").write_text("X = 1\n", encoding="utf-8")
    (root / "evals" / "acc" / "vnext.py").write_text("EVALUATOR = object\n", encoding="utf-8")
    (root / "evals" / "acc" / "run.py").write_text("RUNNER = object\n", encoding="utf-8")

    said = scripts.problems()

    assert any("`INGESTOR` is missing" in p["what"] for p in said)
    assert any("`PROCESS_FACTORY` is missing" in p["what"] for p in said)
    assert any("v1.py, v2.py" in p["what"] for p in said)
    assert _imported(root) == []


def test_a_script_that_does_not_parse_is_reported_not_raised(tmp_path, monkeypatch):
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "things" / "ingest.py").write_text("def (:\n", encoding="utf-8")

    assert any("cannot be read" in p["what"] for p in scripts.problems())


# --- the form is read, never run ---------------------------------------------

INPUT_PY = '''from typing import Literal
from pydantic import BaseModel


class Input(BaseModel):
    n: int = 40
    site: str
    split: Literal["train", "val"] = "val"
    skip_empty: bool = True
    weights: float = compute()
'''


def test_a_form_is_read_out_of_the_declaration_and_never_run(tmp_path, monkeypatch):
    """The file a form comes from is not executed, so nothing in it can reach the page.

    That is the difference between a rule and a promise: `ingest.py` may import a model
    or a solver, and no code path draws a form from it.
    """
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "things" / "input.py").write_text(INPUT_PY, encoding="utf-8")

    declared = scripts.fields("datasets", "things")
    assert declared is not None
    got = {f["name"]: f for f in declared}

    assert [f["widget"] for f in got.values()] == ["number", "text", "select", "bool", "number"]
    assert got["split"]["options"] == ["train", "val"] and got["split"]["default"] == "val"
    assert got["site"]["required"] and not got["n"]["required"]
    assert got["weights"]["default"] == ""      # a call is no default a form can show
    assert _imported(root) == []


def test_without_a_declaration_there_is_no_generated_form(tmp_path, monkeypatch):
    """No file, no form — pasted JSONL still works. Nothing is imported to find out."""
    root = _scripts(tmp_path, monkeypatch)

    assert scripts.fields("datasets", "things") is None
    assert scripts.fields("evals", "acc") is None
    assert _imported(root) == []


def test_a_declaration_that_does_not_parse_is_no_form_either(tmp_path, monkeypatch):
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "things" / "input.py").write_text("class Input(:\n", encoding="utf-8")

    assert scripts.fields("datasets", "things") is None


def test_a_declaration_that_drifts_from_its_script_is_reported(tmp_path, monkeypatch):
    """The form and the model that validates it have to name the same fields."""
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "things" / "input.py").write_text(
        "from pydantic import BaseModel\n\n\nclass Input(BaseModel):\n    nope: int = 1\n",
        encoding="utf-8")
    registry.ensure_loaded()

    said = registry.declaration_mismatches()

    assert len(said) == 1 and "declares ['nope']" in said[0]["what"]
    assert registry.declaration_mismatches.__doc__      # it is `cairn check`, not a page


def test_a_declaration_that_matches_its_script_is_not_reported(tmp_path, monkeypatch):
    """The demo's own wiring is the case that has to stay quiet."""
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "things" / "input.py").write_text(
        "from pydantic import BaseModel\n\n\nclass Input(BaseModel):\n"
        "    n: int = 40\n    seed: int = 1\n    site: str = 'A'\n    anomaly_rate: float = 0.3\n",
        encoding="utf-8")
    (root / "evals" / "acc" / "config.py").write_text(   # declared, but nothing registered for it
        "from pydantic import BaseModel\n\n\nclass Config(BaseModel):\n    threshold: float = 0.6\n",
        encoding="utf-8")
    registry.ensure_loaded()

    assert registry.declaration_mismatches() == []


def test_a_declaration_without_the_class_is_no_form(tmp_path, monkeypatch):
    """A file that parses but declares nothing we recognise is simply no form."""
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "things" / "input.py").write_text(
        "from pydantic import BaseModel\n\n\nclass Other(BaseModel):\n    n: int = 1\n",
        encoding="utf-8")

    assert scripts.fields("datasets", "things") is None


def test_an_annotation_that_is_not_a_plain_attribute_is_skipped(tmp_path, monkeypatch):
    """Only `name: type` is a field; anything else in the class body is not."""
    root = _scripts(tmp_path, monkeypatch)
    (root / "datasets" / "things" / "input.py").write_text(
        "from pydantic import BaseModel\n\n\nclass Input(BaseModel):\n"
        "    n: int = 1\n    obj.attr: int = 2\n",
        encoding="utf-8")

    declared = scripts.fields("datasets", "things")
    assert declared is not None
    assert [f["name"] for f in declared] == ["n"]
