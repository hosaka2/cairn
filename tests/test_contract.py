"""Discovery of user scripts and the contract they must satisfy.

- an export that breaks the contract is not registered, and the reason is kept in problems()
- eval version files are `v` + digits only, so `validate.py` is never imported
- an import failure is recorded as a problem instead of stopping discovery
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from cairn import contract, registry


def _scripts(tmp_path: Path, ingest: str = "", run: str = "", v1: str = "") -> Path:
    if ingest:
        d = tmp_path / "datasets" / "ds"
        d.mkdir(parents=True)
        (d / "ingest.py").write_text(ingest, encoding="utf-8")
    if run or v1:
        e = tmp_path / "evals" / "ev"
        e.mkdir(parents=True)
        if run:
            (e / "run.py").write_text(run, encoding="utf-8")
        if v1:
            (e / "v1.py").write_text(v1, encoding="utf-8")
    return tmp_path


def _fresh() -> None:
    registry._PROBLEMS.clear()
    registry._INGESTORS.clear()
    registry._EVALUATORS.clear()
    registry._INFERENCE.clear()


GOOD_INGEST = """
from pydantic import BaseModel

class Ingest:
    class Input(BaseModel):
        source: str = ""
    def ingest(self, inp, ctx):
        yield {"id": "a"}

INGESTOR = Ingest
"""


def test_contract_checks():
    """The contract checks themselves: each reports what is missing, in words."""
    class NoInput:
        def ingest(self, inp, ctx): ...

    class PlainInput:
        class Input:  # not a pydantic model
            ...
        def ingest(self, inp, ctx): ...

    assert any("Input" in e for e in contract.check_ingestor(NoInput))
    assert any("BaseModel" in e for e in contract.check_ingestor(PlainInput))
    assert len(contract.check_ingestor(object())) == 1   # not a class: one problem, not a cascade

    class NoScore:
        ...

    assert any("score" in e for e in contract.check_evaluator(NoScore))
    assert contract.check_evaluator(type("E", (), {"score": lambda self, ctx: None})) == []
    assert contract.check_process_factory(lambda rows: None) == []
    assert contract.check_process_factory(3) != []


def test_discover_records_problems(tmp_path):
    """A contract violation is not registered; the reason is recorded, not silently dropped."""
    _fresh()
    root = _scripts(
        tmp_path,
        ingest="class Ingest:\n    pass\n\nINGESTOR = Ingest\n",       # no Input, no ingest
        run="RUNNER = 1\n",                                            # wrong shape, nothing else
        v1="EVALUATOR = 'not a class'\n",
    )
    registry.discover(str(root))
    assert registry.get_ingestor("ds") is None
    assert registry.get_inference("ev") is None
    assert registry.get_evaluators("ev") == {}
    what = " ".join(p["what"] for p in registry.problems())
    assert "Input" in what and "ingest" in what
    assert "PROCESS_FACTORY" in what          # a missing export is named too
    assert "EVALUATOR" in what


def test_discover_import_error_is_reported(tmp_path):
    """A syntax error does not stop discovery; the reason is recorded."""
    _fresh()
    root = _scripts(tmp_path, ingest="def broken(:\n")
    registry.discover(str(root))
    assert any("SyntaxError" in p["what"] for p in registry.problems())


def test_version_files_only_v_digits(tmp_path):
    """Only vN.py files are eval versions; anything else is reported, never executed."""
    _fresh()
    root = _scripts(tmp_path, v1="class E:\n    def score(self, ctx): ...\n\nEVALUATOR = E\n")
    (root / "evals" / "ev" / "validate.py").write_text("raise RuntimeError('must not be imported')\n")
    registry.discover(str(root))
    assert set(registry.get_evaluators("ev")) == {"v1"}
    assert any("validate.py" in p["where"] for p in registry.problems())


def test_good_scripts_register(tmp_path):
    _fresh()
    root = _scripts(tmp_path, ingest=GOOD_INGEST)
    registry.discover(str(root))
    assert registry.get_ingestor("ds") is not None
    assert registry.problems() == []


# --- each way an export can be wrong ----------------------------------------

class _NotAModel:
    """Stands in for a `Config`/`Input` that is a plain class, not a pydantic model."""


def test_an_export_that_is_not_callable_says_to_define_a_method():
    class Ingest:
        class Input(BaseModel):
            pass

        ingest = "not a method"

    assert "`ingest` is not callable" in " ".join(contract.check_ingestor(Ingest))


def test_a_method_taking_too_few_arguments_is_named_with_what_it_needs():
    class Ingest:
        class Input(BaseModel):
            pass

        def ingest(self):
            pass

    errs = " ".join(contract.check_ingestor(Ingest))
    assert "must accept 2 argument(s) (it takes 0)" in errs
    assert "def ingest(self, inp, ctx)" in errs


def test_a_signature_that_cannot_be_read_is_accepted():
    """Some callables are C-implemented and have no signature to inspect."""
    import math

    class Ingest:
        class Input(BaseModel):
            pass

        ingest = math.log       # inspect.signature refuses this one

    assert contract.check_ingestor(Ingest) == []


def test_an_evaluator_config_must_be_a_model_when_it_exists():
    class Eval:
        Config = _NotAModel

        def score(self, ctx):
            return None

    assert contract.check_evaluator(Eval) == ["`Config` must subclass pydantic BaseModel (omit it if unused)"]


def test_a_runner_needs_a_config_and_a_plan():
    class NoConfig:
        def plan(self, sample_ids, cfg):
            return []

    class BadEverything:
        Config = _NotAModel

        def plan(self):
            return []

    assert "Nested class `Config` is missing" in " ".join(contract.check_runner(NoConfig))
    errs = " ".join(contract.check_runner(BadEverything))
    assert "`Config` must subclass pydantic BaseModel" in errs
    assert "def plan(self, sample_ids, cfg) -> list[RunSpec]" in errs


def test_config_must_be_a_pydantic_model():
    assert contract.check_config(_NotAModel) == ["CONFIG must be a class subclassing pydantic BaseModel"]
    assert contract.check_config(_Cfg) == []


class _Cfg(BaseModel):
    threshold: float = 0.5


def _ingest_function(inp, ctx):
    """A module-level implementation, wired in as a staticmethod."""
    yield {}


def test_a_method_defined_as_a_plain_function_is_read_the_same_way():
    class Ingest:
        class Input(BaseModel):
            pass

        ingest = staticmethod(_ingest_function)

    assert contract.check_ingestor(Ingest) == []
