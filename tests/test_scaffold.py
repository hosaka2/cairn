"""What `cairn init` / `cairn new …` / `cairn demo-init` generate.

The generated tree is the first thing a user edits, so it must already be correct:
every .py compiles, every .yaml parses with the real parser, and discovery registers
the scripts with no problems. Existing files are never overwritten.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cairn import registry
from cairn.cli.main import app
from cairn.core.schema import parse_dataset_schema, parse_eval_table

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Scaffolding tests share the process-wide registry, so each starts and ends empty."""
    monkeypatch.setenv("CAIRN_SCRIPTS", str(tmp_path / "no-scripts"))
    _forget_registrations()
    yield
    _forget_registrations()


def _forget_registrations() -> None:
    registry._PROBLEMS.clear()
    registry._INGESTORS.clear()
    registry._EVALUATORS.clear()
    registry._INFERENCE.clear()


def _ok(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result


def _generated(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix in {".py", ".yaml"})


# --- layout -----------------------------------------------------------------

def test_init_creates_the_convention_directories(tmp_path):
    result = _ok("init", "--dir", str(tmp_path))

    assert (tmp_path / "datasets").is_dir()
    assert (tmp_path / "evals").is_dir()
    assert (tmp_path / "README.md").exists()
    assert "datasets/<name>/" in (tmp_path / "README.md").read_text(encoding="utf-8")
    assert str(tmp_path / "README.md") in result.output


def test_new_dataset_and_new_eval_write_the_documented_paths(tmp_path):
    _ok("init", "--dir", str(tmp_path))
    dataset_out = _ok("new", "dataset", "readings", "--dir", str(tmp_path))
    eval_out = _ok("new", "eval", "quality", "--dir", str(tmp_path))

    expected = [
        tmp_path / "datasets" / "readings" / "schema.yaml",
        tmp_path / "datasets" / "readings" / "ingest.py",
        tmp_path / "evals" / "quality" / "table.yaml",
        tmp_path / "evals" / "quality" / "run.py",
        tmp_path / "evals" / "quality" / "v1.py",
    ]
    for path in expected:
        assert path.exists(), f"{path} was not generated"
    printed = dataset_out.output + eval_out.output
    for path in expected:
        assert f"write {path}" in printed


def test_new_dataset_names_the_dataset_in_its_schema(tmp_path):
    _ok("new", "dataset", "readings", "--dir", str(tmp_path))
    schema = parse_dataset_schema((tmp_path / "datasets" / "readings" / "schema.yaml")
                                  .read_text(encoding="utf-8"))
    assert schema.name == "readings"
    assert schema.key == "id"


# --- everything generated is valid ------------------------------------------

def test_every_generated_python_file_compiles(tmp_path):
    _ok("new", "dataset", "readings", "--dir", str(tmp_path))
    _ok("new", "eval", "quality", "--dir", str(tmp_path))
    _ok("new", "eval-version", "quality", "v2", "--dir", str(tmp_path))

    sources = [p for p in _generated(tmp_path) if p.suffix == ".py"]
    assert len(sources) == 4
    for path in sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_every_generated_yaml_parses_with_the_real_parsers(tmp_path):
    _ok("new", "dataset", "readings", "--dir", str(tmp_path))
    _ok("new", "eval", "quality", "--dir", str(tmp_path))

    schema = parse_dataset_schema((tmp_path / "datasets" / "readings" / "schema.yaml")
                                  .read_text(encoding="utf-8"))
    table = parse_eval_table((tmp_path / "evals" / "quality" / "table.yaml")
                             .read_text(encoding="utf-8"))
    assert [c.name for c in schema.columns] == ["id", "gt"]
    assert table.name == "quality"
    assert [c.name for c in table.columns] == ["score", "coverage"]


def test_generated_scripts_satisfy_the_contract(tmp_path):
    """Discovery registers the scaffolded ingestor, runner and evaluator, with nothing to report."""
    _ok("new", "dataset", "readings", "--dir", str(tmp_path))
    _ok("new", "eval", "quality", "--dir", str(tmp_path))

    registry.discover(str(tmp_path))

    assert registry.problems() == []
    assert registry.get_ingestor("readings") is not None
    assert set(registry.get_evaluators("quality")) == {"v1"}
    inference = registry.get_inference("quality")
    assert inference is not None
    assert set(inference) == {"runner", "process_factory", "config"}


# --- eval versions ----------------------------------------------------------

def test_new_eval_version_adds_a_version_that_is_discovered(tmp_path):
    _ok("new", "eval", "quality", "--dir", str(tmp_path))
    result = _ok("new", "eval-version", "quality", "v2", "--dir", str(tmp_path))

    added = tmp_path / "evals" / "quality" / "v2.py"
    assert f"write {added}" in result.output
    assert "v3.py" in added.read_text(encoding="utf-8")   # points at the next version to cut

    registry.discover(str(tmp_path))
    assert set(registry.get_evaluators("quality")) == {"v1", "v2"}
    assert registry.problems() == []


def test_new_eval_version_rejects_a_name_that_is_not_a_version(tmp_path):
    _ok("new", "eval", "quality", "--dir", str(tmp_path))

    result = runner.invoke(app, ["new", "eval-version", "quality", "hotfix", "--dir", str(tmp_path)])

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "version" in result.output.lower()
    assert not (tmp_path / "evals" / "quality" / "hotfix.py").exists()


# --- idempotence ------------------------------------------------------------

def test_scaffolding_twice_does_not_clobber_existing_files(tmp_path):
    _ok("new", "dataset", "readings", "--dir", str(tmp_path))
    _ok("new", "eval", "quality", "--dir", str(tmp_path))
    edited = tmp_path / "datasets" / "readings" / "ingest.py"
    edited.write_text("# my own work\nINGESTOR = None\n", encoding="utf-8")

    again = _ok("new", "dataset", "readings", "--dir", str(tmp_path))
    _ok("new", "eval", "quality", "--dir", str(tmp_path))
    _ok("init", "--dir", str(tmp_path))

    assert f"skip (exists) {edited}" in again.output
    assert edited.read_text(encoding="utf-8") == "# my own work\nINGESTOR = None\n"
    assert "TODO" in (tmp_path / "evals" / "quality" / "v1.py").read_text(encoding="utf-8")


# --- demo -------------------------------------------------------------------

def test_demo_init_writes_a_wiring_that_passes_discovery(tmp_path):
    result = _ok("demo-init", "--dir", str(tmp_path))
    assert "cairn demo-seed" in result.output

    for name in ("sensor-anomaly-A", "sensor-anomaly-B"):
        parse_dataset_schema((tmp_path / "datasets" / name / "schema.yaml").read_text(encoding="utf-8"))
    for name in ("anomaly", "anomaly-siteB"):
        parse_eval_table((tmp_path / "evals" / name / "table.yaml").read_text(encoding="utf-8"))
    for path in _generated(tmp_path):
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    registry.discover(str(tmp_path))

    assert registry.problems() == []
    assert registry.get_ingestor("sensor-anomaly-A") is not None
    assert set(registry.get_evaluators("anomaly")) == {"v1", "v2"}
    assert registry.get_inference("anomaly-siteB") is not None


def test_the_generated_demo_schema_matches_what_the_demo_ingestor_yields(tmp_path):
    """schema.yaml and the ingestor feeding it must not drift apart.

    Both come from `cairn.demo.pipeline`; an undeclared column would be stored and then
    silently missing from the list view, which is exactly what this catches.
    """
    from cairn.demo.ingest import SyntheticIngest
    from cairn.interfaces.ingestor import IngestContext

    _ok("demo-init", "--dir", str(tmp_path))
    schema = parse_dataset_schema(
        (tmp_path / "datasets" / "sensor-anomaly-A" / "schema.yaml").read_text(encoding="utf-8"))

    ctx = IngestContext(dataset="sensor-anomaly-A", tmpdir=str(tmp_path), created_by="test")
    rows = list(SyntheticIngest().ingest(SyntheticIngest.Input(n=5, seed=1, site="A"), ctx))

    assert rows
    for row in rows:
        assert schema.validate_row(row) == []
        assert set(row) == set(schema.scalar_names())
