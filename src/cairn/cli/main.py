"""Command line entry point: `cairn <group> <command>`."""

from __future__ import annotations

import getpass
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import typer

from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.config import load_config
from cairn.core.storage import Storage

app = typer.Typer(add_completion=False, help="cairn — an append-only registry for evaluations and datasets")
dataset_app = typer.Typer(help="Dataset commands")
eval_app = typer.Typer(help="Eval commands")
app.add_typer(dataset_app, name="dataset")
app.add_typer(eval_app, name="eval")


@app.callback()
def _global(
    root: str = typer.Option(None, "--root", help="Override CAIRN_ROOT (falls back to env / .env)"),
) -> None:
    """Shared by every command: load .env, then let --root override CAIRN_ROOT."""
    from cairn.core.config import load_dotenv

    load_dotenv()
    if root:
        os.environ["CAIRN_ROOT"] = root


def _st() -> Storage:
    return Storage(load_config().root)


def _who() -> str:
    return os.environ.get("CAIRN_USER") or getpass.getuser()


def _load_obj(spec: str) -> Any:
    """Import an object given as `module.path:Attr`.

    The current directory is put on the import path first, because user scripts live there.
    """
    import sys

    mod, _, attr = spec.partition(":")
    if not attr:
        raise typer.BadParameter("format must be 'module.path:Attr'")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    return getattr(importlib.import_module(mod), attr)


# --- root -------------------------------------------------------------------

@app.command()
def root() -> None:
    """Show the current CAIRN_ROOT."""
    cfg = load_config()
    typer.echo(f"CAIRN_ROOT = {cfg.root}")
    typer.echo(f"tag_prefix = {cfg.tag_prefix}")


new_app = typer.Typer(help="Scaffold convention directories")
app.add_typer(new_app, name="new")


@app.command()
def init(dir: Path = typer.Option(Path("."), help="Where to create them")) -> None:
    """Create the datasets/ and evals/ convention directories."""
    from cairn import scaffold

    for line in scaffold.init(dir):
        typer.echo(line)


@new_app.command("dataset")
def new_dataset(name: str, kind: str = typer.Option("tabular"), dir: Path = typer.Option(Path("."))) -> None:
    """Scaffold datasets/<name>/{schema.yaml, ingest.py}."""
    from cairn import scaffold

    for line in scaffold.new_dataset(dir, name, kind=kind):
        typer.echo(line)


@new_app.command("eval")
def new_eval(name: str, dir: Path = typer.Option(Path("."))) -> None:
    """Scaffold evals/<name>/{table.yaml, run.py, v1.py}."""
    from cairn import scaffold

    for line in scaffold.new_eval(dir, name):
        typer.echo(line)


@new_app.command("eval-version")
def new_eval_version(name: str, version: str, dir: Path = typer.Option(Path("."))) -> None:
    """Add evals/<name>/<version>.py — a new eval version (never edit an old one)."""
    from cairn import scaffold

    try:
        lines = scaffold.new_eval_version(dir, name, version)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    for line in lines:
        typer.echo(line)


@app.command()
def check(dir: Path = typer.Option(Path("."), help="Where the convention directories live")) -> None:
    """Check that scripts satisfy the contract (run this first when you start writing)."""
    import os as _os

    from cairn import registry
    from cairn.core.schema import parse_dataset_schema, parse_eval_table

    problems: list[tuple[str, str]] = []
    # 1. the column definitions parse
    for sub, parser, fname in (("datasets", parse_dataset_schema, "schema.yaml"),
                               ("evals", parse_eval_table, "table.yaml")):
        for d in sorted((dir / sub).glob("*")):
            f = d / fname
            if not d.is_dir():
                continue
            if not f.exists():
                problems.append((str(d), f"{fname} is missing"))
                continue
            try:
                parser(f.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001 - report the YAML error to the user
                problems.append((str(f), f"{type(e).__name__}: {e}"))

    # 2. the scripts satisfy the contract
    _os.environ["CAIRN_SCRIPTS"] = str(dir)
    registry.discover(str(dir))
    problems += [(p["where"], p["what"]) for p in registry.problems()]
    # 3. a declared form and the script it belongs to still name the same fields
    problems += [(p["where"], p["what"]) for p in registry.declaration_mismatches()]

    n_ds = len(list((dir / "datasets").glob("*/schema.yaml")))
    n_ev = len(list((dir / "evals").glob("*/table.yaml")))
    if not problems:
        typer.echo("✅ " + f"no problems ({n_ds} dataset(s) / {n_ev} eval(s))")
        return
    typer.echo("❌ " + f"{len(problems)} problem(s):")
    for where, what in problems:
        typer.echo(f"  {where}\n      {what}")
    raise typer.Exit(1)


@app.command()
def docs(page: str = typer.Argument("", help="manual / scripting (omit to list)")) -> None:
    """Print the bundled manual (same content as Help in the web UI)."""
    from cairn import docs as _docs

    if not page:
        typer.echo("usage: cairn docs <page>")
        for p in _docs.pages():
            typer.echo(f"  {p['slug']:12} {p['title']}")
        return
    try:
        typer.echo(_docs.read(page))
    except FileNotFoundError:
        raise typer.BadParameter(f"No such page: {page}") from None


@app.command("demo-init")
def demo_init(dir: Path = typer.Option(Path("."))) -> None:
    """Write the demo out as convention directories (a worked example that actually runs)."""
    from cairn import scaffold

    for line in scaffold.demo_init(dir):
        typer.echo(line)
    typer.echo("Next: `cairn demo-seed` to load data, then `cairn web`.")


@app.command("demo-seed")
def demo_seed() -> None:
    """Seed the bundled demo data into CAIRN_ROOT (a smoke test right after install)."""
    from cairn.demo import seed

    seed(_st())
    typer.echo("Demo data loaded. Open it with `cairn web`.")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload (development)"),
) -> None:
    """Start the web UI."""
    import uvicorn

    import cairn

    load_config()  # fail now if CAIRN_ROOT is missing
    # uvicorn watches the working directory, which holds the scripts. cairn itself lives
    # elsewhere (often an editable install), and a reload that ignores it is a trap.
    reload_dirs = [os.getcwd(), str(Path(cairn.__file__).parent)] if reload else None
    uvicorn.run("cairn.web.app:app", host=host, port=port, reload=reload, reload_dirs=reload_dirs)


# --- dataset ----------------------------------------------------------------

@dataset_app.command("ls")
def dataset_ls() -> None:
    from cairn.core import concurrent as conc

    st = _st()
    names = ds.list_datasets(st)
    for info in conc.each(names, lambda n: ds.info(st, n)):
        typer.echo(f"{info['name']:24} rows={info['rows']:<7} ingests={info['ingests']}")


@dataset_app.command("create")
def dataset_create(schema: Path = typer.Option(..., help="Path to schema.yaml")) -> None:
    st = _st()
    name = ds.create_dataset(st, schema.read_text(encoding="utf-8"), created_by=_who())
    typer.echo(f"created dataset '{name}'")


@dataset_app.command("ingest")
def dataset_ingest(
    name: str,
    jsonl: Path = typer.Option(..., help="JSONL file, one record per line"),
) -> None:
    """Append rows from JSONL (no versions — just append)."""
    st = _st()
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    rec = ds.append(st, name, rows, input={"jsonl": str(jsonl)}, created_by=_who())
    typer.echo(f"added {rec['n_ops']} / rejected {rec['n_rejected']} (ingest {rec['ingest_id']})")
    for r in rec.get("rejected", [])[:10]:
        typer.echo(f"  ✗ {r['key']}: {r['errors']}")


@dataset_app.command("delete")
def dataset_delete(
    name: str,
    key: list[str] = typer.Option(..., "--key", help="Row key to delete; repeat for more"),
) -> None:
    """Delete rows by key, by appending tombstones.

    The rows stay in the log; the dataset simply stops showing them. This changes the
    dataset's content, so runs scored after it are no longer comparable with earlier ones.
    """
    st = _st()
    before = ds.snapshot(st, name)["snapshot_id"]
    rec = ds.delete_rows(st, name, list(key), created_by=_who())
    after = ds.snapshot(st, name)["snapshot_id"]
    typer.echo(f"deleted {rec['n_ops']} (ingest {rec['ingest_id']})")
    typer.echo(f"snapshot {before} -> {after}; runs scored before this are on different data")


@dataset_app.command("show")
def dataset_show(name: str, limit: int = typer.Option(10)) -> None:
    st = _st()
    typer.echo(f"# {name}  rows={ds.count_rows(st, name)}")
    for row in ds.rows_page(st, name, limit=limit):
        typer.echo("  " + json.dumps(row, ensure_ascii=False))


@app.command()
def vacuum(
    keep: int = typer.Option(ds.VACUUM_KEEP, help="Newest checkpoints to leave alone, whatever their age"),
    min_age_days: int = typer.Option(ds.VACUUM_MIN_AGE_DAYS,
                                     help="Leave superseded checkpoints this long, in case a read is using one"),
) -> None:
    """Collect old manifest checkpoints (derived garbage — safe)."""
    st = _st()
    for r in ds.vacuum_all(st, keep=keep, min_age_days=min_age_days):
        typer.echo(f"{r['dataset']:24} removed_checkpoints={r['removed_checkpoints']}")


# --- eval -------------------------------------------------------------------

@eval_app.command("create-table")
def eval_create_table(table: Path = typer.Option(..., help="Path to table.yaml")) -> None:
    name = evals.create_eval_table(_st(), table.read_text(encoding="utf-8"))
    typer.echo(f"created eval table '{name}'")


@eval_app.command("ls")
def eval_ls(table: str = typer.Option("", help="With a table name, list its runs")) -> None:
    st = _st()
    if not table:
        from cairn.core import concurrent as conc

        tables = evals.list_eval_tables(st)
        for t, runs in zip(tables, conc.each(tables, lambda t: evals.list_runs(st, t))):
            typer.echo(f"{t:20} runs={len(runs)}")
        return
    for row in evals.list_runs(st, table):
        typer.echo(f"{row['eval_id']:30} {row.get('evaluator_version','')} "
                   f"{row.get('title') or row.get('comment','')}")


@eval_app.command("create-run")
def eval_create_run(
    table: str,
    dataset: str = typer.Option(...),
    evaluator_version: str = typer.Option(...),
    title: str = typer.Option(..., help="Short name for this run (required)"),
    comment: str = typer.Option("", help="Optional note"),
    config: str = typer.Option("{}", help="Run config as JSON (model version and parameters go here)"),
) -> None:
    """Start a run: take an eval_id and pin the snapshot (predictions are written afterwards)."""
    eid = evals.create_run(
        _st(), table, dataset=dataset,
        evaluator_version=evaluator_version,
        title=title, comment=comment, created_by=_who(), config=json.loads(config),
    )
    typer.echo(eid)


@eval_app.command("targets")
def eval_targets(table: str, eval_id: str) -> None:
    """Print the sample ids this run is pinned to, one per line.

    What a job running elsewhere reads to know its work: the dataset as it was when the
    run started, whatever has been appended since.
    """
    for sample_id in evals.target_ids(_st(), table, eval_id):
        typer.echo(sample_id)


@eval_app.command("put-prediction")
def eval_put_prediction(
    table: str,
    eval_id: str,
    sample_id: str = typer.Option("", help="Sample this prediction belongs to"),
    file: Path = typer.Option(None, help="File holding the prediction ('-' reads stdin)"),
    jsonl: Path = typer.Option(None, help='Batch: one {"sample_id": …, "prediction": …} per line'),
    ext: str = typer.Option("bin", help="Extension of the stored file (cosmetic)"),
) -> None:
    """Write predictions from outside this process, e.g. from an orchestrator task.

    The payload is opaque to cairn: only the evaluator of this table interprets it.
    """
    st = _st()
    try:  # a typo in the eval_id would otherwise write predictions nothing ever reads
        evals.ensure_writable(st, table, eval_id)
    except FileNotFoundError as e:
        raise typer.BadParameter(f"no run '{eval_id}' in '{table}'") from e
    except FileExistsError as e:
        raise typer.BadParameter(str(e)) from e

    if jsonl is not None:
        # One read of the targets for the whole batch: a stale id is a mistake worth
        # naming here rather than a prediction that silently scores nothing.
        targets = set(evals.target_ids(st, table, eval_id))
        batch: list[tuple[str, bytes]] = []
        for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                payload, sid = doc["prediction"], str(doc["sample_id"])
            except (ValueError, KeyError, TypeError) as e:
                raise typer.BadParameter(
                    f'{jsonl} line {i}: expected {{"sample_id": …, "prediction": …}} ({e})') from e
            if sid not in targets:
                raise typer.BadParameter(f"{jsonl} line {i}: '{sid}' is not a sample of this run")
            batch.append((sid, payload.encode("utf-8") if isinstance(payload, str)
                          else json.dumps(payload).encode("utf-8")))
        # The whole file is checked before anything is written, and then written at once.
        n = evals.write_predictions(st, table, eval_id, batch, "json" if ext == "bin" else ext)
        typer.echo(f"wrote {n} prediction(s)")
        return

    if not sample_id or file is None:
        raise typer.BadParameter("give --sample-id and --file, or --jsonl for a batch")
    data = sys.stdin.buffer.read() if str(file) == "-" else file.read_bytes()
    evals.write_prediction(st, table, eval_id, sample_id, data, ext)
    typer.echo(f"wrote prediction for {sample_id}")


@eval_app.command("run")
def eval_run(
    table: str,
    eval_id: str,
    chunk_size: int = typer.Option(16, help="Samples per unit of work"),
) -> None:
    """Perform the inference of a started run here, in this process.

    Uses the script registered for this table (`RUNNER` / `PROCESS_FACTORY` in run.py) and
    writes its predictions. Meant for a small dataset or for trying a script out: a real
    one is inferred by the pipeline, which writes predictions with `put-prediction`.
    """
    from cairn import registry
    from cairn.runner import run_inference

    st = _st()
    registry.ensure_loaded()
    inf = registry.get_inference(table)
    if inf is None:
        raise typer.BadParameter(f"no inference script is registered for '{table}' "
                                 "(RUNNER / PROCESS_FACTORY / CONFIG in run.py; see `cairn check`)")

    n = run_inference(st, table, eval_id, inf, chunk_size=chunk_size)
    typer.echo(f"wrote {n} prediction(s) — score it with `cairn eval score`")


@eval_app.command("score")
def eval_score(
    table: str,
    eval_id: str,
    evaluator: str = typer.Option(..., help="'module.path:EvaluatorClass'"),
) -> None:
    """Score the predictions and write the result."""
    cls = _load_obj(evaluator)
    row = evals.score_run(_st(), table, eval_id, cls())
    typer.echo(json.dumps(row, ensure_ascii=False, indent=2))


@eval_app.command("rescore")
def eval_rescore(
    table: str,
    eval_id: str,
    evaluator: str = typer.Option(..., help="'module.path:EvaluatorClass'"),
    evaluator_version: str = typer.Option(..., help="Version this evaluator implements, e.g. v2"),
    title: str = typer.Option("", help="Short name for the new run (defaults to the original's)"),
) -> None:
    """Score an existing run's predictions with another evaluator version.

    Starts a new run so both results stay side by side; the original is untouched.
    """
    cls = _load_obj(evaluator)
    new_id = evals.rescore(_st(), table, eval_id, cls(),
                           evaluator_version=evaluator_version, title=title, created_by=_who())
    typer.echo(new_id)


@eval_app.command("withdraw")
def eval_withdraw(
    table: str,
    eval_id: str,
    reason: str = typer.Option("", help="Why, for whoever reads the run later"),
) -> None:
    """Take a run out of the listings: a mistake, or a job that is never coming back.

    The run itself stays on the ledger — metadata, predictions, and a note saying it was
    withdrawn — but it stops being read as a result. This is final; start a run to retry.
    """
    note = evals.withdraw_run(_st(), table, eval_id, created_by=_who(), reason=reason)
    typer.echo(f"withdrew {eval_id} ({note['withdrawn_at']})")


@eval_app.command("show")
def eval_show(table: str, eval_id: str) -> None:
    run = evals.get_run(_st(), table, eval_id)
    typer.echo(json.dumps(run["meta"], ensure_ascii=False, indent=2))
    if run["row"]:
        typer.echo("\n# result row"); typer.echo(json.dumps(run["row"], ensure_ascii=False, indent=2))
    if run["report_md"]:
        typer.echo("\n# report.md"); typer.echo(run["report_md"])


def main() -> None:
    """Console entry point: report what the registry refused, not how it found out.

    These commands are meant to be called from an orchestrator, so a failure ends up in
    a job log. A traceback there buries the one useful line under cairn's own frames,
    and its absolute paths say less than the name of what was asked for. Set
    `CAIRN_TRACEBACK=1` to get the traceback back while debugging a script.
    """
    try:
        app()
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        if os.environ.get("CAIRN_TRACEBACK"):
            raise
        from rich.console import Console
        from rich.panel import Panel

        Console(stderr=True).print(Panel(str(e), title="Error", title_align="left", border_style="red"))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
