"""Evaluation runs: creating them, taking predictions, scoring, and listing results.

A run writes predictions only; metrics are computed when it is scored, because measures
like mAP or AUC cannot be decomposed per sample. Creating a run pins a snapshot of the
dataset, and scoring reads exactly that snapshot, so a result always refers to a known
state of the data.

Listings are read straight from the result files with DuckDB — there is no index to
keep in sync.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NamedTuple
from urllib.parse import quote, unquote

from pydantic import BaseModel

from cairn.core import dataset as ds
from cairn.core import ids
from cairn.core.records import EvalResult
from cairn.core.schema import EvalTableSchema, parse_eval_table
from cairn.core.storage import Storage
from cairn.core.timeutil import now_iso
from cairn.interfaces.evaluator import DatasetView, EvalContext, Evaluator, PredictionView


class _EmptyConfig(BaseModel):
    """Passed to evaluators that declare no `Config`."""

    model_config = {"extra": "ignore"}


def _tdir(table: str) -> str:
    return f"evals/{table}"


def _rdir(table: str, eval_id: str) -> str:
    return f"evals/{table}/runs/{eval_id}"


# --- eval tables ------------------------------------------------------------

def create_eval_table(st: Storage, schema: EvalTableSchema | str) -> str:
    if isinstance(schema, str):
        schema = parse_eval_table(schema)
    if st.exists(f"{_tdir(schema.name)}/table.yaml"):
        raise FileExistsError(f"eval table '{schema.name}' already exists")
    import yaml

    st.write_bytes(
        f"{_tdir(schema.name)}/table.yaml",
        yaml.safe_dump(schema.model_dump(exclude_none=True), allow_unicode=True, sort_keys=False).encode(),
    )
    return schema.name


def load_eval_table(st: Storage, table: str) -> EvalTableSchema:
    try:
        raw = st.read_fixed(f"{_tdir(table)}/table.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"no eval table '{table}' in this registry") from None
    return parse_eval_table(raw.decode("utf-8"))


def list_eval_tables(st: Storage) -> list[str]:
    """Names under `evals/` that actually have a table. One fetch, not one per name."""
    names = st.ls("evals")
    have = st.read_many([f"evals/{n}/table.yaml" for n in names], missing_ok=True)
    st.remember(have)
    return [n for n in names if f"evals/{n}/table.yaml" in have]


# --- creating runs ----------------------------------------------------------

def create_run(
    st: Storage,
    table: str,
    *,
    dataset: str,
    evaluator_version: str,
    title: str,
    comment: str = "",
    created_by: str = "",
    code_commit: str = "",
    config: dict[str, Any] | None = None,
    code: dict[str, str] | None = None,
) -> str:
    """Start a run: take an eval_id and pin a snapshot of the dataset.

    `title` is required: it is how the run is identified in the list view.
    """
    if not str(title).strip():
        raise ValueError("title is required (a short name identifying this run)")
    load_eval_table(st, table)
    snap = ds.snapshot(st, dataset)
    eval_id = ids.eval_id()
    meta = {
        "eval_id": eval_id,
        "dataset": dataset,
        "snapshot_id": snap["snapshot_id"],
        "row_count": snap["row_count"],
        "evaluator_version": evaluator_version,
        "code_commit": code_commit,
        "title": title,
        "comment": comment,
        "created_by": created_by,
        "created_at": now_iso(),
        "status": "created",
        "config": config or {},
    }
    st.write_json(f"{_rdir(table, eval_id)}/_meta.json", meta)
    st.write_json(f"{_rdir(table, eval_id)}/_snapshot.json", snap)
    for fname, content in (code or {}).items():
        st.write_bytes(f"{_rdir(table, eval_id)}/code/{fname}", content.encode("utf-8"))
    return eval_id


class _State(NamedTuple):
    """Where a run stands, read in one go."""

    meta: dict[str, Any]
    snapshot: dict[str, Any]
    scored: bool
    withdrawn: bool


def _state(st: Storage, table: str, eval_id: str, *, want_snapshot: bool = False) -> _State:
    """Read a run and where it stands, in one fetch.

    Whether it exists, whether it was evaluated and whether it was withdrawn are files
    under one prefix; asked one at a time they are three round trips before any work
    starts. What to do about each answer is the caller's, and the order it asks in shows
    in the error the caller gets.
    """
    import json

    rdir = _rdir(table, eval_id)
    meta_p, row_p, note_p, snap_p = (f"{rdir}/_meta.json", f"{rdir}/result/row.json",
                                     f"{rdir}/_withdrawn.json", f"{rdir}/_snapshot.json")
    want = [meta_p, row_p, note_p] + ([snap_p] if want_snapshot else [])
    got = st.read_many(want, missing_ok=True)
    if meta_p not in got:
        raise FileNotFoundError(f"no run '{eval_id}' in eval table '{table}'")
    if want_snapshot and snap_p not in got:
        raise FileNotFoundError(
            f"run '{eval_id}' in eval table '{table}' has no snapshot: it never finished starting")
    return _State(json.loads(got[meta_p].decode("utf-8")),
                  json.loads(got[snap_p].decode("utf-8")) if snap_p in got else {},
                  row_p in got, note_p in got)


def get_meta(st: Storage, table: str, eval_id: str) -> dict[str, Any]:
    return _read_run(st, table, eval_id, "_meta.json")


def get_snapshot(st: Storage, table: str, eval_id: str) -> dict[str, Any]:
    return _read_run(st, table, eval_id, "_snapshot.json")


def _read_run(st: Storage, table: str, eval_id: str, name: str) -> dict[str, Any]:
    """Read one of a run's own files, naming the run when it is not there.

    The path is cairn's business; what the caller mistyped is a table or an eval_id.
    """
    try:
        return st.read_json(f"{_rdir(table, eval_id)}/{name}")
    except FileNotFoundError:
        raise FileNotFoundError(f"no run '{eval_id}' in eval table '{table}'") from None


# --- predictions ------------------------------------------------------------

def _pred_name(sample_id: str) -> str:
    """File name for a sample id.

    Sample ids come from dataset keys, so they can contain anything. Percent-encode
    everything that would change the path — dots included, so the only dot left in the
    file name is the one before the extension — and keep it reversible.
    """
    return quote(sample_id, safe="").replace(".", "%2E")


def write_prediction(st: Storage, table: str, eval_id: str, sample_id: str, data: bytes,
                     ext: str = "bin") -> None:
    """Write the prediction for one sample.

    The payload is opaque bytes; only the evaluator interprets it. `ext` is cosmetic.
    """
    st.write_bytes(f"{_rdir(table, eval_id)}/predictions/{_pred_name(sample_id)}.{ext}", data)


def write_predictions(st: Storage, table: str, eval_id: str,
                      items: list[tuple[str, bytes]], ext: str = "bin") -> int:
    """Write a chunk of predictions in one go. Returns how many were written."""
    st.write_many({f"{_rdir(table, eval_id)}/predictions/{_pred_name(sid)}.{ext}": data
                   for sid, data in items})
    return len(items)


def count_predictions(st: Storage, table: str, eval_id: str) -> int:
    return len(st.ls(f"{_rdir(table, eval_id)}/predictions"))


PREDICTION_BATCH = 64
"""Predictions fetched per round trip. A payload can be an image, so not all at once."""


def _prediction_loader(st: Storage, table: str, eval_id: str, names: list[str] | None = None):
    def _iter() -> Iterator[tuple[str, bytes]]:
        dir_ = f"{_rdir(table, eval_id)}/predictions"
        listed = st.ls(dir_) if names is None else names
        for i in range(0, len(listed), PREDICTION_BATCH):
            batch = listed[i:i + PREDICTION_BATCH]
            blobs = st.read_many([f"{dir_}/{n}" for n in batch])
            for fname in batch:
                # Names are `{quoted sample id}.{ext}`; the id itself never contains a dot
                # once quoted, so splitting on the first one is exact.
                yield unquote(fname.split(".", 1)[0]), blobs[f"{dir_}/{fname}"]

    return _iter


# --- scoring ----------------------------------------------------------------

def score_run(st: Storage, table: str, eval_id: str, evaluator: Evaluator,
              *, evaluator_version: str | None = None) -> dict[str, Any]:
    """Score a run's predictions and write the result.

    `evaluator_version` must match the run's: a result is only meaningful next to the
    version that produced it. To score the same predictions a different way, use
    `rescore`, which issues a run of its own.
    """
    from cairn.core.schema import COMMON_EVAL_COLUMNS

    schema = load_eval_table(st, table)
    state = _state(st, table, eval_id, want_snapshot=True)
    meta, snap = state.meta, state.snapshot
    if evaluator_version is not None and evaluator_version != meta["evaluator_version"]:
        raise ValueError(
            f"this run is scored with {meta['evaluator_version']}, not {evaluator_version}. "
            f"Use rescore to score the same predictions with another version.")
    if state.scored:
        raise FileExistsError(
            f"run {eval_id} already has a result. Scoring again would rewrite a record others "
            f"may have read; rescore scores the same predictions as a run of its own.")
    if state.withdrawn:
        raise FileExistsError(f"run {eval_id} was withdrawn. Start a run instead.")

    rows = ds.read_rows(st, meta["dataset"], snap["keys"])  # exactly the pinned snapshot
    dschema = ds.load_schema(st, meta["dataset"])
    dkey = dschema.key
    # Nested columns stay out of the list view and are handed to the evaluator on demand.
    nested_fields = list(dschema.nested.keys())
    by_key = {r.get(dkey): r for r in rows}

    def _frames_loader(sid: str, field: str | None) -> list[dict[str, Any]]:
        if field is None:
            if len(nested_fields) > 1:
                raise ValueError(f"this dataset declares several nested tables; pass one of {nested_fields} to frames()")
            field = nested_fields[0]
        elif field not in nested_fields:
            raise ValueError(f"'{field}' is not a nested table of this dataset")
        value = (by_key.get(sid) or {}).get(field)
        return value if isinstance(value, list) else []

    dsv = DatasetView(rows=rows, key=dkey, frames_loader=_frames_loader if nested_fields else None)
    # One listing of the predictions: how many there are and what they say come from it.
    pred_names = st.ls(f"{_rdir(table, eval_id)}/predictions")
    predv = PredictionView(loader=_prediction_loader(st, table, eval_id, pred_names))

    # `Config` is optional for evaluators.
    cfg_cls = getattr(type(evaluator), "Config", None) or _EmptyConfig
    ctx = EvalContext(
        dataset=dsv, predictions=predv, config=cfg_cls(**meta.get("config", {})),
        expected_n=len(rows), actual_n=len(pred_names),
    )
    result: EvalResult = evaluator.score(ctx)

    errs = schema.validate_result_row(result.row)
    if errs:
        raise ValueError(f"the row returned by score does not match table.yaml: {errs}")
    final_row = {**result.row, **{k: meta.get(k) for k in COMMON_EVAL_COLUMNS}}
    final_row["status"] = "scored"

    meta["status"] = "scored"
    meta["scored_at"] = now_iso()
    # Free-form JSON from the evaluator, kept beside the run config.
    if result.metadata:
        meta["metadata"] = result.metadata

    rdir = _rdir(table, eval_id)
    # Everything a result is made of goes up together; only the marker below is ordered.
    st.write_many({
        f"{rdir}/result/metrics.parquet": _metrics_parquet(result),
        f"{rdir}/result/report.md": result.report_md.encode("utf-8"),
        f"{rdir}/_meta.json": st.json_bytes(meta),
        **{f"{rdir}/result/assets/{aname}": adata for aname, adata in result.assets.items()},
    })

    # Last: `row.json` is what makes a run evaluated — for the listings, and for the
    # refusal above. Writing it after everything else means a scoring that died halfway
    # leaves the run unevaluated and can simply be done again.
    st.write_json(f"{rdir}/result/row.json", final_row)
    return final_row


def rescore(st: Storage, table: str, eval_id: str, evaluator: Evaluator, *,
            evaluator_version: str, title: str = "", created_by: str = "") -> str:
    """Score an existing run's predictions with another evaluator version.

    The predictions and the snapshot are what they were; only the way they are read
    changes. That is a different result, so it becomes a run of its own and the original
    stays exactly as it was.
    """
    meta = get_meta(st, table, eval_id)
    if evaluator_version == meta["evaluator_version"]:
        raise ValueError(f"this run is already scored with {evaluator_version}")

    new_id = ids.eval_id()
    new_meta = {**meta, "eval_id": new_id, "evaluator_version": evaluator_version,
                "title": title or meta.get("title", ""), "created_by": created_by or meta.get("created_by", ""),
                "created_at": now_iso(), "status": "created",
                "rescored_from": eval_id}
    new_meta.pop("scored_at", None)
    new_meta.pop("metadata", None)
    st.write_many({f"{_rdir(table, new_id)}/_meta.json": st.json_bytes(new_meta),
                   f"{_rdir(table, new_id)}/_snapshot.json": st.json_bytes(get_snapshot(st, table, eval_id))})
    # The predictions are the run's evidence; copy them so the new run stands on its own.
    # A chunk at a time: one at a time is two round trips per prediction, and a run has
    # as many predictions as the dataset has rows.
    src, dst = f"{_rdir(table, eval_id)}/predictions", f"{_rdir(table, new_id)}/predictions"
    names = st.ls(src)
    for i in range(0, len(names), PREDICTION_BATCH):
        batch = names[i:i + PREDICTION_BATCH]
        blobs = st.read_many([f"{src}/{n}" for n in batch])
        st.write_many({f"{dst}/{n}": blobs[f"{src}/{n}"] for n in batch})

    score_run(st, table, new_id, evaluator, evaluator_version=evaluator_version)
    return new_id


def _metrics_parquet(result: EvalResult) -> bytes:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    recs = [m.model_dump() for m in result.metrics]
    if recs:
        arrow = pa.Table.from_pylist(recs)
    else:
        arrow = pa.table({"name": pa.array([], pa.string()), "value": pa.array([], pa.float64()),
                          "sample_id": pa.array([], pa.string()), "sub_id": pa.array([], pa.string())})
    buf = io.BytesIO()
    pq.write_table(arrow, buf)
    return buf.getvalue()


def read_metrics(st: Storage, table: str, eval_id: str) -> list[dict[str, Any]]:
    import io

    import pyarrow.parquet as pq

    path = f"{_rdir(table, eval_id)}/result/metrics.parquet"
    if not st.exists(path):
        return []
    return pq.read_table(io.BytesIO(st.read_bytes(path))).to_pylist()


def read_asset(st: Storage, table: str, eval_id: str, name: str) -> bytes:
    return st.read_bytes(f"{_rdir(table, eval_id)}/result/assets/{name}")


def get_run(st: Storage, table: str, eval_id: str) -> dict[str, Any]:
    import json

    rdir = _rdir(table, eval_id)
    meta_path, row_path, report_path = f"{rdir}/_meta.json", f"{rdir}/result/row.json", f"{rdir}/result/report.md"
    # Four files of one run: asked for one at a time, that is a round trip each, and
    # "is it there?" doubles it.
    got = st.read_many([meta_path, row_path, report_path], missing_ok=True)
    if meta_path not in got:
        raise FileNotFoundError(f"no run '{eval_id}' in eval table '{table}'")
    return {
        "meta": json.loads(got[meta_path].decode("utf-8")),
        "row": json.loads(got[row_path].decode("utf-8")) if row_path in got else None,
        "report_md": got[report_path].decode("utf-8") if report_path in got else "",
        "predictions": count_predictions(st, table, eval_id),
    }


# --- listing ----------------------------------------------------------------

def run_ids(st: Storage, table: str) -> list[str]:
    """The runs a table has, from one listing that stops above the predictions."""
    return st.ls(f"{_tdir(table)}/runs")


class Runs(NamedTuple):
    """What the listings need to know about a table's runs.

    A run is running or evaluated, and a withdrawn one is neither; all three are the
    same question about the same runs, so it is asked once.
    """

    ids: list[str]
    scored: dict[str, dict[str, Any]]
    withdrawn: set[str]

    def rows(self, evaluator_version: str | None = None) -> list[dict[str, Any]]:
        """The results, newest first."""
        rows = [r for r in self.scored.values()
                if not evaluator_version or r.get("evaluator_version") == evaluator_version]
        rows.sort(key=lambda r: (r.get("created_at", ""), r.get("eval_id", "")), reverse=True)
        return rows

    def pending(self) -> list[str]:
        """Started, with no result and not withdrawn."""
        return [e for e in self.ids if e not in self.scored and e not in self.withdrawn]


def runs_view(st: Storage, table: str) -> Runs:
    """One listing of the runs, and one fetch of what each of them holds."""

    ids = run_ids(st, table)
    # A result nobody can read is not a result: that run shows as running, where a
    # person can see it, rather than taking the whole table's listing down.
    rows = st.read_many_json([f"{_rdir(table, e)}/result/row.json" for e in ids],
                             missing_ok=True, skip_unreadable=True)
    scored = {e: rows[k] for e in ids if (k := f"{_rdir(table, e)}/result/row.json") in rows}
    # Withdrawing removes the result, so only a run without one can have been withdrawn:
    # asking about the rest would be one request per run for an answer already known.
    rest = [e for e in ids if e not in scored]
    notes = st.read_many([f"{_rdir(table, e)}/_withdrawn.json" for e in rest], missing_ok=True)
    withdrawn = {e for e in rest if f"{_rdir(table, e)}/_withdrawn.json" in notes}
    return Runs(ids, scored, withdrawn)


def count_runs(st: Storage, table: str) -> int:
    """How many results the table holds. For one version's worth, count `list_runs`."""
    return len(runs_view(st, table).scored)


def withdraw_run(st: Storage, table: str, eval_id: str, *, created_by: str = "",
                 reason: str = "") -> dict[str, Any]:
    """Take a run out of the listings without erasing that it happened.

    For a run that should not be read as a result: a mistake, or one whose job died and
    is never coming back. The record stays — its metadata, its predictions and this note —
    but the result row is removed, which is what every listing reads, so the run is neither
    running nor evaluated any more. It is final: to try again, start a run.
    """
    get_meta(st, table, eval_id)     # a run that does not exist cannot be withdrawn
    note = {"eval_id": eval_id, "withdrawn_at": now_iso(), "withdrawn_by": created_by,
            "reason": reason}
    # The result is taken away first. A withdrawal that dies in the middle must never
    # leave a run that still reads as a result — without the note it reads as running,
    # which is on the screen and can simply be withdrawn again.
    st.rm(f"{_rdir(table, eval_id)}/result/row.json")
    st.write_json(f"{_rdir(table, eval_id)}/_withdrawn.json", note)
    return note


def _run_markers(st: Storage, table: str) -> tuple[set[str], set[str]]:
    """Which runs have a result and which were withdrawn, from a single listing.

    One listing names the runs, and what each one holds is fetched for all of them at
    once — never a walk through the predictions, of which there are as many as the
    dataset has rows.
    """
    view = runs_view(st, table)
    return set(view.scored), view.withdrawn


def withdrawn_runs(st: Storage, table: str) -> set[str]:
    """Ids of the runs taken out of the listings."""
    return _run_markers(st, table)[1]


def pending_runs(st: Storage, table: str, view: Runs | None = None) -> list[dict[str, Any]]:
    """Runs that were started but not evaluated yet, newest first.

    A run exists from `create_run`; a result appears only when `score_run` writes one.
    This says which runs are in between, and nothing about their progress: whether the
    inference has finished is known where it runs, not here. Even "something has arrived"
    would not mean finished, so it is not asked.
    """
    pending = (view or runs_view(st, table)).pending()
    # One fetch for the lot; a run being written right now has no _meta.json yet and is
    # simply left out.
    metas = st.read_many_json([f"{_rdir(table, e)}/_meta.json" for e in pending],
                              missing_ok=True, skip_unreadable=True)
    out = list(metas.values())
    out.sort(key=lambda m: (m.get("created_at", ""), m["eval_id"]), reverse=True)
    return out


def target_ids(st: Storage, table: str, eval_id: str) -> list[str]:
    """Sample ids this run is pinned to: the dataset as it was when the run started.

    What an orchestrator needs in order to write predictions from somewhere else. One
    small read: the snapshot names them. Runs pinned before it did are read the long way.
    """
    snap = get_snapshot(st, table, eval_id)
    if "sample_ids" in snap:
        return list(snap["sample_ids"])
    dataset = get_meta(st, table, eval_id)["dataset"]
    rows = ds.read_rows(st, dataset, snap["keys"])
    return [r[ds.load_schema(st, dataset).key] for r in rows]


def ensure_writable(st: Storage, table: str, eval_id: str) -> None:
    """Refuse to add predictions to a run that is no longer collecting them.

    A result stands on the predictions that were there when it was written; adding more
    afterwards would leave a run whose evidence and result disagree.
    """
    state = _state(st, table, eval_id)
    if state.scored:
        raise FileExistsError(f"run {eval_id} is already evaluated. Start a run for new predictions.")
    if state.withdrawn:
        raise FileExistsError(f"run {eval_id} was withdrawn. Start a run instead.")


def list_runs(st: Storage, table: str, evaluator_version: str | None = None) -> list[dict[str, Any]]:
    """All scored runs, newest first.

    Every caller of this wants the whole set — to count it, to find the newest, to
    compare two scored the same way — so it is one listing and one fetch rather than a
    query that would have to read every result to sort them anyway.
    """
    return runs_view(st, table).rows(evaluator_version)
