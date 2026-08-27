"""Datasets: an append-only log, plus snapshots and checkpoints.

No locking is needed because nothing is ever overwritten:

1. Writes always create a new key (`rows/{ulid}.json`), so writers cannot collide.
2. A snapshot records the set of row files at that moment; since files are immutable,
   the same set always yields the same content.
3. Derived files are named after their input (`manifest/{ulid}.parquet`), so compaction
   never conflicts either.
4. Listings are a cache, rebuilt from what is in storage.

Each row file holds upsert and delete operations. The current state is those operations
applied in ULID order, last write wins. `rows/` is the truth; `manifest/` can always be
rebuilt from it.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

from cairn.core import ids
from cairn.core.schema import DatasetSchema, parse_dataset_schema
from cairn.core.storage import Storage
from cairn.core.timeutil import now_iso


def _dir(name: str) -> str:
    return f"datasets/{name}"


# --- schema and listing -----------------------------------------------------

def load_schema(st: Storage, name: str) -> DatasetSchema:
    try:
        raw = st.read_fixed(f"{_dir(name)}/schema.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"no dataset '{name}' in this registry") from None
    return parse_dataset_schema(raw.decode("utf-8"))


def list_datasets(st: Storage) -> list[str]:
    """Names under `datasets/` that actually have a schema.

    The schemas are fetched together rather than asked about one at a time — each
    question would otherwise be its own round trip — and kept, because whoever asks
    which datasets exist is about to read them.
    """
    names = st.ls("datasets")
    have = st.read_many([f"datasets/{n}/schema.yaml" for n in names], missing_ok=True)
    st.remember(have)
    return [n for n in names if f"datasets/{n}/schema.yaml" in have]


def create_dataset(st: Storage, schema: DatasetSchema | str, created_by: str = "") -> str:
    """Create a dataset from its schema. Rows are added later with `append`."""
    if isinstance(schema, str):
        schema = parse_dataset_schema(schema)
    name = schema.name
    if st.exists(f"{_dir(name)}/schema.yaml"):
        raise FileExistsError(f"dataset '{name}' already exists")
    st.write_bytes(f"{_dir(name)}/schema.yaml", _dump_schema(schema).encode("utf-8"))
    return name


# --- appending --------------------------------------------------------------

def append(
    st: Storage, name: str, rows: Iterable[dict[str, Any]], *,
    input: dict[str, Any] | None = None, created_by: str = "",
) -> dict[str, Any]:
    """Append rows that pass schema validation, as upserts."""
    schema = load_schema(st, name)
    ops: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        errs = schema.validate_row(row)
        if errs:
            errors.append({"index": i, "key": row.get(schema.key), "errors": errs})
            continue
        ops.append({"op": "upsert", "row": row})
    meta = {"n_ops": len(ops), "n_rejected": len(errors), "input": input or {}}
    rec = _write_ops(st, name, ops, meta, created_by)
    return {**rec, "rejected": errors}


def delete_rows(st: Storage, name: str, keys: list[str], *, created_by: str = "") -> dict[str, Any]:
    """Delete rows by appending tombstones, keeping the log immutable."""
    ops = [{"op": "delete", "key": k} for k in keys]
    return _write_ops(st, name, ops, {"n_ops": len(ops), "delete_keys": keys}, created_by)


def _write_ops(st: Storage, name: str, ops: list[dict], meta: dict, created_by: str) -> dict[str, Any]:
    iid = ids.ingest_id()
    rows_file = f"{ids.ulid()}.json"
    st.write_json(f"{_dir(name)}/rows/{rows_file}", {"ingest_id": iid, "ops": ops})
    record = {**meta, "ingest_id": iid, "created_at": now_iso(), "created_by": created_by,
              "rows_file": rows_file}
    st.write_json(f"{_dir(name)}/ingests/{iid}.json", record)
    # The rows are committed at this point. A checkpoint is only a cache, so a failure
    # here (or skipping it) must never fail the append or block later ones.
    if _needs_checkpoint(st, name):
        try:
            write_checkpoint(st, name)
        except Exception:  # noqa: BLE001 - reads fall back to merging the row files
            pass
    return record


class Files(NamedTuple):
    """What a dataset is made of, from one listing.

    Appends, checkpoints and the row log live under the same prefix, so asking for
    them separately is three round trips where a bucket would have answered once.
    """

    rows: list[str]
    ingests: list[str]
    manifests: list[str]


def files(st: Storage, name: str) -> Files:
    """List a dataset once and sort the keys into what they are."""
    rows, ingests, manifests = [], [], []
    prefix = f"{_dir(name)}/"
    for key in st.find(_dir(name)):
        kind, _, fname = key[len(prefix):].partition("/")
        if kind == "rows":
            rows.append(fname)
        elif kind == "ingests":
            ingests.append(fname)
        elif kind == "manifest" and fname.endswith(".json"):
            manifests.append(fname)
    return Files(sorted(rows), sorted(ingests), sorted(manifests))


def list_ingests(st: Storage, name: str, listed: Files | None = None) -> list[dict[str, Any]]:
    names = (listed or files(st, name)).ingests
    # One unreadable record must not hide the appends that are fine.
    records = st.read_many_json([f"{_dir(name)}/ingests/{f}" for f in names], skip_unreadable=True)
    out = list(records.values())
    out.sort(key=lambda r: r.get("created_at", ""))
    return out


# --- merging ----------------------------------------------------------------

def _row_files(st: Storage, name: str) -> list[str]:
    return files(st, name).rows


def merge(st: Storage, name: str, keys: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Apply row files in ULID order and return key -> row. `keys=None` reads them all."""
    schema = load_schema(st, name)
    row_files = sorted(keys) if keys is not None else _row_files(st, name)
    # The file set is known up front, so fetch it in one go: read one by one and a
    # merge costs a network round trip per append.
    blobs = st.read_many_json([f"{_dir(name)}/rows/{f}" for f in row_files])
    state: dict[str, dict[str, Any]] = {}
    for f in row_files:
        blob = blobs[f"{_dir(name)}/rows/{f}"]
        for op in blob.get("ops", []):
            if op["op"] == "upsert":
                state[op["row"][schema.key]] = op["row"]
            elif op["op"] == "delete":
                state.pop(op["key"], None)
    return state


def read_rows(st: Storage, name: str, keys: list[str] | None = None) -> list[dict[str, Any]]:
    """Merged rows, either all of them or exactly the ones in a snapshot."""
    return list(merge(st, name, keys).values())


# --- snapshots --------------------------------------------------------------

def snapshot(st: Storage, name: str) -> dict[str, Any]:
    """Record the current content of a dataset.

    `snapshot_id` hashes the merged rows, not the set of files: re-ingesting identical
    rows keeps it the same, while changing or deleting a row changes it. Two runs are
    comparable exactly when their snapshot_id matches.

    `keys` keeps the underlying file set, which reproduces that content exactly, and
    `sample_ids` names what was pinned — free here, since the rows are already merged,
    and it saves everything downstream from merging them again just to ask.
    """
    import json

    keys = _row_files(st, name)
    merged = merge(st, name, keys)
    canon = json.dumps({k: merged[k] for k in sorted(merged)}, sort_keys=True,
                        ensure_ascii=False, default=str)
    sid = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]
    return {"keys": keys, "snapshot_id": sid, "row_count": len(merged),
            "sample_ids": list(merged)}


# --- checkpoints ------------------------------------------------------------

CHECKPOINT_EVERY = 16
"""Row files allowed to pile up before a new checkpoint is written.

Checkpointing rewrites the whole dataset, so doing it on every append makes ingest
quadratic. Reads stay correct in between by merging the uncovered row files.
"""


def _needs_checkpoint(st: Storage, name: str) -> bool:
    cp = _newest_checkpoint(st, name)
    covered = cp[1] if cp else set()
    return len(set(_row_files(st, name)) - covered) >= CHECKPOINT_EVERY

def write_checkpoint(st: Storage, name: str) -> None:
    """Fold the rows visible now into a parquet checkpoint, recording which files it covers."""
    schema = load_schema(st, name)
    cols = schema.scalar_names()
    files = _row_files(st, name)
    merged = list(merge(st, name, files).values())
    records = [{c: r.get(c) for c in cols} for r in merged]
    table = pa.Table.from_pylist(records) if records else pa.table({c: pa.array([], pa.string()) for c in cols})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    cid = ids.ulid()
    st.write_bytes(f"{_dir(name)}/manifest/{cid}.parquet", buf.getvalue())
    st.write_json(f"{_dir(name)}/manifest/{cid}.json", {"covered": files, "count": len(merged),
                                                        "created_at": now_iso()})


class Checkpoint(NamedTuple):
    """A folded copy of the rows, and which row files went into it."""

    file: str
    covered: set[str]


def _checkpoints(st: Storage, name: str, listed: Files | None = None) -> list[Checkpoint]:
    """Every checkpoint, widest coverage last."""
    import json

    names = (listed or files(st, name)).manifests
    blobs = st.read_many([f"{_dir(name)}/manifest/{f}" for f in names], missing_ok=True)
    out = []
    for f in names:
        try:
            meta = json.loads(blobs[f"{_dir(name)}/manifest/{f}"])
        except (KeyError, ValueError):   # a half-written checkpoint is just ignored
            continue
        out.append(Checkpoint(f, set(meta.get("covered", []))))
    out.sort(key=lambda c: (len(c.covered), c.file))
    return out


def _newest_checkpoint(st: Storage, name: str,
                       listed: Files | None = None) -> tuple[str, set[str]] | None:
    """The checkpoint covering the most row files.

    Not simply the newest one: a writer that started earlier can finish later, and its
    checkpoint would cover less while sorting last.
    """
    cps = _checkpoints(st, name, listed)
    if not cps:
        return None
    fname, covered = cps[-1]
    return f"{_dir(name)}/manifest/{fname[:-5]}.parquet", covered


# --- reading ----------------------------------------------------------------

def _scalar(st: Storage, name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cols = load_schema(st, name).scalar_names()
    return [{c: r.get(c) for c in cols} for r in rows]


def rows_page(st: Storage, name: str, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    from cairn.core import query

    listed = files(st, name)
    cp = _newest_checkpoint(st, name, listed)
    if cp is not None:
        parquet_rel, covered = cp
        residual = set(listed.rows) - covered
        if not residual:  # normal path: page in DuckDB without loading everything
            return query.read_page(st.duckdb_path(parquet_rel), limit=limit, offset=offset)
    # Rows written since the last checkpoint: merge in Python instead.
    merged = _scalar(st, name, read_rows(st, name))
    return merged[offset:offset + limit]


def count_rows(st: Storage, name: str, listed: Files | None = None) -> int:
    from cairn.core import query

    listed = listed or files(st, name)
    cp = _newest_checkpoint(st, name, listed)
    if cp is not None and not (set(listed.rows) - cp[1]):
        return query.count(st.duckdb_path(cp[0]))
    return len(merge(st, name, listed.rows))


# --- summary ----------------------------------------------------------------

def info(st: Storage, name: str) -> dict[str, Any]:
    """Everything the listings show about a dataset, from a single listing of it."""
    listed = files(st, name)
    ings = list_ingests(st, name, listed)
    return {"name": name, "rows": count_rows(st, name, listed), "ingests": len(ings),
            "updated": ings[-1]["created_at"] if ings else ""}


# --- vacuum ----------------------------------------------------------------

VACUUM_KEEP = 2
"""Checkpoints left standing, newest first, however old they are."""

VACUUM_MIN_AGE_DAYS = 7
"""How long a superseded checkpoint is left alone before it is collected."""


def vacuum(st: Storage, name: str, *, keep: int = VACUUM_KEEP,
           min_age_days: int = VACUUM_MIN_AGE_DAYS) -> dict[str, Any]:
    """Collect checkpoints that nothing should still be reading.

    This is the one place in cairn where a file a reader already chose can disappear
    underneath it: everything else only ever adds keys. So a checkpoint goes only once
    it is both superseded — `keep` newer ones exist — and older than `min_age_days`,
    which is longer than any read. Only derived files are removed; `rows/` is never
    touched, so snapshots taken by past runs keep resolving.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=min_age_days)
    superseded = _checkpoints(st, name)[:-keep] if keep else _checkpoints(st, name)
    removed = 0
    for cp in superseded:
        if ids.ulid_time(cp.file[:-5]) > cutoff:
            continue                      # still young enough that a reader may hold it
        st.rm(f"{_dir(name)}/manifest/{cp.file}")
        st.rm(f"{_dir(name)}/manifest/{cp.file[:-5]}.parquet")
        removed += 1
    return {"dataset": name, "removed_checkpoints": removed}


def vacuum_all(st: Storage, *, keep: int = VACUUM_KEEP,
               min_age_days: int = VACUUM_MIN_AGE_DAYS) -> list[dict[str, Any]]:
    return [vacuum(st, n, keep=keep, min_age_days=min_age_days) for n in list_datasets(st)]


# --- serialization ----------------------------------------------------------

def _dump_schema(schema: DatasetSchema) -> str:
    import yaml

    return yaml.safe_dump(schema.model_dump(by_alias=True, exclude_none=True),
                          allow_unicode=True, sort_keys=False)
