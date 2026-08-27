"""Reading and paging through stored results with DuckDB.

Queries run against the files in storage directly, so listing a large dataset never
loads it into memory. Scoring, which needs everything at once, reads the JSON instead.
"""

from __future__ import annotations

import os
from typing import Any

import duckdb


def _needs_httpfs(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "gcs://", "http://", "https://"))


def _connect(path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if not _needs_httpfs(path):
        return con
    con.execute("INSTALL httpfs; LOAD httpfs;")
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    # Prefer the same credential chain as botocore: env, ~/.aws, IAM roles.
    try:
        con.execute("INSTALL aws; LOAD aws;")
        opts = ["TYPE S3", "PROVIDER credential_chain"]
        if endpoint:  # MinIO and friends
            host = endpoint.replace("https://", "").replace("http://", "")
            opts += [f"ENDPOINT '{host}'", "URL_STYLE 'path'"]
            if endpoint.startswith("http://"):
                opts.append("USE_SSL false")
        con.execute(f"CREATE SECRET cairn_s3 ({', '.join(opts)})")
    except Exception:  # noqa: BLE001 - without the aws extension, fall back to env vars
        if region := (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
            con.execute("SET s3_region=?", [region])
        for var, setting in (
            ("AWS_ACCESS_KEY_ID", "s3_access_key_id"),
            ("AWS_SECRET_ACCESS_KEY", "s3_secret_access_key"),
            ("AWS_SESSION_TOKEN", "s3_session_token"),
        ):
            if os.environ.get(var):
                con.execute(f"SET {setting}=?", [os.environ[var]])
        if endpoint:
            con.execute("SET s3_endpoint=?", [endpoint.replace("https://", "").replace("http://", "")])
            if endpoint.startswith("http://"):
                con.execute("SET s3_use_ssl=false")
    return con


def _rows(res: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    cols = [d[0] for d in res.description]
    return [dict(zip(cols, r)) for r in res.fetchall()]


def read_page(
    path: str,
    *,
    limit: int = 50,
    offset: int = 0,
    order_by: str | None = None,
    where_col: str | None = None,
    where_val: Any = None,
) -> list[dict[str, Any]]:
    """Page through a parquet file. `order_by` and `where_col` come from code, not user input."""
    con = _connect(path)
    try:
        sql = "SELECT * FROM read_parquet(?)"
        binds: list[Any] = [path]
        if where_col is not None:
            sql += f" WHERE {where_col} = ?"
            binds.append(where_val)
        if order_by:
            sql += f" ORDER BY {order_by}"
        sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        return _rows(con.execute(sql, binds))
    finally:
        con.close()


def count(path: str, *, where_col: str | None = None, where_val: Any = None) -> int:
    con = _connect(path)
    try:
        sql = "SELECT count(*) FROM read_parquet(?)"
        binds: list[Any] = [path]
        if where_col is not None:
            sql += f" WHERE {where_col} = ?"
            binds.append(where_val)
        row = con.execute(sql, binds).fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


