"""Paging straight from storage with DuckDB.

Listing must not depend on how much is stored, so queries run against the files
themselves. The remote setup is exercised against a stub connection: what matters is
which statements a remote root produces, not that a bucket answers.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cairn.core import query

ROWS = [
    {"eval_id": "e1", "version": "v1", "score": 0.5},
    {"eval_id": "e2", "version": "v2", "score": 0.9},
    {"eval_id": "e3", "version": "v2", "score": 0.7},
]


@pytest.fixture
def parquet(tmp_path) -> str:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), path)
    return str(path)


def test_a_page_is_ordered_and_sliced(parquet):
    page = query.read_page(parquet, limit=2, offset=1, order_by="score DESC")
    assert [r["eval_id"] for r in page] == ["e3", "e1"]


def test_a_page_can_be_restricted_to_one_value(parquet):
    page = query.read_page(parquet, where_col="version", where_val="v2", order_by="eval_id")
    assert [r["eval_id"] for r in page] == ["e2", "e3"]


def test_counting_matches_the_filter(parquet):
    assert query.count(parquet) == 3
    assert query.count(parquet, where_col="version", where_val="v2") == 2


# --- remote roots -----------------------------------------------------------

class _StubConnection:
    """Records the SQL it is given; optionally fails the way a missing extension does."""

    def __init__(self, fail_on: str = "") -> None:
        self.sql: list[str] = []
        self._fail_on = fail_on

    def execute(self, sql: str, binds: list[Any] | None = None):
        self.sql.append(sql if binds is None else f"{sql} {binds}")
        if self._fail_on and self._fail_on in sql:
            raise RuntimeError("extension unavailable")
        return self


@pytest.fixture
def no_aws_env(monkeypatch):
    for var in ("AWS_ENDPOINT_URL", "AWS_REGION", "AWS_DEFAULT_REGION",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def _connect(monkeypatch, path: str, *, fail_on: str = "") -> _StubConnection:
    con = _StubConnection(fail_on=fail_on)
    monkeypatch.setattr(query.duckdb, "connect", lambda *a, **k: con)
    assert query._connect(path) is con
    return con


def test_a_local_root_loads_nothing(monkeypatch, no_aws_env):
    assert _connect(monkeypatch, "/tmp/store/rows.parquet").sql == []


def test_a_remote_root_uses_the_same_credential_chain_as_boto(monkeypatch, no_aws_env):
    sql = _connect(monkeypatch, "s3://bucket/rows.parquet").sql
    assert "INSTALL httpfs; LOAD httpfs;" in sql
    assert "CREATE SECRET cairn_s3 (TYPE S3, PROVIDER credential_chain)" in sql


def test_a_custom_endpoint_is_path_style_and_may_be_plain_http(monkeypatch, no_aws_env):
    """MinIO and friends: the endpoint is not AWS and usually not TLS either."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:9000")
    secret = next(s for s in _connect(monkeypatch, "s3://bucket/rows.parquet").sql if "SECRET" in s)
    assert "ENDPOINT 'localhost:9000'" in secret
    assert "URL_STYLE 'path'" in secret and "USE_SSL false" in secret


def test_without_the_aws_extension_the_settings_come_from_the_environment(monkeypatch, no_aws_env):
    monkeypatch.setenv("AWS_REGION", "ap-northeast-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "token")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:9000")

    sql = _connect(monkeypatch, "s3://bucket/rows.parquet", fail_on="INSTALL aws").sql

    assert "SET s3_region=? ['ap-northeast-1']" in sql
    assert "SET s3_access_key_id=? ['key']" in sql
    assert "SET s3_secret_access_key=? ['secret']" in sql
    assert "SET s3_session_token=? ['token']" in sql
    assert "SET s3_endpoint=? ['localhost:9000']" in sql
    assert "SET s3_use_ssl=false" in sql


def test_a_gcs_or_http_root_is_remote_too():
    assert all(query._needs_httpfs(p) for p in
               ("s3://b/k", "gs://b/k", "gcs://b/k", "http://h/k", "https://h/k"))
    assert not query._needs_httpfs("/local/k")


def test_an_https_endpoint_keeps_tls_on(monkeypatch, no_aws_env):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.example")
    secret = next(s for s in _connect(monkeypatch, "s3://bucket/rows.parquet").sql if "SECRET" in s)
    assert "ENDPOINT 'minio.example'" in secret and "USE_SSL" not in secret


def test_without_the_aws_extension_and_without_credentials_nothing_is_set(monkeypatch, no_aws_env):
    """An instance role supplies the credentials; there is nothing to copy from the env."""
    sql = _connect(monkeypatch, "s3://bucket/rows.parquet", fail_on="INSTALL aws").sql
    assert not [s for s in sql if s.startswith("SET ")]


def test_without_the_aws_extension_an_https_endpoint_keeps_tls(monkeypatch, no_aws_env):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.example")

    sql = _connect(monkeypatch, "s3://bucket/rows.parquet", fail_on="INSTALL aws").sql

    assert "SET s3_endpoint=? ['minio.example']" in sql
    assert "SET s3_use_ssl=false" not in sql
