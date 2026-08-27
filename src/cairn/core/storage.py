"""Object storage access, through fsspec only.

The same code works for `file://`, `s3://` and `gs://`; S3 and GCS just need the
optional `s3fs` / `gcsfs` dependency. Nothing here imports boto3.
"""

from __future__ import annotations

import json
import posixpath
import re
from typing import Any

import fsspec

from cairn.core import ids

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]+:")
"""Two characters at least, so a Windows drive letter is a path and not a protocol."""


def _check_root(root: str) -> None:
    """Refuse a root that names no place, before it quietly becomes the wrong one.

    `s3:::bucket/x` parses as the whole of S3 and `s3:/bucket/x` as a local directory
    of that name; either way the registry comes up empty, or fills up somewhere nobody
    is looking, and the page that says "0 datasets" looks just like an empty registry.
    """
    _, sep, rest = root.partition("://")
    if _SCHEME.match(root) and not sep:
        raise ValueError(f"storage root '{root}' is not a URL: it needs '://', "
                         "as in 's3://bucket/prefix' or 'file:///path/to/dir'")
    if sep and not rest.strip("/"):
        raise ValueError(f"storage root '{root}' names no bucket or directory: "
                         "give the place the registry lives, as in 's3://bucket/prefix'")


class Storage:
    """Reads and writes paths relative to `root`."""

    def __init__(self, root: str) -> None:
        _check_root(root)
        self.root = root.rstrip("/")
        # url_to_fs returns the filesystem and the native path without the protocol.
        # Listings are never cached: another process appends to the same registry, and a
        # cached listing would keep a long-running reader — the web — on the runs that
        # existed when it started.
        self.fs, base = fsspec.core.url_to_fs(self.root, use_listings_cache=False)
        self.base = base.rstrip("/")
        self._fixed: dict[str, bytes] = {}
        self._local = "file" in (self.fs.protocol if isinstance(self.fs.protocol, tuple) else (self.fs.protocol,))

    # --- paths --------------------------------------------------------------
    def _p(self, rel: str) -> str:
        rel = rel.strip("/")
        if ".." in rel.split("/"):
            raise ValueError(f"path escapes the storage root: {rel}")
        return f"{self.base}/{rel}"

    def rel_of(self, abs_path: str) -> str:
        """Turn a native absolute path back into a root-relative one."""
        p = abs_path.rstrip("/")
        if p.startswith(self.base):
            return p[len(self.base):].lstrip("/")
        return p.lstrip("/")

    def duckdb_path(self, rel: str) -> str:
        """Path or URL that DuckDB can read directly.

        DuckDB does not go through fsspec: local roots become plain paths, remote
        roots stay URLs and are read by its httpfs extension.
        """
        rel = rel.strip("/")
        if "://" not in self.root or self.root.startswith("file://"):
            return f"{self.base}/{rel}"
        return f"{self.root}/{rel}"

    # --- basic IO -----------------------------------------------------------
    def exists(self, rel: str) -> bool:
        return bool(self.fs.exists(self._p(rel)))

    def write_bytes(self, rel: str, data: bytes) -> None:
        p = self._p(rel)
        parent = posixpath.dirname(p)
        try:
            self.fs.makedirs(parent, exist_ok=True)
        except Exception:  # noqa: BLE001 - object stores have no directories
            pass
        if self._local:
            # A local write is not atomic, so a reader could see a half-written file.
            # Object stores publish a PUT atomically and need no temporary file.
            tmp = f"{p}.tmp-{ids.ulid()}"
            with self.fs.open(tmp, "wb") as f:
                f.write(data)
            self.fs.mv(tmp, p)
            return
        with self.fs.open(p, "wb") as f:
            f.write(data)

    def read_bytes(self, rel: str) -> bytes:
        with self.fs.open(self._p(rel), "rb") as f:
            return f.read()

    def write_many(self, blobs: dict[str, bytes]) -> None:
        """Write a batch of keys at once.

        The mirror of `read_many`: a bucket takes each PUT as its own round trip, and a
        job writing a chunk of predictions should not pay for them one after another. A
        local root keeps writing one file at a time, where a write is not atomic and the
        temporary-file dance is what makes a reader safe.
        """
        if not blobs:
            return
        if self._local:
            for rel, data in blobs.items():
                self.write_bytes(rel, data)
            return
        self.fs.pipe({self._p(rel): data for rel, data in blobs.items()})

    def read_fixed(self, rel: str) -> bytes:
        """Read a file cairn only ever writes once, and remember it for this instance.

        Schemas and eval tables are created and never rewritten, and a single page reads
        the same one several times. A Storage lives for one request or one command, so
        remembering the bytes cannot go stale the way remembering a listing would.
        """
        if rel not in self._fixed:
            self._fixed[rel] = self.read_bytes(rel)
        return self._fixed[rel]

    def remember(self, blobs: dict[str, bytes]) -> None:
        """Seed `read_fixed` with files already fetched, so nothing is asked for twice."""
        self._fixed.update(blobs)

    def read_many(self, rels: list[str], *, missing_ok: bool = False) -> dict[str, bytes]:
        """Read a known set of keys in one go, relative paths in and out.

        On object storage a read is a network round trip, and doing them one after
        another is what makes a page wait: fsspec fetches a list concurrently. With
        `missing_ok` a key that is not there is left out instead of raising — only that,
        never a read that failed for some other reason.
        """
        if not rels:
            return {}
        paths = [self._p(r) for r in rels]
        # "return" hands back each key's exception instead of dropping it. Omitting them
        # would turn a bucket refusing a read into "that file is not there", and a run
        # that was evaluated would read as one that never was.
        got = self.fs.cat(paths, on_error="return")
        by_path = dict(zip(paths, rels))
        out: dict[str, bytes] = {}
        for path, blob in got.items():
            if isinstance(blob, BaseException):
                if missing_ok and isinstance(blob, FileNotFoundError):
                    continue
                raise blob
            out[by_path[path]] = blob
        return out

    def read_many_json(self, rels: list[str], *, missing_ok: bool = False,
                       skip_unreadable: bool = False) -> dict[str, Any]:
        """`read_many`, decoded, naming the file that would not decode.

        `skip_unreadable` is for listings: one file something else wrote into the store
        must not take a whole page off the screen. Everywhere the content is the answer
        — the rows a run was scored on — it stays off, because quietly dropping data
        would change what was measured.
        """
        out: dict[str, Any] = {}
        for rel, blob in self.read_many(rels, missing_ok=missing_ok).items():
            try:
                out[rel] = json.loads(blob.decode("utf-8"))
            except ValueError:
                if skip_unreadable:
                    continue
                raise ValueError(f"{rel} is not readable JSON") from None
        return out

    @staticmethod
    def json_bytes(obj: Any) -> bytes:
        """How cairn writes JSON, for a caller putting one into a batch."""
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8")

    def write_json(self, rel: str, obj: Any) -> None:
        self.write_bytes(rel, self.json_bytes(obj))

    def read_json(self, rel: str) -> Any:
        return json.loads(self.read_bytes(rel).decode("utf-8"))

    def ls(self, rel: str) -> list[str]:
        """Names directly under `rel`; empty if it does not exist."""
        p = self._p(rel)
        if not self.fs.exists(p):
            return []
        return sorted(n.rstrip("/").split("/")[-1] for n in self.fs.ls(p, detail=False))

    def find(self, rel: str) -> list[str]:
        """Every key under `rel`, however deep, relative to the root."""
        return [self.rel_of(p) for p in self.fs.find(self._p(rel))]

    def rm(self, rel: str) -> None:
        """Delete one key. Used by vacuum for derived artifacts only."""
        p = self._p(rel)
        if self.fs.exists(p):
            self.fs.rm(p)
