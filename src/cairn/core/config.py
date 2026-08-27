"""Runtime settings, resolved from the environment or a `.env` file.

Deployment-specific values (bucket, endpoint, header names) live here and nowhere
else in the code base.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DOTENV_LOADED = False


def load_dotenv(path: str | Path = ".env") -> None:
    """Load `.env` into the environment once, without overriding existing values."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.removeprefix("export ").partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


@dataclass(frozen=True)
class Config:
    """Resolved settings for one cairn process."""

    root: str
    """Storage root, e.g. file:///var/cairn, s3://bucket/cairn, gs://bucket/cairn."""
    tag_prefix: str = "cairn/"
    """Prefix for tags handed to an orchestrator."""
    auth_header: str = "X-Forwarded-User"
    """Header carrying the user name, used for `created_by`."""


def load_config() -> Config:
    """Build a Config from the environment. CAIRN_ROOT is required."""
    load_dotenv()
    root = os.environ.get("CAIRN_ROOT")
    if not root:
        raise RuntimeError(
            "CAIRN_ROOT is not set. Pass --root, export it, or put it in .env, "
            "e.g. CAIRN_ROOT=file:///var/cairn or s3://bucket/cairn."
        )
    return Config(
        root=root,        # Storage normalises it; keep what was written for error messages
        tag_prefix=os.environ.get("CAIRN_TAG_PREFIX", "cairn/"),
        auth_header=os.environ.get("CAIRN_AUTH_HEADER", "X-Forwarded-User"),
    )
