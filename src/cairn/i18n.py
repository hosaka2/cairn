"""Language of the prose: the manual pages and the help beside the YAML forms.

The rest of the interface is English and is not translated — the labels are short
technical terms, and keeping them fixed makes screenshots and reports consistent.

English source strings are the catalog keys, so a missing translation shows the English
text rather than breaking. The language comes from `CAIRN_LANG` and nothing else: the OS
locale is left out on purpose, so the same command reads the same way on every machine.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

_LOCALES = Path(__file__).parent / "locales"

SUPPORTED = ("en", "ja")


def lang() -> str:
    """Current language; anything unsupported falls back to English."""
    from cairn.core.config import load_dotenv

    load_dotenv()  # so CAIRN_LANG can be set in .env, like CAIRN_ROOT
    code = (os.environ.get("CAIRN_LANG") or "").split(".")[0].split("_")[0].lower()
    return code if code in SUPPORTED else "en"


@cache
def _catalog(code: str) -> dict[str, str]:
    f = _LOCALES / f"{code}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def t(text: str, **kw: object) -> str:
    """Translate an English source string, falling back to it when untranslated.

    Supports `{name}` placeholders: `t("{n} rows", n=418)`.
    """
    code = lang()
    out = _catalog(code).get(text, text) if code != "en" else text
    if not kw:
        return out
    try:
        return out.format(**kw)
    except (KeyError, IndexError, ValueError):
        return text.format(**kw)  # a broken translation must not break the page


def missing(code: str, texts: list[str]) -> list[str]:
    """Source strings that have no translation, for completeness checks."""
    cat = _catalog(code)
    return [s for s in texts if s not in cat]
