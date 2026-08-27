"""The bundled manual, shown by `cairn docs` and by Help in the web UI.

It ships inside the package so it is available wherever cairn is installed. Pages exist
per language and fall back to the other one rather than 404.
"""

from __future__ import annotations

from pathlib import Path

from cairn import i18n

_DIR = Path(__file__).parent

# Listed in reading order. Titles are English source strings, translated on display.
ORDER = [
    ("manual", "How to use"),
    ("scripting", "Writing scripts"),
]


def _path(slug: str, code: str | None = None) -> Path:
    code = code or i18n.lang()
    en = _DIR / f"{slug}.en.md"
    ja = _DIR / f"{slug}.md"
    if code == "ja":
        return ja if ja.exists() else en
    return en if en.exists() else ja


def pages() -> list[dict[str, str]]:
    return [{"slug": s, "title": i18n.t(t)} for s, t in ORDER if _path(s).exists()]


def read(slug: str) -> str:
    """Markdown of one page. Only the pages listed above can be read."""
    if slug not in {s for s, _ in ORDER}:
        raise FileNotFoundError(slug)
    return _path(slug).read_text(encoding="utf-8")


def title(slug: str) -> str:
    return next((i18n.t(t) for s, t in ORDER if s == slug), slug)
