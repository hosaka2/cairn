"""Display language.

Only the long-form help matters here: the manual pages and the explanations beside the
YAML forms. Everything else in the UI is English, untranslated by design.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cairn import docs, i18n

SRC = Path(__file__).resolve().parents[1] / "src" / "cairn"


@pytest.fixture(autouse=True)
def _clear_cache():
    i18n._catalog.cache_clear()
    yield
    i18n._catalog.cache_clear()


def test_default_is_english(monkeypatch):
    """The OS locale must not decide it: the same command reads the same everywhere."""
    monkeypatch.delenv("CAIRN_LANG", raising=False)
    monkeypatch.setenv("LANG", "ja_JP.UTF-8")
    monkeypatch.setenv("LC_ALL", "ja_JP.UTF-8")
    assert i18n.lang() == "en"
    assert i18n.t("Heavy assets") == "Heavy assets"


def test_japanese_when_requested(monkeypatch):
    monkeypatch.setenv("CAIRN_LANG", "ja")
    assert i18n.lang() == "ja"
    assert i18n.t("Heavy assets") == "重いアセット"


def test_unknown_text_falls_back(monkeypatch):
    monkeypatch.setenv("CAIRN_LANG", "ja")
    assert i18n.t("no translation for this") == "no translation for this"


def test_docs_follow_the_language(monkeypatch):
    monkeypatch.setenv("CAIRN_LANG", "ja")
    assert docs.read("manual").startswith("# 使い方")
    monkeypatch.setenv("CAIRN_LANG", "en")
    assert docs.read("manual").startswith("# Usage")


def _translated_texts() -> list[str]:
    """Every string still passed through `t()` in the templates, plus the page titles."""
    texts: set[str] = set()
    for f in (SRC / "web" / "templates").glob("*.html"):
        texts |= {m[1] for m in re.findall(r"""\bt\(\s*(["'])(.*?)\1""", f.read_text(), re.DOTALL)}
    texts |= {title for _, title in docs.ORDER}
    return sorted(texts)


def test_japanese_catalog_is_complete():
    """A gap would put an English sentence in the middle of the Japanese help."""
    texts = _translated_texts()
    assert len(texts) > 20, "extraction failed"
    assert i18n.missing("ja", texts) == []


def test_japanese_catalog_has_no_dead_entries():
    cat = json.loads((SRC / "locales" / "ja.json").read_text(encoding="utf-8"))
    assert set(cat) - set(_translated_texts()) == set()


def test_ui_chrome_is_not_translated():
    """Buttons, headers and labels stay English; only the help asides use `t()`."""
    aside = re.compile(r'<aside class="yaml-help">.*?</aside>', re.DOTALL)
    for path in (SRC / "web" / "templates").glob("*.html"):
        chrome = aside.sub("", path.read_text())   # the help asides are the exception
        assert not re.search(r"\bt\(", chrome), f"{path.name} still translates chrome"


def test_no_translation_repeats_markup_the_template_supplies():
    """A `<code>` in the catalog prints twice: the template already emits its own."""
    cat = json.loads((SRC / "locales" / "ja.json").read_text(encoding="utf-8"))
    assert [k for k, v in cat.items() if "<code>" in v] == []


def test_a_language_with_no_catalog_file_falls_back_to_the_source_text():
    assert i18n._catalog("de") == {}


def test_a_placeholder_the_translation_got_wrong_does_not_break_the_page(monkeypatch):
    """A translator can mistype `{n}`; the sentence still has to render."""
    monkeypatch.setenv("CAIRN_LANG", "ja")
    monkeypatch.setattr(i18n, "_catalog", lambda code: {"{n} rows": "{lines} 行"})

    assert i18n.t("{n} rows", n=418) == "418 rows"
    assert i18n.t("{n} rows") == "{lines} 行"      # nothing to substitute, nothing to check
