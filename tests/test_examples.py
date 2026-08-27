"""The YAML we hand people: the manual, the help beside the forms, the form defaults.

These are the first thing a reader copies, so each one has to load as a real schema.
Prose can go stale quietly; an example that no longer parses cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cairn.core.schema import parse_dataset_schema, parse_eval_table
from cairn.web.app import _SAMPLE_SCHEMA, _SAMPLE_TABLE

SRC = Path(__file__).resolve().parents[1] / "src" / "cairn"


def _parse(text: str):
    """Datasets are the ones with a row key; everything else is an eval table."""
    return parse_dataset_schema(text) if re.search(r"^key:", text, re.MULTILINE) else parse_eval_table(text)


def _blocks(path: Path, pattern: str) -> list[str]:
    return re.findall(pattern, path.read_text(encoding="utf-8"), re.DOTALL)


DOC_EXAMPLES = [(p.name, b) for p in sorted((SRC / "docs").glob("*.md"))
                for b in _blocks(p, r"```yaml\n(.*?)```")]
HELP_EXAMPLES = [("form_yaml.html", b)
                 for b in _blocks(SRC / "web" / "templates" / "form_yaml.html",
                                  r'<pre class="hlp-pre">(.*?)</pre>')]


@pytest.mark.parametrize("where, text", DOC_EXAMPLES + HELP_EXAMPLES,
                         ids=[w for w, _ in DOC_EXAMPLES + HELP_EXAMPLES])
def test_every_example_loads_as_a_schema(where, text):
    assert _parse(text).name


def test_the_examples_are_actually_found():
    """A change in fencing must not turn this file into a no-op."""
    assert len(DOC_EXAMPLES) >= 4 and len(HELP_EXAMPLES) == 2


def test_the_form_defaults_load_as_schemas():
    """What the create forms open with is itself an example."""
    assert parse_dataset_schema(_SAMPLE_SCHEMA).key
    assert parse_eval_table(_SAMPLE_TABLE).primary_column()
