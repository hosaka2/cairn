"""Every template must compile and render.

A syntax error in a template is invisible until the page is requested, so it is checked
here directly rather than only through the routes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "cairn" / "web" / "templates"
NAMES = sorted(p.name for p in TEMPLATES.glob("*.html"))


def _context(name: str) -> dict:
    """Enough context to render any page once, with nothing to show."""
    common = {"root": "file:///tmp/x", "who": "tester", "tab": "evals", "title": "t",
              "description": "", "name": "d", "table": "t", "eval_id": "e_1"}
    # `preview` is deliberately absent: the ingest form branches on `is not defined`.
    return common | {
        "datasets": [], "evals": [], "tables": [], "runs": [], "views": [], "table_rows": [],
        "columns": [], "ingests": [], "pages": [], "problems": [],
        "cmp_ids": [], "cmp_choices": [], "comparing": False, "selected": None,
        "subject_title": "", "primary_label": "", "report_html": "", "html": "",
        "total": 0, "offset": 0, "page_size": 25, "has_prev": False, "has_next": False,
        "versions": [], "cfg_fields": [], "can_run": False, "datasets_": [],
        "modes": ["jsonl"], "mode": "jsonl", "fields": None, "has_ingestor": False,
        "jsonl_example": "{}", "kind": "eval", "action": "/x", "label": "table.yaml",
        "value": "", "sel_dataset": "", "default_dataset": None, "page": "manual",
        "dataset": "", "evaluator_version": "v1",
    }


@pytest.fixture(scope="module")
def env() -> Environment:
    e = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    e.globals["t"] = lambda text, **kw: text.format(**kw) if kw else text
    e.globals["lang"] = lambda: "en"
    from cairn.web.app import _num

    e.filters["num"] = _num
    return e


def test_templates_exist():
    assert NAMES, "no templates found"


@pytest.mark.parametrize("name", NAMES)
def test_template_compiles(env, name):
    env.get_template(name)


@pytest.mark.parametrize("name", NAMES)
def test_template_renders(env, name):
    """Compiling is not enough: a mangled expression only fails once it is evaluated."""
    env.get_template(name).render(**_context(name))


def test_yaml_form_names_the_section_it_came_from(env):
    """The breadcrumb follows `kind`; a collapsed conditional would label both the same."""
    tpl = env.get_template("form_yaml.html")
    common = {"title": "t", "action": "/x", "label": "l", "value": "", "datasets": [], "root": "r"}
    dataset_page = tpl.render(kind="dataset", **common)
    eval_page = tpl.render(kind="eval", **common)
    assert '<a href="/datasets">Datasets</a>' in dataset_page
    assert '<a href="/evals">Evals</a>' in eval_page


def test_metric_formatting_survives_any_value(env):
    """A column may hold anything, so the table must not assume every value is a number."""
    from cairn.web.app import _num

    assert _num(0.5) == "0.500"
    assert _num(None) == "—"
    assert _num({"k": 1}) == "{'k': 1}"      # a json column
    assert _num("text") == "text"
    assert _num(float("nan")) == "nan"
    assert _num(True) == "True"              # bools are not metrics
