"""The web UI.

Every route is a thin wrapper over `cairn.core`, so the UI can only do what the CLI can.
Evaluation runs are drawn as stones stacked on top of each other, widest for the best
value of the primary metric.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from pathlib import Path
from typing import Any

import markdown as _md
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from cairn import i18n, registry, scripts
from cairn.core import concurrent as conc
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.config import load_config
from cairn.core.storage import Storage
from cairn.scaffold import VERSION_RE

_SAMPLE_SCHEMA = """name: my-dataset
kind: tabular
description: description
key: id
columns:
  - {name: id, type: str, required: true}
  - {name: value, type: float}
  - {name: gt, type: int, required: true}
"""

_SAMPLE_TABLE = """name: my-eval
description: description
columns:
  # primary: true marks the primary metric (= stone width). Put it on exactly one column.
  - {name: score, type: float, display: "score", primary: true, direction: higher, scale: [0, 1]}
  - {name: coverage, type: float, display: "coverage"}
default_sort: created_at desc
"""

_HERE = Path(__file__).parent
_templates = Jinja2Templates(directory=str(_HERE / "templates"))
# Templates are written in English and translated at render time.
_templates.env.globals["t"] = i18n.t
_templates.env.globals["lang"] = i18n.lang


def _num(value: Any, places: int = 3) -> str:
    """Format a metric. A column may hold anything, so non-numbers are shown as they are."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—" if value is None else str(value)
    if isinstance(value, int):
        return str(value)          # a count is a count: 40 rows, not 40.000
    return f"{value:.{places}f}"


_templates.env.filters["num"] = _num

# Form body limit. The default of 1MB rejects a few thousand pasted JSONL lines; larger
# imports belong on the CLI (`cairn dataset ingest --jsonl file`).
_MAX_FORM_BYTES = 16 * 1024 * 1024


# --- helpers ----------------------------------------------------------------

def _fmt_when(v: Any) -> str:
    # Timestamps arrive as a string from JSON and as a datetime from DuckDB.
    if not v:
        return ""
    if isinstance(v, _dt.datetime):
        return v.strftime("%m-%d %H:%M")
    try:
        return _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).strftime("%m-%d %H:%M")
    except ValueError:
        return str(v)


def _widths(values: list[Any], lo: int = 16, hi: int = 84,
            scale: list[float] | None = None, direction: str = "higher") -> list[int]:
    """Scale values to stone widths in pixels.

    With `scale` the mapping is absolute, so a small difference stays small. Without it
    the widths are relative to the values on screen.
    """
    values = [v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0.0
              for v in values]
    if not values:
        return []
    if scale and len(scale) == 2 and math.isfinite(scale[0]) and math.isfinite(scale[1]) and scale[1] != scale[0]:
        vmin, vmax = float(scale[0]), float(scale[1])
    else:
        vmin, vmax = min(values), max(values)
        if vmax - vmin < 1e-12:
            return [(lo + hi) // 2] * len(values)
    out = []
    for v in values:
        t = (v - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        if direction == "lower":
            t = 1.0 - t
        out.append(round(lo + t * (hi - lo)))
    return out


_STONE_PATINA = "#0E6B63"  # latest run
_STONE_GREY = "#9AA3AC"    # same eval version
_STONE_LIGHT = "#DDE1E6"   # older eval version, not comparable


def _primary(row: dict[str, Any], primary_col: str | None) -> Any:
    """The primary metric of a run, or nothing when the table declares no column."""
    return (row.get(primary_col) if primary_col else None) or 0.0


def _cairn_svg(runs_desc: list[dict[str, Any]], primary_col: str | None, latest_ev: str | None,
               limit: int = 8, scale: list[float] | None = None, direction: str = "higher") -> str:
    """Draw recent runs as a stack of stones, newest on top.

    A dashed line marks where the eval version changes, because runs either side of it
    are not comparable.
    """
    shown = runs_desc[:limit]
    if not shown:
        return ""
    widths = _widths([_primary(r, primary_col) for r in shown], lo=22, hi=76, scale=scale, direction=direction)
    sep_i = next((i for i, r in enumerate(shown) if r.get("evaluator_version") != latest_ev), None)
    parts: list[str] = []
    for i, r in enumerate(shown):
        y = i * 5.2
        w = widths[i]
        x = (112 - w) / 2
        fill = _STONE_PATINA if i == 0 else (_STONE_GREY if r.get("evaluator_version") == latest_ev else _STONE_LIGHT)
        if sep_i is not None and i == sep_i:
            ly = y - 0.8
            parts.append(f'<line x1="14" y1="{ly:.2f}" x2="98" y2="{ly:.2f}" '
                         f'stroke="#CBD0D7" stroke-width=".8" stroke-dasharray="2 2"/>')
        parts.append(f'<rect x="{x:.1f}" y="{y:.2f}" width="{w:.1f}" height="3.6" rx="1" fill="{fill}"/>')
    h = len(shown) * 5.2
    return (f'<svg class="cairn" viewBox="0 0 112 {h:.1f}" width="112" height="{h:.1f}" '
            f'aria-hidden="true">{"".join(parts)}</svg>')


def _report_html(md: str, table: str, eval_id: str) -> str:
    """Render a report, pointing its relative asset links at the served URLs."""
    if not md:
        return ""
    md = md.replace("](assets/", f"](/evals/{table}/runs/{eval_id}/assets/")
    return _md.markdown(md, extensions=["tables", "fenced_code"])


def _jsonl_example(schema) -> str:
    """Two example JSONL lines built from the schema, one plausible value per type."""
    import json as _json

    def val(col, i: int) -> Any:
        if col.name == schema.key:
            return ["a", "b"][i]
        return {
            "str": ["text", "other"][i],
            "int": [1, 0][i],
            "float": [1.0, 0.4][i],
            "bool": [True, False][i],
            "datetime": "2026-01-01T09:00:00",
            "list[float]": [[1.0, 2.0], [0.5, 1.5]][i],
            "list[int]": [[0, 1], [2, 3]][i],
            "list[str]": [["x", "y"], ["z"]][i],
            "json": {},
        }.get(col.type, ["a", "b"][i])

    lines = [_json.dumps({c.name: val(c, i) for c in schema.columns}, ensure_ascii=False) for i in range(2)]
    return "\n".join(lines)


def _delta(base: Any, val: Any, direction: str) -> dict[str, Any] | None:
    """Difference from the baseline, and whether it is an improvement."""
    if not isinstance(base, (int, float)) or not isinstance(val, (int, float)):
        return None
    if isinstance(base, bool) or isinstance(val, bool):
        return None
    d = val - base
    if abs(d) < 1e-12:
        return {"value": 0.0, "good": None}
    return {"value": d, "good": (d > 0) == (direction == "higher")}


def _metadata_json(meta: dict[str, Any]) -> str:
    """The run config merged with whatever the evaluator recorded, shown as JSON."""
    import json as _json

    md = {**(meta.get("config") or {}), **(meta.get("metadata") or {})}
    return _json.dumps(md, ensure_ascii=False, indent=2) if md else ""


def _with_dataset(yaml_text: str, dataset: str) -> str:
    """Put the selected dataset into the table definition, if one was selected."""
    if not dataset:
        return yaml_text
    import yaml as _yaml

    doc = _yaml.safe_load(yaml_text) or {}
    if not isinstance(doc, dict):
        # This is malformed input, not a programming error.
        raise ValueError("table.yaml must be a mapping")  # noqa: TRY004
    doc["dataset"] = dataset
    return _yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)


def _ctx(st: Storage) -> dict[str, Any]:
    cfg = load_config()
    return {"root": cfg.root, "who": os.environ.get("CAIRN_USER", "cairn")}


def _render(name: str, ctx: dict[str, Any], status: int = 200) -> HTMLResponse:
    tpl = _templates.get_template(name)
    return HTMLResponse(tpl.render(**ctx), status_code=status)


# --- app --------------------------------------------------------------------

# Reports and assets come from user scripts and may contain arbitrary markup, so the
# pages that show them are served with scripting disabled.
# Scripting stays off, which is what stops injected markup from doing anything. Inline
# styles are allowed because the templates size elements (stone widths) with them.
_CSP = ("default-src 'none'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; form-action 'self'; base-uri 'none'")


def create_app() -> FastAPI:
    app = FastAPI(title="cairn")
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", _CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    def st() -> Storage:
        return Storage(load_config().root)

    @app.get("/")
    def index() -> RedirectResponse:
        return RedirectResponse("/evals")

    # --- datasets ---
    @app.get("/datasets", response_class=HTMLResponse)
    def datasets() -> HTMLResponse:
        s = st()

        def row(name: str) -> dict[str, Any]:
            schema = ds.load_schema(s, name)
            info = ds.info(s, name)          # already counts the ingests and dates them
            return {"name": name, "kind": schema.kind or "—", "rows": info["rows"],
                    "ingests": info["ingests"], "updated": _fmt_when(info["updated"])}

        rows = conc.each(ds.list_datasets(s), row)
        rows.sort(key=lambda r: r["updated"], reverse=True)
        return _render("datasets.html", {**_ctx(s), "tab": "datasets", "datasets": rows})

    # Register /datasets/new before /datasets/{name}: routes match in order.
    @app.get("/datasets/new", response_class=HTMLResponse)
    def dataset_new_form() -> HTMLResponse:
        return _render("form_yaml.html", {**_ctx(st()), "tab": "datasets", "kind": "dataset",
                                          "title": "Create dataset", "action": "/datasets/new",
                                          "label": "schema.yaml", "value": _SAMPLE_SCHEMA})

    @app.get("/datasets/{name}", response_class=HTMLResponse)
    def dataset_detail(name: str, offset: int = 0, page_size: int = 25) -> HTMLResponse:
        s = st()
        _require_dataset(s, name)
        page_size = max(1, min(page_size, 200))
        schema = ds.load_schema(s, name)
        total = ds.count_rows(s, name)                                       # DuckDB: count(*)
        offset = max(0, offset)
        preview = ds.rows_page(s, name, limit=page_size, offset=offset)       # DuckDB: LIMIT/OFFSET
        ings = ds.list_ingests(s, name)
        widths = _widths([i.get("n_ops", 0) for i in ings])
        ing_view = [
            {"n": i.get("n_ops", 0), "note": i.get("input", {}).get("jsonl", ""),
             "when": _fmt_when(i.get("created_at", "")), "width": w}
            for i, w in zip(reversed(ings), reversed(widths))
        ]
        columns = [
            {"name": c.name, "display": c.display or c.name,
             "num": c.type in ("int", "float"), "bool": c.type == "bool",
             "mono": c.name == schema.key}
            for c in schema.columns
        ]
        return _render("dataset.html", {
            **_ctx(s), "tab": "datasets", "name": name, "description": schema.description,
            "total": total, "preview": preview, "columns": columns, "ingests": ing_view,
            "offset": offset, "page_size": page_size,
            "has_prev": offset > 0, "has_next": offset + page_size < total,
        })

    # --- evals ---
    @app.get("/evals", response_class=HTMLResponse)
    def evals_list() -> HTMLResponse:
        s = st()
        # Two listings that have nothing to do with each other; done in turn, the page
        # waits for the network twice before it starts.
        dataset_names, table_names = conc.both(lambda: ds.list_datasets(s),
                                              lambda: evals.list_eval_tables(s))

        def table_row(t: str) -> dict[str, Any]:
            schema = evals.load_eval_table(s, t)
            pc = schema.primary_column()
            primary_col = pc.name if pc else None
            # Read the table's results once: the count, the newest, and the one before it
            # scored the same way all come from the same rows, and asking storage per
            # question is four round trips where one will do.
            runs = evals.list_runs(s, t)
            total = len(runs)
            recent = runs[:8]
            latest = runs[0] if runs else None
            latest_ev = latest.get("evaluator_version") if latest else ""
            # Compare against the previous run scored the same way.
            same = [r for r in runs if r.get("evaluator_version") == latest_ev] if latest else []
            same2 = same[:2]
            primary = same2[0].get(primary_col) if same2 and primary_col else None
            prev = same2[1].get(primary_col) if len(same2) > 1 and primary_col else None
            # Whether a delta is an improvement depends on the column, not on its sign.
            d = _delta(prev, primary, pc.direction) if pc else None
            delta = d["value"] if d else None
            delta_good = d["good"] if d else None
            n_latest = len(same)
            method_note = ((f"{latest_ev} only" if n_latest == total
                            else f"{latest_ev}: {n_latest}") if latest else "")
            dataset = latest.get("dataset") if latest else None
            rows = (ds.info(s, dataset)["rows"]
                    if isinstance(dataset, str) and dataset in dataset_names else None)
            return {
                "name": t, "desc": schema.description, "dataset": dataset, "rows": rows,
                "runs": total, "method_note": method_note,
                "svg": _cairn_svg(recent, primary_col, latest_ev,
                                  scale=pc.scale if pc else None,
                                  direction=pc.direction if pc else "higher") if recent else "",
                "latest_id": latest.get("eval_id") if latest else None,
                "latest_when": _fmt_when(latest.get("created_at", "")) if latest else "",
                "primary": primary, "delta": delta, "delta_good": delta_good,
            }

        tables = conc.each(table_names, table_row)
        return _render("evals.html", {**_ctx(s), "tab": "evals", "tables": tables})

    # --- help ---
    @app.get("/help", response_class=HTMLResponse)
    @app.get("/help/{page}", response_class=HTMLResponse)
    def help_page(page: str = "manual") -> HTMLResponse:
        from cairn import docs as _docs

        try:
            md = _docs.read(page)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such page") from e
        return _render("help.html", {
            **_ctx(st()), "tab": "help", "pages": _docs.pages(), "page": page,
            "title": _docs.title(page),
            "html": _md.markdown(md, extensions=["tables", "fenced_code", "toc"]),
            # Read from the files, never by importing them: a page must not pull in what
            # a script depends on to tell the reader that the script is there.
            "problems": scripts.problems(),
        })

    @app.get("/evals/new", response_class=HTMLResponse)
    def eval_new_form() -> HTMLResponse:
        s = st()
        return _render("form_yaml.html", {**_ctx(s), "tab": "evals", "kind": "eval",
                                          "title": "Create eval table", "action": "/evals/new",
                                          "label": "table.yaml", "value": _SAMPLE_TABLE,
                                          "datasets": ds.list_datasets(s)})

    def _run_view(s: Storage, table: str, eval_id: str, *, detail: bool = True) -> dict[str, Any] | None:
        """One run for the page. `detail` adds the note, metadata and report."""
        try:
            d = evals.get_run(s, table, eval_id)
        except FileNotFoundError:
            return None
        if not d["row"]:
            return None
        meta = d["meta"]
        view = {"eval_id": eval_id, "row": d["row"],
                "title": d["row"].get("title") or meta.get("title") or eval_id,
                "when": _fmt_when(d["row"].get("created_at", ""))}
        if detail:
            view |= {"memo": meta.get("comment") or "",
                     "metadata_json": _metadata_json(meta),
                     "report_html": _report_html(d["report_md"], table, eval_id)}
        return view

    @app.get("/evals/{table}", response_class=HTMLResponse)
    def eval_detail(table: str, run: str | None = None, cmp: list[str] = Query([]),
                    started: str = "") -> HTMLResponse:
        # `?run=` selects a run; each `?cmp=` adds another one to compare it against.
        # `?started=` points at a run just created, whose id the caller still needs.
        s = st()
        _require_table(s, table)
        schema = evals.load_eval_table(s, table)
        pc = schema.primary_column()
        primary_col = pc.name if pc else None
        primary_label = (pc.display or primary_col) if pc else ""

        # The stones show the most recent runs scored with the current eval version.
        view = evals.runs_view(s, table)
        runs = view.rows()
        latest_ev = runs[0].get("evaluator_version", "") if runs else ""
        same = [r for r in runs if r.get("evaluator_version") == latest_ev][:8] if latest_ev else []

        # Selected run: the one asked for, otherwise the latest.
        selected = next((r for r in same if r["eval_id"] == run), None)
        if selected is None and run:
            try:
                selected = evals.get_run(s, table, run)["row"]
            except FileNotFoundError:
                selected = None
        if selected is None:
            selected = same[0] if same else None
        selected_id = selected["eval_id"] if selected else None

        # Runs to compare against.
        # Comparable means "scored the same way", not "among the eight on screen".
        def _same_version(eval_id: str) -> bool:
            try:
                return evals.get_meta(s, table, eval_id).get("evaluator_version") == latest_ev
            except (FileNotFoundError, ValueError):
                return False

        cmp_ids = [c for c in dict.fromkeys(cmp) if c and c != selected_id and _same_version(c)]

        widths = _widths([_primary(r, primary_col) for r in same],
                         scale=pc.scale if pc else None, direction=pc.direction if pc else "higher")
        run_view = [
            {"eval_id": r["eval_id"], "primary": r.get(primary_col) if primary_col else None,
             "title": r.get("title") or r.get("comment", ""),
             "when": _fmt_when(r.get("created_at", "")), "width": w,
             "selected": r["eval_id"] == selected_id, "compared": r["eval_id"] in cmp_ids}
            for r, w in zip(same, widths)
        ]

        # Only the selected run needs its note, metadata and report; the runs it is
        # compared against contribute a row to the table and nothing else.
        views = [v for v in ([_run_view(s, table, selected_id)] if selected_id else []) if v]
        views += [v for v in (_run_view(s, table, e, detail=False) for e in cmp_ids) if v]
        # Only runs scored the same way can be added to a comparison. Same data is the other
        # half of comparability, and that one is not enforced: a run on different data may be
        # worth looking at, so it is offered and then marked as such wherever it appears.
        selected_snapshot = (selected or {}).get("snapshot_id")

        def _other_data(row: dict[str, Any]) -> bool:
            return bool(selected_snapshot and row.get("snapshot_id") != selected_snapshot)

        cmp_choices = [{"eval_id": r["eval_id"], "title": r.get("title") or r["eval_id"],
                        "other_data": _other_data(r)}
                       for r in same if r["eval_id"] != selected_id and r["eval_id"] not in cmp_ids]

        # Started but not evaluated: the inference is running somewhere else, or has not begun.
        evers = scripts.versions(table)
        pending = [{"eval_id": p["eval_id"], "title": p.get("title") or p["eval_id"],
                    "when": _fmt_when(p.get("created_at", "")),
                    "version": p.get("evaluator_version", ""),
                    # Whoever watches the inference decides when it is done, so the
                    # button is always there; only a missing evaluator hides it.
                    "can_score": p.get("evaluator_version") in evers}
                   for p in evals.pending_runs(s, table, view)]

        columns = [{"name": c.name, "display": c.display or c.name} for c in schema.columns]
        # The page is about the selected run, so every other row answers the same question:
        # against this one, how did the selected run do? The delta is therefore the selected
        # value minus the row's, coloured by the column's direction rather than by its sign.
        # The selected run leads the table; the rest follow oldest first. A table with no
        # scored run has nothing to show here, and the template says so.
        table_rows = views[:1] + sorted(views[1:], reverse=True,
                                        key=lambda v: (v["row"].get("created_at") or "", v["eval_id"]))
        if len(table_rows) > 1:
            subject = table_rows[0]
            for v in table_rows:
                v["subject"] = v is subject
                if v is subject:
                    continue
                v["delta"] = {c.name: _delta(v["row"].get(c.name), subject["row"].get(c.name), c.direction)
                              for c in schema.columns}
                # Comparing across snapshots is not comparing: say so on the row itself.
                v["other_data"] = _other_data(v["row"])
                v["snapshot_id"] = v["row"].get("snapshot_id", "")

        return _render("eval.html", {
            **_ctx(s), "tab": "evals", "table": table, "description": schema.description,
            "runs": run_view, "selected": selected, "columns": columns,
            "primary_label": primary_label, "views": views, "table_rows": table_rows,
            "subject_title": views[0]["title"] if views else "",
            "cmp_ids": cmp_ids, "cmp_choices": cmp_choices, "comparing": len(views) > 1,
            "pending": pending, "started": started,
        })

    # The report on its own, for reading a long one at full width.
    @app.get("/evals/{table}/runs/{eval_id}/report", response_class=HTMLResponse)
    def run_report(table: str, eval_id: str) -> HTMLResponse:
        s = st()
        _require_table(s, table)
        try:
            run = evals.get_run(s, table, eval_id)
        except (FileNotFoundError, ValueError) as e:  # ValueError: a path-shaped id
            raise HTTPException(404, "run not found") from e
        row = run["row"] or {}
        return _render("report.html", {
            **_ctx(s), "tab": "evals", "table": table, "eval_id": eval_id,
            "title": row.get("title") or run["meta"].get("title") or eval_id,
            "evaluator_version": row.get("evaluator_version", ""),
            "dataset": row.get("dataset", ""),
            "snapshot_id": row.get("snapshot_id", ""),
            "report_html": _report_html(run["report_md"], table, eval_id),
        })

    # Images referenced by a report are served from here.
    @app.get("/evals/{table}/runs/{eval_id}/assets/{asset}")
    def run_asset(table: str, eval_id: str, asset: str) -> Response:
        s = st()
        _require_table(s, table)
        try:
            data = evals.read_asset(s, table, eval_id, asset)
        except (FileNotFoundError, ValueError) as e:  # ValueError: a path-shaped id
            raise HTTPException(404, "asset not found") from e
        media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "gif": "image/gif", "webp": "image/webp",
                 "svg": "image/svg+xml"}.get(asset.rsplit(".", 1)[-1].lower(), "application/octet-stream")
        return Response(content=data, media_type=media,
                        headers={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})

    @app.exception_handler(HTTPException)
    def _http_error(request: Request, exc: HTTPException) -> HTMLResponse:
        """Refusals are read by a person in a browser, so they get a page, not JSON."""
        referer = request.headers.get("referer", "")
        back = referer if referer.startswith(str(request.base_url)) else "/evals"
        return _render("error.html", {**_ctx(st()), "tab": "", "status": exc.status_code,
                                      "detail": exc.detail, "back": back}, status=exc.status_code)

    # --- write flows ---

    def _text(form: FormData, key: str, default: str = "") -> str:
        """One form field as text. A form may also carry uploads; this app posts text."""
        v = form.get(key)
        return v if isinstance(v, str) else default

    def _texts(form: FormData) -> dict[str, str]:
        return {k: v for k, v in form.items() if isinstance(v, str)}

    def _who() -> str:
        return os.environ.get("CAIRN_USER", "cairn")

    # --- create a dataset ---
    @app.post("/datasets/new")
    async def dataset_new(request: Request):
        form = await request.form(max_part_size=_MAX_FORM_BYTES)
        try:
            name = ds.create_dataset(st(), _text(form, "yaml"), created_by=_who())
        except Exception as e:  # noqa: BLE001
            return _render("form_yaml.html", {**_ctx(st()), "tab": "datasets", "kind": "dataset",
                                              "title": "Create dataset", "action": "/datasets/new",
                                              "label": "schema.yaml", "value": _text(form, "yaml"),
                                              "error": str(e)})
        return RedirectResponse(f"/datasets/{name}", status_code=303)

    # --- create an eval table ---
    @app.post("/evals/new")
    async def eval_new(request: Request):
        s = st()
        form = await request.form(max_part_size=_MAX_FORM_BYTES)
        text = _text(form, "yaml")
        try:
            # The dataset comes from the selector; the YAML wins when nothing is selected.
            text = _with_dataset(text, _text(form, "dataset").strip())
            name = evals.create_eval_table(s, text)
        except Exception as e:  # noqa: BLE001
            return _render("form_yaml.html", {**_ctx(s), "tab": "evals", "kind": "eval",
                                              "title": "Create eval table", "action": "/evals/new",
                                              "label": "table.yaml", "value": _text(form, "yaml"),
                                              "datasets": ds.list_datasets(s),
                                              "sel_dataset": _text(form, "dataset"), "error": str(e)})
        return RedirectResponse(f"/evals/{name}", status_code=303)

    # --- add data ---
    def _run_ingest(s: Storage, name: str, form: FormData) -> tuple[list[dict], list[dict]]:
        ing = registry.load_ingestor(name)
        mode = _text(form, "mode") or ("script" if ing is not None else "jsonl")
        if mode == "script" and ing is not None:
            from cairn.interfaces.ingestor import IngestContext

            inp = ing.Input(**registry.coerce(ing.Input, _texts(form)))
            ctx = IngestContext(dataset=name, tmpdir="/tmp", created_by=_who())
            rows = list(ing().ingest(inp, ctx))
        else:  # JSONL, which stays available even when a script exists
            return _rows_from_jsonl(s, name, _text(form, "jsonl"))
        schema = ds.load_schema(s, name)
        errors = [{"i": i, "errs": e} for i, r in enumerate(rows) if (e := schema.validate_row(r))]
        return rows, errors

    def _rows_from_jsonl(s: Storage, name: str, text: str) -> tuple[list[dict], list[dict]]:
        """Parse pasted JSONL. A bad line is reported, not raised: the preview exists to
        catch exactly this before anything is stored."""
        import json as _json

        schema = ds.load_schema(s, name)
        rows: list[dict] = []
        errors: list[dict] = []
        for i, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                row = _json.loads(line)
            except ValueError as e:
                errors.append({"i": None, "errs": [f"line {i + 1} is not valid JSON: {e}"]})
                continue
            if not isinstance(row, dict):
                errors.append({"i": None, "errs": [f"line {i + 1} is not a JSON object"]})
                continue
            if errs := schema.validate_row(row):
                errors.append({"i": len(rows), "errs": errs})
            rows.append(row)
        return rows, errors

    def _good_rows(rows: list[dict], errors: list[dict]) -> list[dict]:
        """The rows that will actually be appended."""
        bad = {e["i"] for e in errors if e["i"] is not None}
        return [r for i, r in enumerate(rows) if i not in bad]

    def _require_dataset(s: Storage, name: str) -> None:
        if name not in ds.list_datasets(s):
            raise HTTPException(404, "dataset not found")

    def _require_table(s: Storage, table: str) -> None:
        if table not in evals.list_eval_tables(s):
            raise HTTPException(404, "eval table not found")

    def _ingest_ctx(s: Storage, name: str, *, mode: str | None = None, **extra) -> dict[str, Any]:
        _require_dataset(s, name)
        # The form is built from `input.py`, which cairn reads and never imports. The
        # ingest script itself is imported only when the rows are actually produced.
        fields = scripts.fields("datasets", name)
        if fields is None:      # nothing declared on disk: only a registered module can say
            ing = registry.registered_ingestor(name)
            fields = registry.form_fields(ing.Input) if ing else None
        has_ing = fields is not None
        # Two ways in: the script form when one is declared, and pasted JSONL.
        modes = (["script"] if has_ing else []) + ["jsonl"]
        if mode not in modes:
            mode = modes[0]
        return {**_ctx(s), "tab": "datasets", "name": name,
                "fields": fields,
                "has_ingestor": has_ing,
                "jsonl_example": _jsonl_example(ds.load_schema(s, name)),
                "modes": modes, "mode": mode, **extra}

    @app.get("/datasets/{name}/ingest", response_class=HTMLResponse)
    def ingest_form(name: str, mode: str | None = None) -> HTMLResponse:
        return _render("form_ingest.html", _ingest_ctx(st(), name, mode=mode))

    @app.post("/datasets/{name}/ingest/preview", response_class=HTMLResponse)
    async def ingest_preview(name: str, request: Request) -> HTMLResponse:
        form = await request.form(max_part_size=_MAX_FORM_BYTES)
        rows, errors = _run_ingest(st(), name, form)
        return _render("form_ingest.html", _ingest_ctx(
            st(), name, mode=_text(form, "mode") or None, preview=rows[:10], n=len(rows),
            n_good=len(_good_rows(rows, errors)), errors=errors, raw=_texts(form)))

    @app.post("/datasets/{name}/ingest")
    async def ingest_confirm(name: str, request: Request):
        form = await request.form(max_part_size=_MAX_FORM_BYTES)
        rows, errors = _run_ingest(st(), name, form)
        ds.append(st(), name, _good_rows(rows, errors), created_by=_who(), input={"via": "web"})
        return RedirectResponse(f"/datasets/{name}", status_code=303)

    # --- run an evaluation ---
    @app.get("/evals/{table}/run", response_class=HTMLResponse)
    def run_form(table: str) -> HTMLResponse:
        s = st()
        _require_table(s, table)
        default_ds = evals.load_eval_table(s, table).dataset
        return _render("form_run.html", {
            **_ctx(s), "tab": "evals", "table": table,
            "datasets": [d for d in ds.list_datasets(s) if ds.info(s, d)["rows"]],
            "default_dataset": default_ds,
            "versions": scripts.versions(table),
            # Declared in `config.py`, read without importing it. The values are stored
            # with the run and read by whatever performs the inference.
            "cfg_fields": scripts.fields("evals", table) or _registered_cfg(table),
        })

    def _registered_cfg(table: str) -> list[dict[str, Any]]:
        inf = registry.registered_inference(table)
        return registry.form_fields(inf["config"]) if inf else []

    def _run_request(s: Storage, table: str, form: FormData) -> tuple[str, str, str, str]:
        """Validate a run request before anything is created. Returns what to create it with."""
        dataset, ev = _text(form, "dataset"), _text(form, "evaluator").strip()
        if dataset not in ds.list_datasets(s):
            raise HTTPException(400, "dataset not found")
        if not VERSION_RE.match(ev):
            raise HTTPException(400, "eval version must be v1, v2, … (the vN.py that scores it)")
        title = _text(form, "title").strip()
        if not title:
            raise HTTPException(400, "Title is required")
        return dataset, ev, title, _text(form, "comment")

    def _run_config(form: FormData, inf: dict[str, Any] | None) -> dict[str, Any]:
        """The config stored with the run: the script's fields, or free JSON without one."""
        if inf is not None:
            return registry.coerce(inf["config"], _texts(form))
        import json as _json

        try:
            cfg = _json.loads(_text(form, "config") or "{}")
        except ValueError as e:
            raise HTTPException(400, f"config is not valid JSON: {e}") from e
        if not isinstance(cfg, dict):
            raise HTTPException(400, "config must be a JSON object")
        return cfg

    @app.post("/evals/{table}/run")
    async def run_execute(table: str, request: Request):
        """Start a run. The inference is never performed here: a pipeline performs it."""
        s = st()
        _require_table(s, table)
        form = await request.form(max_part_size=_MAX_FORM_BYTES)
        dataset, ev, title, comment = _run_request(s, table, form)
        cfg = _run_config(form, registry.load_inference(table))

        eval_id = evals.create_run(s, table, dataset=dataset, evaluator_version=ev,
                                   title=title, comment=comment, created_by=_who(), config=cfg)
        return RedirectResponse(f"/evals/{table}?started={eval_id}", status_code=303)

    @app.post("/evals/{table}/runs/{eval_id}/score")
    def run_score(table: str, eval_id: str):
        """Score a run whose predictions were written elsewhere."""
        s = st()
        _require_table(s, table)
        try:
            version = evals.get_meta(s, table, eval_id).get("evaluator_version", "")
        except FileNotFoundError as e:
            raise HTTPException(404, "run not found") from e
        evaluator = registry.load_evaluator(table, version)
        if evaluator is None:
            raise HTTPException(400, f"No evaluator registered for {version or 'this run'}")
        evals.score_run(s, table, eval_id, evaluator())
        return RedirectResponse(f"/evals/{table}?run={eval_id}", status_code=303)

    return app


app = create_app()
