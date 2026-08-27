"""Example Evaluator: metrics computed from ctx, no hidden state. Predictions + GT -> metrics.

F1, precision and recall are only defined over the whole prediction set (they do not decompose
per sample), so each run writes predictions only and the aggregates are computed here.

v1 and v2 fill the **same eval table columns** but compute them differently:
  - v1: a rough early version that substitutes accuracy for F1
  - v2: F1 derived from precision and recall
Results are therefore not comparable across evaluator versions.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from cairn.core.records import EvalResult, Metric
from cairn.interfaces.evaluator import EvalContext


def _collect(ctx: EvalContext) -> tuple[dict[str, int], dict[str, dict], int]:
    gt = {r["sample_id"]: int(r["gt"]) for r in ctx.dataset.rows()}
    preds: dict[str, dict] = {}
    for sid, data in ctx.predictions.iter():
        preds[sid] = json.loads(data)
    return gt, preds, len(gt)


def _confusion(gt: dict[str, int], preds: dict[str, dict]) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for sid, g in gt.items():
        if sid not in preds:
            continue  # samples without a prediction are left out of the denominator (see coverage)
        p = int(preds[sid]["pred"])
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _svg_bars(precision: float, recall: float, f1: float) -> bytes:
    bars = [("precision", precision, "#3D5A80"), ("recall", recall, "#0E6B63"),
            ("F1", f1, "#0A4F49")]
    rects = []
    for i, (_, v, color) in enumerate(bars):
        y = 12 + i * 34
        w = max(2, v * 300)
        rects.append(f'<rect x="70" y="{y}" width="{w:.1f}" height="20" rx="2" fill="{color}"/>'
                     f'<text x="{70 + w + 6:.1f}" y="{y + 15}" font-family="monospace" font-size="12" '
                     f'fill="#4A525C">{v:.3f}</text>')
        rects.append(f'<text x="8" y="{y + 15}" font-family="monospace" font-size="11" fill="#7E8892">'
                     f'{["P", "R", "F1"][i]}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 120" width="100%">'
            f'{"".join(rects)}</svg>').encode()


def _report(name: str, conf: dict[str, int], acc: float, prec: float, rec: float, f1: float,
            coverage: float, worst: list[str]) -> str:
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    lines = [
        f"## {name}",
        "",
        (f"Precision **{prec:.3f}** / recall **{rec:.3f}** / F1 **{f1:.3f}** / accuracy {acc:.3f}"
         f" (coverage {coverage:.0%})."),
        "",
        "### Confusion matrix",
        "",
        "| | predicted anomaly | predicted normal |",
        "|---|---:|---:|",
        f"| **actual anomaly** | {tp} (TP) | {fn} (FN) |",
        f"| **actual normal** | {fp} (FP) | {tn} (TN) |",
        "",
        "### Metrics",
        "",
        "![metrics](assets/metrics.svg)",
        "",
    ]
    if worst:
        lines += ["### Missed samples", ""] + [f"- `{w}`" for w in worst]
    return "\n".join(lines)


def _base(ctx: EvalContext, name: str, use_real_f1: bool) -> EvalResult:
    gt, preds, n = _collect(ctx)
    conf = _confusion(gt, preds)
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    scored = tp + fp + fn + tn
    acc = (tp + tn) / scored if scored else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    real_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f1 = real_f1 if use_real_f1 else round(acc, 3)  # v1 substitutes accuracy for F1
    coverage = scored / n if n else 0.0

    per_sample = []
    worst = []
    for sid, g in gt.items():
        if sid not in preds:
            continue
        p = int(preds[sid]["pred"])
        hit = int(p == g)
        per_sample.append(Metric(name="correct", value=float(hit), sample_id=sid))
        if not hit:
            worst.append(sid)

    metrics = [Metric(name="accuracy", value=acc), Metric(name="precision", value=prec),
               Metric(name="recall", value=rec), Metric(name="f1", value=f1),
               Metric(name="coverage", value=coverage), *per_sample]
    row = {"f1": round(f1, 3), "precision": round(prec, 3), "recall": round(rec, 3),
           "accuracy": round(acc, 3)}
    return EvalResult(row=row, metrics=metrics,
                      report_md=_report(name, conf, acc, prec, rec, f1, coverage, worst[:8]),
                      assets={"metrics.svg": _svg_bars(prec, rec, f1)},
                      # free-form JSON with no column definition; merged with config as metadata
                      metadata={"confusion": conf, "n_scored": scored, "n_missing": n - scored})


class AnomalyEvalV1:
    """Rough early version. Substitutes accuracy for F1."""

    class Config(BaseModel):
        model_config = {"extra": "ignore"}

    def score(self, ctx: EvalContext) -> EvalResult:
        return _base(ctx, "Eval v1 (rough: F1 approximated by accuracy)", use_real_f1=False)


class AnomalyEvalV2:
    """Proper version. F1 derived from precision and recall."""

    class Config(BaseModel):
        model_config = {"extra": "ignore"}

    def score(self, ctx: EvalContext) -> EvalResult:
        return _base(ctx, "Eval v2 (proper F1)", use_real_f1=True)
