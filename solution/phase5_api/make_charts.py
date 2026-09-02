"""Phase 5 :: charts backing the deployment findings.

One function per finding -> ``(key, path, caption)``; ``build_all(ctx)`` runs
them and writes the CSV exports the report links. House style only
(`capstone.viz`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from capstone import viz

HERE = Path(__file__).resolve().parent
CHART_DIR = HERE / "output" / "charts"
OUTPUT_DIR = HERE / "output"

SRC = ("Source: Phase 5 in-process benchmark of the FastAPI service over the held-out "
       "test-window payload distribution; latency is compute-only.")
SRC_DECISIONS = ("Source: Phase 5 benchmark batch scored through POST /predict/claim-outcome "
                 "over the held-out test-window payload distribution.")


def chart_latency(ctx) -> tuple[str, Path, str]:
    lat = ctx.latency.set_index("endpoint")
    lat.to_csv(OUTPUT_DIR / "latency_summary.csv")
    endpoints = list(lat.index)
    metrics = ["p50", "p95", "p99"]
    fig, ax = viz.new_figure(8.5, 4.6)
    x = np.arange(len(endpoints))
    w = 0.26
    for i, m in enumerate(metrics):
        vals = lat[m].to_numpy()
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m, color=viz.SEQUENTIAL_BLUE[2 + 2 * i])
        viz.label_bars(ax, bars, vals, fmt="{:.1f}")
    ax.set_xticks(x, labels=[e.replace("/predict/", "") for e in endpoints])
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="upper left", ncol=3)
    cap = "Per-request compute latency (p50 / p95 / p99) by prediction endpoint."
    p95b = lat.loc["/predict/claim-outcome", "p95"]
    p95a = lat.loc["/predict/visit-risk", "p95"]
    path = viz.finalize(
        fig, ax,
        title=f"Claim-outcome p95 is {p95b:.0f} ms, visit-risk p95 is {p95a:.0f} ms",
        subtitle="Per-request latency percentiles, in-process client (no network). Model B is the "
                 "gradient-boosted pipeline plus probability calibration; Model A is a constant classifier.",
        source=SRC, out_path=CHART_DIR / "latency.png")
    return "latency", path, cap


def chart_decisions(ctx) -> tuple[str, Path, str]:
    mix = ctx.claim_mix.reindex(["Paid", "Pending", "Rejected"]).fillna(0).astype(int)
    split = ctx.decision_split.reindex(["submit", "review"]).fillna(0).astype(int)
    pd.DataFrame({"predicted_class": mix, "decision_action": split.reindex(mix.index)}).to_csv(
        OUTPUT_DIR / "benchmark_decisions.csv")
    total = int(split.sum())
    thr = ctx.config["models"]["B"]["operating_threshold"]

    fig, ax = viz.new_figure(8.5, 4.4)
    labels = ["Submit as-is", f"Flag for review\n(P(Rejected) ≥ {thr:.2f})"]
    vals = [int(split["submit"]), int(split["review"])]
    colors = [viz.STATUS["neutral"], viz.STATUS["critical"]]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    viz.label_bars(ax, bars, [f"{v}  ({v / total:.0%})" for v in vals], fmt="{}")
    ax.set_ylabel("Claims in benchmark batch")
    ax.set_ylim(0, max(vals) * 1.18)
    cap = (f"Model B review/submit split on a {total}-claim benchmark batch "
           f"at the operating threshold P(Rejected) >= {thr:.2f}.")
    path = viz.finalize(
        fig, ax,
        title=f"Model B flags {vals[1] / total:.0%} of claims for pre-submission review",
        subtitle=f"Benchmark batch of {total} claims drawn from the test-window distribution, "
                 f"scored at the Phase 4 operating threshold.",
        source=SRC_DECISIONS, out_path=CHART_DIR / "decisions.png")
    return "decisions", path, cap


def build_all(ctx) -> list[tuple[str, Path, str]]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    viz.apply_house_style()
    out = []
    for fn in (chart_latency, chart_decisions):
        out.append(fn(ctx))
    # extra CSV exports referenced by the report
    ctx.latency.to_csv(OUTPUT_DIR / "latency_summary.csv", index=False)
    ctx.endpoints.to_csv(OUTPUT_DIR / "endpoints.csv", index=False)
    pd.DataFrame([
        {"model": k, **{kk: vv for kk, vv in g.items()}} for k, g in ctx.golden.items()
    ]).to_csv(OUTPUT_DIR / "golden_regression.csv", index=False)
    return out
