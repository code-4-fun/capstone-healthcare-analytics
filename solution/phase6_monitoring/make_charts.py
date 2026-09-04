"""Phase 6 :: charts backing every monitoring finding.

One function per finding -> ``(key, path, caption)``; ``build_all(ctx)`` runs
them and writes the CSV exports the report appendix links. House style only
(`capstone.viz`) - the drift dashboard is Grafana; these are the deck charts.

``ctx`` is the :class:`report.Phase6Context` the notebook assembles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from capstone import monitoring as mon
from capstone import viz

HERE = Path(__file__).resolve().parent
CHART_DIR = HERE / "output" / "charts"
OUTPUT_DIR = HERE / "output"

SRC = ("Source: Phase 6 drift monitor - seeded replay of Phase 3 test-window visits through the "
       "serving path, compared to the Phase 3 test-window reference.")


def _psi_bar(feat: pd.DataFrame, key: str, title: str, subtitle: str, out: str,
             *, top: int = 14):
    feat = feat.sort_values("psi", ascending=False)
    shown = feat[feat["psi"] >= 0.01]
    if len(shown) < top:
        shown = feat.head(top)
    omitted = len(feat) - len(shown)
    shown = shown.sort_values("psi", ascending=True)

    colors = [viz.STATUS["critical"] if b == "significant"
              else viz.STATUS["warning"] if b == "moderate"
              else viz.STATUS["neutral"] for b in shown["band"]]
    fig, ax = viz.new_figure(8.5, max(3.8, 0.34 * len(shown) + 1.6))
    bars = ax.barh(shown["feature"], shown["psi"], color=colors)
    viz.label_bars(ax, bars, shown["psi"], fmt="{:.2f}", horizontal=True)
    xmax = max(0.32, shown["psi"].max() * 1.18)
    ytop = len(shown) - 0.35
    for x, lbl in ((mon.PSI_STABLE, "moderate 0.10"), (mon.PSI_SIGNIFICANT, "significant 0.25")):
        if x < xmax:
            ax.axvline(x, color=viz.INK_MUTED, lw=0.9, ls="--")
            ax.text(x, ytop, lbl, fontsize=7.5, color=viz.INK_MUTED, va="bottom", ha="center")
    ax.set_xlabel("Population Stability Index vs the reference window")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(shown) - 0.1)
    note = f"  ({omitted} further features below 0.01 PSI, not shown)" if omitted else ""
    path = viz.finalize(fig, ax, title=title, subtitle=subtitle + note, source=SRC,
                        out_path=CHART_DIR / out)
    return key, path, subtitle


def chart_gate_offences(ctx):
    df = ctx.gate_offences.copy()
    df = df.sort_values("n_offending", ascending=True)
    df.to_csv(OUTPUT_DIR / "gate_offences.csv", index=False)
    fig, ax = viz.new_figure(8.5, 4.2)
    colors = [viz.SEVERITY_COLORS.get(s, viz.CATEGORICAL[0]) for s in df["severity"]]
    bars = ax.barh(df["rule"], df["n_offending"], color=colors)
    viz.label_bars(ax, bars, df["n_offending"], fmt="{:.0f}", horizontal=True)
    ax.set_xlabel(f"Requests rejected (malformed batch of {ctx.gate_bad_n})")
    cap = ("The validation gate rejects every malformed request in a "
           f"{ctx.gate_bad_n}-request probe batch; the clean batch of {ctx.gate_ok_n} passes with "
           "zero offences.")
    path = viz.finalize(
        fig, ax,
        title="The request gate catches every malformed field before it reaches a model",
        subtitle="Phase 2 data-quality rules (enum domains + pre-submission numeric ranges) run "
                 "as a batch gate over served request payloads.",
        source="Source: Phase 6 validation gate (capstone.monitoring.validation_gate) over a "
               "hand-built malformed probe batch.",
        out_path=CHART_DIR / "gate_offences.png")
    return "gate_offences", path, cap


def chart_feature_psi_baseline(ctx):
    ctx.feat_baseline.to_csv(OUTPUT_DIR / "feature_psi_baseline.csv", index=False)
    return _psi_bar(
        ctx.feat_baseline, "feature_psi_baseline",
        "Un-perturbed replay traffic shows no feature drift",
        "Model B monitored features, baseline window vs the Phase 3 test-window reference - "
        "every feature stays inside the stable band.",
        "feature_psi_baseline.png")


def chart_feature_psi_drift(ctx):
    ctx.feat_drift.to_csv(OUTPUT_DIR / "feature_psi_drift.csv", index=False)
    n_sig = int((ctx.feat_drift["band"] == "significant").sum())
    return _psi_bar(
        ctx.feat_drift, "feature_psi_drift",
        f"Injected drift trips {n_sig} monitored features past the significant line",
        "Model B monitored features, drifted window vs the Phase 3 test-window reference - "
        "billed amount, department mix, insurer mix and patient age all move.",
        "feature_psi_drift.png")


def chart_prediction_mix(ctx):
    mix = ctx.prediction_mix.copy()
    mix.to_csv(OUTPUT_DIR / "prediction_mix.csv", index=False)
    classes = mix["predicted_class"].tolist()
    x = np.arange(len(classes))
    w = 0.26
    fig, ax = viz.new_figure(8.5, 4.4)
    for i, (col, lbl, c) in enumerate((
        ("reference_share", "reference", viz.SEQUENTIAL_BLUE[2]),
        ("baseline_share", "baseline window", viz.SEQUENTIAL_BLUE[4]),
        ("drift_share", "drifted window", viz.STATUS["critical"]),
    )):
        vals = mix[col].to_numpy()
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=c)
        viz.label_bars(ax, bars, vals, fmt="{:.0%}")
    ax.set_xticks(x, labels=classes)
    ax.set_ylabel("Share of predictions")
    ax.set_ylim(0, max(mix[["reference_share", "baseline_share", "drift_share"]].max()) * 1.25)
    ax.legend(loc="upper right", ncol=3)

    moved = mix.assign(delta=mix["drift_share"] - mix["reference_share"])
    gain = moved.loc[moved["delta"].idxmax(), "predicted_class"]
    drop = moved.loc[moved["delta"].idxmin(), "predicted_class"]
    cap = (f"Model B predicted-class mix: baseline PSI {ctx.pred_psi_baseline:.03f}, "
           f"drifted PSI {ctx.pred_psi_drift:.03f} (alert threshold "
           f"{mon.ALERT_RULES['prediction_psi']}).")
    path = viz.finalize(
        fig, ax,
        title=f"The drifted window pushes prediction-mix PSI to {ctx.pred_psi_drift:.2f}",
        subtitle="Predicted-class shares vs the Phase 3 test-window reference. The baseline window "
                 f"tracks the reference; the drifted window shifts toward {gain} and away from "
                 f"{drop} (inflated billed amounts move out of the mid-value rejection-peak band).",
        source=SRC, out_path=CHART_DIR / "prediction_mix.png")
    return "prediction_mix", path, cap


def chart_performance_drift(ctx):
    d = ctx.performance_drift
    pd.DataFrame([d]).to_csv(OUTPUT_DIR / "performance_drift.csv", index=False)
    labels = ["Phase 4\nbaseline", "baseline\nwindow", "drifted\nwindow"]
    vals = [d["recall_baseline"], d["recall_baseline_window"], d["recall_drift_window"]]
    colors = [viz.STATUS["neutral"], viz.STATUS["good"], viz.STATUS["critical"]]
    fig, ax = viz.new_figure(8.0, 4.4)
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    viz.label_bars(ax, bars, vals, fmt="{:.0%}")
    floor = d["recall_baseline"] - mon.ALERT_RULES["recall_costly_drop_pts"]
    ax.axhline(floor, color=viz.INK_MUTED, lw=0.9, ls="--")
    ax.text(2.5, floor, f" alert floor {floor:.0%}", fontsize=7.5, color=viz.INK_MUTED,
            va="bottom", ha="right")
    ax.set_ylabel("Recall on Rejected claims")
    ax.set_ylim(0, max(vals) * 1.25)
    cap = (f"Model B recall on to-be-Rejected claims at P(Rejected) >= "
           f"{d['operating_threshold']}: {d['recall_drift_window']:.0%} on the drifted window vs a "
           f"{d['recall_baseline']:.0%} Phase 4 baseline - a "
           f"{(d['recall_baseline'] - d['recall_drift_window']) * 100:.0f}-point drop that breaches "
           "the alert floor.")
    path = viz.finalize(
        fig, ax,
        title="Under drift, Model B misses more of the rejections it exists to catch",
        subtitle="Recall on the costly class, actual claim outcomes joined back to the served "
                 "predictions on the caller's visit_id.",
        source=SRC, out_path=CHART_DIR / "performance_drift.png")
    return "performance_drift", path, cap


def chart_alert_summary(ctx):
    rows = ctx.drift_rows[ctx.drift_rows["alert"]].copy()
    by_kind = rows.groupby("metric_kind").size().sort_values(ascending=True)
    by_kind.to_csv(OUTPUT_DIR / "alert_summary.csv")
    fig, ax = viz.new_figure(8.0, 3.6)
    bars = ax.barh(by_kind.index, by_kind.to_numpy(), color=viz.STATUS["critical"])
    viz.label_bars(ax, bars, by_kind.to_numpy(), fmt="{:.0f}", horizontal=True)
    ax.set_xlabel("Alerts raised in the drift run")
    cap = (f"The drifted-window run raised {int(by_kind.sum())} alerts across "
           f"{by_kind.size} metric kinds; the baseline run raised "
           f"{int(ctx.baseline_alert_count)}.")
    path = viz.finalize(
        fig, ax,
        title=f"One drift run, {int(by_kind.sum())} alerts written to drift_report",
        subtitle="Alert rows by metric kind for the drifted window. Alert rules: feature PSI > "
                 f"{mon.ALERT_RULES['feature_psi']}, prediction PSI > "
                 f"{mon.ALERT_RULES['prediction_psi']}, costly-recall drop > "
                 f"{mon.ALERT_RULES['recall_costly_drop_pts']:.0%}, gate fail-rate > "
                 f"{mon.ALERT_RULES['gate_fail_rate']:.0%}.",
        source=SRC, out_path=CHART_DIR / "alert_summary.png")
    return "alert_summary", path, cap


_CHARTS = (
    chart_gate_offences,
    chart_feature_psi_baseline,
    chart_feature_psi_drift,
    chart_prediction_mix,
    chart_performance_drift,
    chart_alert_summary,
)


def build_all(ctx) -> list[tuple[str, Path, str]]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    viz.apply_house_style()
    out = []
    for fn in _CHARTS:
        try:
            out.append(fn(ctx))
        except Exception as exc:  # noqa: BLE001 - a missing optional chart must not abort the run
            print(f"  chart {fn.__name__} skipped: {exc}")
    (ctx.drift_rows.drop(columns=["detail"], errors="ignore")
     .to_csv(OUTPUT_DIR / "drift_report_rows.csv", index=False))
    return out
