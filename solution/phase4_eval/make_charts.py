"""Phase 4 :: charts backing every evaluation / explainability / fairness finding.

One function per finding -> ``(key, path, caption)``; ``build_all(ctx)`` runs
them all and writes the CSV exports the report appendix links. House style only
(`capstone.viz`) - no seaborn, no per-chart restyling.

``ctx`` is a :class:`capstone.evaluation.EvalContext` assembled by the notebook.
Charts that depend on optional SHAP output return ``None`` when it is absent;
``build_all`` filters those out and the report notes the fallback.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capstone import evaluation as E
from capstone import modeling as M
from capstone import viz

HERE = Path(__file__).resolve().parent
CHART_DIR = HERE / "output" / "charts"
OUTPUT_DIR = HERE / "output"

SRC = ("Source: Phase 4 evaluation of the persisted Phase 3 models on the held-out "
       "test window (4,260 visits, time-split on visit_date; threshold chosen on the "
       "validation month).")

_METRIC_COLORS = {
    "precision": viz.CATEGORICAL[0],
    "recall": viz.CATEGORICAL[1],
    "f1": viz.CATEGORICAL[2],
}


# --------------------------------------------------------------------------
# 1. per-class precision / recall / F1  (Model B, pipeline predictions)
# --------------------------------------------------------------------------
def chart_per_class_b(ctx: E.EvalContext) -> tuple[str, Path, str]:
    pc = ctx.per_class["B"].set_index("class")
    pc.to_csv(OUTPUT_DIR / "per_class_metrics_b.csv")
    classes = list(pc.index)
    fig, ax = viz.new_figure(8.5, 4.6)
    x = np.arange(len(classes))
    w = 0.26
    for i, metric in enumerate(("precision", "recall", "f1")):
        bars = ax.bar(x + (i - 1) * w, pc[metric].values, w,
                      label=metric.capitalize(), color=_METRIC_COLORS[metric])
        viz.label_bars(ax, bars, pc[metric].values, fmt="{:.2f}")
    ax.set_xticks(x, labels=[f"{c}\n(n={int(pc.loc[c, 'support'])})" for c in classes])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=3)
    return "per_class_b", viz.finalize(
        fig, ax,
        title="Model B trades Paid precision for a high catch rate on Rejected claims",
        subtitle="Per-class precision, recall and F1 on the test window (uncalibrated pipeline "
                 "labels). Recall on Rejected is 66%; the cost is lower precision - most flags are "
                 "Paid claims sent for a second look.",
        source=SRC, out_path=CHART_DIR / "per_class_b.png",
    ), "Model B per-class precision / recall / F1 on the held-out test window."


# --------------------------------------------------------------------------
# 2. confusion at the operating threshold  (Model B)
# --------------------------------------------------------------------------
def chart_confusion_operating_b(ctx: E.EvalContext) -> tuple[str, Path, str]:
    labels = M.CLASS_ORDER["B"]
    cm = E.confusion_at_threshold(ctx.preds["B"], labels, costly="Rejected",
                                  threshold=ctx.threshold)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        OUTPUT_DIR / "confusion_operating_b.csv")
    fig, ax = viz.new_figure(5.8, 4.9)
    viz.heatmap(ax, cm, labels, labels, cbar_label="Row share")
    ax.set_xlabel("Decision at operating threshold")
    ax.set_ylabel("Actual claim_status")
    bs = ctx.business
    return "confusion_operating_b", viz.finalize(
        fig, ax,
        title=f"At the operating threshold Model B catches {bs['costly_caught']} of "
              f"{bs['costly_total']} rejections",
        subtitle=f"Rejected is assigned when calibrated P(Rejected) >= {ctx.threshold:.2f} "
                 f"(chosen on validation to maximise net recovered leakage); otherwise the "
                 f"argmax of Paid/Pending. {bs['flagged_n']:,} claims flagged for review.",
        source=SRC, out_path=CHART_DIR / "confusion_operating_b.png",
    ), "Model B confusion at the operating threshold (not the Phase 3 argmax)."


# --------------------------------------------------------------------------
# 3 + 4. ROC and PR, one-vs-rest  (Model B)
# --------------------------------------------------------------------------
def _curve_chart(ctx: E.EvalContext, curve: str) -> tuple[str, Path, str]:
    d = ctx.roc_pr["B"]
    d = d[d["curve"] == curve]
    fig, ax = viz.new_figure(7.2, 4.8)
    for i, cls in enumerate(M.CLASS_ORDER["B"]):
        dc = d[d["cls"] == cls]
        if dc.empty:
            continue
        metric = dc["auc"].iloc[0] if curve == "roc" else dc["ap"].iloc[0]
        tag = "AUC" if curve == "roc" else "AP"
        ax.plot(dc["x"], dc["y"], "-", color=viz.CATEGORICAL[i], linewidth=1.8,
                label=f"{cls} ({tag} {metric:.2f})")
        if curve == "pr":
            ax.axhline(dc["base_rate"].iloc[0], color=viz.CATEGORICAL[i],
                       linestyle=":", linewidth=0.9)
    if curve == "roc":
        ax.plot([0, 1], [0, 1], "--", color=viz.INK_MUTED, linewidth=1, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        title = "Model B ranks Rejected claims well above chance; Pending is near-random"
        sub = ("One-vs-rest ROC on the test window using calibrated probabilities. Rejected "
               "AUC ~0.7 is the usable signal; the dotted line is chance.")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        title = "Precision stays low for Rejected - the base rate is only 14%"
        sub = ("One-vs-rest precision-recall on the test window. Dotted lines mark each class's "
               "base rate (the no-skill precision). Rejected precision is well above its 14% "
               "base rate across the usable recall range.")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right" if curve == "roc" else "upper right")
    return f"{curve}_b", viz.finalize(
        fig, ax, title=title, subtitle=sub, source=SRC,
        out_path=CHART_DIR / f"{curve}_b.png",
    ), f"Model B one-vs-rest {curve.upper()} curves (calibrated probabilities)."


def chart_roc_b(ctx):
    return _curve_chart(ctx, "roc")


def chart_pr_b(ctx):
    return _curve_chart(ctx, "pr")


# --------------------------------------------------------------------------
# 5. calibration reliability  (Model B classes + Model A P(High))
# --------------------------------------------------------------------------
def chart_calibration_ab(ctx: E.EvalContext) -> tuple[str, Path, str]:
    rb, ra = ctx.reliability["B"], ctx.reliability["A"]
    pd.concat([rb.assign(model="B"), ra.assign(model="A")]).to_csv(
        OUTPUT_DIR / "calibration_ab.csv", index=False)
    panels = [("B", "Paid"), ("B", "Pending"), ("B", "Rejected"), ("A", "High")]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.7), sharex=True, sharey=True)
    for ax, (mdl, cls) in zip(axes, panels):
        src = rb if mdl == "B" else ra
        ax.plot([0, 1], [0, 1], "--", color=viz.INK_MUTED, linewidth=1)
        for variant, col in (("uncalibrated", viz.CATEGORICAL[0]),
                             ("calibrated", viz.CATEGORICAL[1])):
            dd = src[(src["cls"] == cls) & (src["variant"] == variant)]
            if not dd.empty:
                ax.plot(dd["mean_predicted"], dd["observed_freq"], "o-", color=col,
                        markersize=4, linewidth=1.6, label=variant)
        ax.text(0.5, 1.02, f"Model {mdl} - {cls}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9.5, color=viz.INK_SECONDARY)
        ax.set_xlabel("Mean predicted prob.")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="upper left", fontsize=8)
    return "calibration_ab", viz.finalize(
        fig, axes[0],
        title="Sigmoid calibration aligns Model B's probabilities; Model A's are a flat prior",
        subtitle="Reliability curves on the test window before and after validation-month "
                 "calibration. Model A predicts one constant probability per class, so its curve "
                 "is a single point - honest but uninformative.",
        source=SRC, out_path=CHART_DIR / "calibration_ab.png",
    ), "Reliability curves, uncalibrated vs calibrated: Model B classes and Model A P(High)."


# --------------------------------------------------------------------------
# 6. threshold sweep  (Model B)
# --------------------------------------------------------------------------
def chart_threshold_sweep_b(ctx: E.EvalContext) -> tuple[str, Path, str]:
    s = ctx.sweep_test
    fig, ax = viz.new_figure(8.5, 4.6)
    ax.plot(s["threshold"], s["recall"], "-", color=viz.CATEGORICAL[1], linewidth=1.8,
            label="Recall on Rejected")
    ax.plot(s["threshold"], s["precision"], "-", color=viz.CATEGORICAL[0], linewidth=1.8,
            label="Precision on Rejected")
    ax.plot(s["threshold"], s["flagged_rate"], "-", color=viz.CATEGORICAL[2], linewidth=1.8,
            label="Share of claims flagged")
    ax.axvline(ctx.threshold, color=viz.INK_MUTED, linestyle="--", linewidth=1,
               label=f"operating threshold ({ctx.threshold:.2f})")
    ax.set_xlabel("Calibrated P(Rejected) threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    return "threshold_sweep_b", viz.finalize(
        fig, ax,
        title="The operating threshold trades recall against review workload",
        subtitle="Test-window recall, precision and flagged share as the calibrated P(Rejected) "
                 "cut-off moves. The chosen threshold is where net recovered leakage peaks on the "
                 "validation month (next chart).",
        source=SRC, out_path=CHART_DIR / "threshold_sweep_b.png",
    ), "Model B recall / precision / alert volume vs the decision threshold."


# --------------------------------------------------------------------------
# 7. net recovered leakage  (Model B) - the money chart
# --------------------------------------------------------------------------
def chart_net_recovery_b(ctx: E.EvalContext) -> tuple[str, Path, str]:
    s = ctx.sweep_test
    s.to_csv(OUTPUT_DIR / "threshold_sweep_b.csv", index=False)
    fig, ax = viz.new_figure(8.5, 4.6)
    ax.plot(s["threshold"], s["leakage_recovered"], "-", color=viz.CATEGORICAL[2],
            linewidth=1.8, label="Recoverable leakage (40% haircut)")
    ax.plot(s["threshold"], s["review_cost_total"], "-", color=viz.CATEGORICAL[3],
            linewidth=1.8, label="Manual-review cost")
    ax.plot(s["threshold"], s["net_recovered"], "-", color=viz.CATEGORICAL[0],
            linewidth=2.2, label="Net recovered")
    ax.axvline(ctx.threshold, color=viz.INK_MUTED, linestyle="--", linewidth=1,
               label=f"operating threshold ({ctx.threshold:.2f})")
    ax.axhline(0, color=viz.INK_MUTED, linewidth=0.8)
    ax.set_xlabel("Calibrated P(Rejected) threshold")
    ax.set_ylabel("Test window total")
    ax.yaxis.set_major_formatter(lambda v, _pos: viz.money(v))
    ax.legend(loc="upper right")
    bs = ctx.business
    return "net_recovery_b", viz.finalize(
        fig, ax,
        title=f"Net recoverable leakage is ~{viz.money(bs['net_recovered'])} at the operating "
              f"threshold (2-month test window)",
        subtitle=f"Recoverable denial leakage on correctly-flagged rejections (assuming 40% is "
                 f"fixable pre-submission and Rs.{int(E.REVIEW_COST):,}/claim to review) minus review "
                 f"cost. Operating point: {bs['recall']:.0%} recall, "
                 f"~{bs['alerts_per_month']:.0f} alerts/month.",
        source=SRC, out_path=CHART_DIR / "net_recovery_b.png",
    ), "Model B net recoverable denial leakage vs the decision threshold."


# --------------------------------------------------------------------------
# 8. permutation importance  (Model B)
# --------------------------------------------------------------------------
def chart_permutation_importance_b(ctx: E.EvalContext) -> tuple[str, Path, str]:
    pi = ctx.perm_importance["B"].head(12).iloc[::-1]
    ctx.perm_importance["B"].to_csv(OUTPUT_DIR / "permutation_importance_b.csv", index=False)
    fig, ax = viz.new_figure(8.5, 5.0)
    y = np.arange(len(pi))
    bars = ax.barh(y, pi["importance_mean"].values, 0.6, color=viz.CATEGORICAL[0],
                   xerr=pi["importance_std"].values, error_kw=dict(ecolor=viz.INK_MUTED, lw=0.8))
    viz.label_bars(ax, bars, pi["importance_mean"].values, fmt="{:.3f}", horizontal=True)
    ax.set_yticks(y, labels=pi["feature"].values, fontsize=8)
    ax.set_xlabel("Drop in balanced accuracy when the feature is shuffled")
    return "permutation_importance_b", viz.finalize(
        fig, ax,
        title="Model B is a billed-amount model - nothing else moves balanced accuracy",
        subtitle="Permutation importance on the test window (10 repeats, balanced-accuracy "
                 "scoring). billed_amount and its band dominate; every other feature is within "
                 "noise of zero - the Phase 2 signal screen, confirmed on the trained model.",
        source=SRC, out_path=CHART_DIR / "permutation_importance_b.png",
    ), "Model B permutation importance (balanced-accuracy drop, 10 repeats)."


# --------------------------------------------------------------------------
# 9. SHAP global summary  (Model B) - optional
# --------------------------------------------------------------------------
def chart_shap_summary_b(ctx: E.EvalContext):
    if ctx.shap is None:
        return None
    ma = ctx.shap.mean_abs().head(12).iloc[::-1]
    ma.to_csv(OUTPUT_DIR / "shap_mean_abs_b.csv", index=False)
    fig, ax = viz.new_figure(8.5, 5.0)
    y = np.arange(len(ma))
    bars = ax.barh(y, ma["mean_abs_shap"].values, 0.6, color=viz.CATEGORICAL[4])
    viz.label_bars(ax, bars, ma["mean_abs_shap"].values, fmt="{:.2f}", horizontal=True)
    ax.set_yticks(y, labels=ma["feature"].values, fontsize=8)
    ax.set_xlabel("Mean |SHAP value| for the model's Rejected score (log-odds)")
    return "shap_summary_b", viz.finalize(
        fig, ax,
        title="SHAP agrees: the billed amount and its 15k-30k band drive the Rejected score",
        subtitle="Mean absolute SHAP contribution to the gradient-boosted model's Rejected score, "
                 "TreeExplainer on a 600-row test sample. Consistent with permutation importance "
                 "and the Phase 2 non-monotonic band finding.",
        source=SRC, out_path=CHART_DIR / "shap_summary_b.png",
    ), "Model B SHAP global summary for the Rejected score (mean |value|)."


# --------------------------------------------------------------------------
# 10. SHAP local explanations  (Model B) - optional
# --------------------------------------------------------------------------
def chart_shap_local_b(ctx: E.EvalContext):
    if ctx.shap is None or not ctx.shap_local:
        return None
    sr = ctx.shap
    items = list(ctx.shap_local.items())[:3]
    fig, axes = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.6))
    if len(items) == 1:
        axes = [axes]
    for ax, (label, row) in zip(axes, items):
        contrib = pd.Series(sr.values[row], index=sr.feature_names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
        colors = [viz.STATUS["critical"] if v > 0 else viz.CATEGORICAL[0] for v in top.values]
        bars = ax.barh(np.arange(len(top)), top.values, 0.6, color=colors)
        viz.label_bars(ax, bars, top.values, fmt="{:+.2f}", horizontal=True)
        ax.axvline(0, color=viz.INK_MUTED, linewidth=0.8)
        ax.set_yticks(np.arange(len(top)), labels=top.index, fontsize=7.5)
        p = ctx.preds["B"].iloc[int(sr.X_raw.index[row])]
        ax.text(0.5, 1.02, f"{label}\nP(Rejected)={p['prob_rejected']:.2f}, actual {p['actual']}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=viz.INK_SECONDARY)
    return "shap_local_b", viz.finalize(
        fig, axes[0],
        title="Local SHAP: each flag is explainable as a billed-amount story",
        subtitle="Per-claim SHAP contributions to the model's Rejected score (log-odds) for three "
                 "test claims. Red pushes toward Rejected, blue toward Paid. The billed-amount "
                 "term dominates every explanation.",
        source=SRC, out_path=CHART_DIR / "shap_local_b.png",
    ), "Model B local SHAP explanations for three sample claims."


# --------------------------------------------------------------------------
# 11. leakage verification by ablation  (Models A + B)
# --------------------------------------------------------------------------
def chart_leakage_ablation(ctx: E.EvalContext) -> tuple[str, Path, str]:
    d = ctx.ablation["B"].reset_index(drop=True)
    d.to_csv(OUTPUT_DIR / "leakage_ablation_b.csv", index=False)
    fig, ax = viz.new_figure(9.0, 4.8)
    y = np.arange(len(d))
    w = 0.38
    for i, leak in enumerate(d["is_leak"]):
        if leak:
            ax.axhspan(i - 0.5, i + 0.5, color=viz.STATUS["critical"], alpha=0.12, zorder=0)
    b1 = ax.barh(y + w / 2, d["balanced_accuracy"].values, w, color=viz.CATEGORICAL[0],
                 label="Balanced accuracy", zorder=3)
    b2 = ax.barh(y - w / 2, d["recall_rejected"].values, w, color=viz.CATEGORICAL[2],
                 label="Recall on Rejected", zorder=3)
    viz.label_bars(ax, b1, d["balanced_accuracy"].values, fmt="{:.2f}", horizontal=True)
    viz.label_bars(ax, b2, d["recall_rejected"].values, fmt="{:.2f}", horizontal=True)
    ax.axvline(1 / 3, color=viz.INK_MUTED, linestyle=":", linewidth=0.9, label="chance (0.33)")
    ax.set_yticks(y, labels=d["variant"].values, fontsize=8)
    for tick, leak in zip(ax.get_yticklabels(), d["is_leak"]):
        if leak:
            tick.set_color(viz.STATUS["critical"])
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    return "leakage_ablation", viz.finalize(
        fig, ax,
        title="Model B's metrics only jump when a post-outcome field is injected",
        subtitle="Retraining Model B's gradient-boosted learner with allowed features dropped "
                 "barely moves the test metrics; adding a forbidden field (red rows) - "
                 "approved_amount, payment_days - spikes them. The shipped model is the 'clean' "
                 "row. Model A ignores all features and cannot leak (permutation importance, "
                 "section 4).",
        source=SRC, out_path=CHART_DIR / "leakage_ablation.png",
    ), "Model B leakage ablation: dropped-feature variants vs deliberately-leaky variants."


# --------------------------------------------------------------------------
# 12. fairness parity  (Model B)
# --------------------------------------------------------------------------
def chart_fairness_b(ctx: E.EvalContext) -> tuple[str, Path, str]:
    frames = ctx.fairness["B"]["frames"]
    ctx.fairness["B"]["summary"].to_csv(OUTPUT_DIR / "fairness_parity_summary_b.csv", index=False)
    for g, fr in frames.items():
        fr.to_csv(OUTPUT_DIR / f"fairness_{g}_b.csv", index=False)
    keys = list(frames)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for ax, g in zip(axes.ravel(), keys):
        fr = frames[g].sort_values("level")
        x = np.arange(len(fr))
        w = 0.38
        b1 = ax.bar(x - w / 2, fr["recall"].values, w, color=viz.CATEGORICAL[1],
                    label="Recall on Rejected")
        b2 = ax.bar(x + w / 2, fr["selection_rate"].values, w, color=viz.CATEGORICAL[0],
                    label="Share flagged")
        viz.label_bars(ax, b1, fr["recall"].values, fmt="{:.2f}")
        viz.label_bars(ax, b2, fr["selection_rate"].values, fmt="{:.2f}")
        ax.set_xticks(x, labels=[t.replace(" ", "\n") for t in fr["level"]], fontsize=8)
        ax.set_ylim(0, 1)
        ax.text(0.5, 1.02, g, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=viz.INK_SECONDARY)
    axes.ravel()[len(keys) - 1].legend(loc="upper right", fontsize=8)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    summ = ctx.fairness["B"]["summary"].set_index("group")
    worst = summ["recall_gap"].idxmax()
    sel_pass = bool(summ["four_fifths_pass"].all())
    return "fairness_b", viz.finalize(
        fig, axes.ravel()[0],
        title="Model B's flag rate holds within a four-fifths band across every group"
              if sel_pass else "Model B shows a selection-rate disparity across groups",
        subtitle=f"Recall on Rejected and share of claims flagged, by protected attribute, at the "
                 f"operating threshold. Widest recall gap: {summ.loc[worst, 'recall_gap']:.2f} "
                 f"across {worst}, tracking base-rate differences between groups; "
                 f"{'no group fails' if sel_pass else 'a group fails'} the four-fifths "
                 f"selection-rate test.",
        source=SRC, out_path=CHART_DIR / "fairness_b.png",
    ), "Model B recall and selection rate by gender / age band / city / insurer."


# --------------------------------------------------------------------------
# 13. Model A degeneracy
# --------------------------------------------------------------------------
def chart_model_a_degeneracy(ctx: E.EvalContext) -> tuple[str, Path, str]:
    p = ctx.preds["A"]
    labels = M.CLASS_ORDER["A"]
    actual = p["actual"].value_counts(normalize=True).reindex(labels).fillna(0) * 100
    pred = p["pred_pipeline"].value_counts(normalize=True).reindex(labels).fillna(0) * 100
    pd.DataFrame({"class": labels, "actual_pct": actual.values.round(1),
                  "predicted_pct": pred.values.round(1)}).to_csv(
        OUTPUT_DIR / "model_a_degeneracy.csv", index=False)
    # back the "no signal" prose claims with exports (CLAUDE.md rule 7)
    ctx.perm_importance["A"].to_csv(OUTPUT_DIR / "permutation_importance_a.csv", index=False)
    (ctx.roc_pr["A"].groupby("cls")["auc"].first().dropna().round(4)
     .rename("roc_auc").reset_index().rename(columns={"cls": "class"})
     .to_csv(OUTPUT_DIR / "model_a_roc_auc.csv", index=False))
    fig, ax = viz.new_figure(8.0, 4.4)
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2, actual.values, w, color=viz.CATEGORICAL[0], label="Actual")
    b2 = ax.bar(x + w / 2, pred.values, w, color=viz.CATEGORICAL[1], label="Model A predicts")
    viz.label_bars(ax, b1, actual.values, fmt="{:.0f}%")
    viz.label_bars(ax, b2, pred.values, fmt="{:.0f}%")
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel("Share of test visits (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right")
    return "model_a_degeneracy", viz.finalize(
        fig, ax,
        title="Model A predicts 'Low' for every visit - it has no signal to do otherwise",
        subtitle="Actual vs predicted risk-band distribution on the test window. Recall on "
                 "High-risk visits is 0%. Model A is retained only as a calibrated monitor of the "
                 "risk-mix distribution, never for prioritising an individual visit.",
        source=SRC, out_path=CHART_DIR / "model_a_degeneracy.png",
    ), "Model A predicted vs actual risk-band distribution (constant classifier)."


# --------------------------------------------------------------------------
# build all
# --------------------------------------------------------------------------
_CHART_FNS = [
    chart_per_class_b,
    chart_confusion_operating_b,
    chart_roc_b,
    chart_pr_b,
    chart_calibration_ab,
    chart_threshold_sweep_b,
    chart_net_recovery_b,
    chart_permutation_importance_b,
    chart_shap_summary_b,
    chart_shap_local_b,
    chart_leakage_ablation,
    chart_fairness_b,
    chart_model_a_degeneracy,
]


def build_all(ctx: E.EvalContext) -> list[tuple[str, Path, str]]:
    viz.apply_house_style()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for fn in _CHART_FNS:
        res = fn(ctx)
        if res is not None:
            out.append(res)
    return out
