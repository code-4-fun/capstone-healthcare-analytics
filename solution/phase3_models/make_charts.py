"""Phase 3 :: charts backing every model-development finding.

One function per finding -> ``(key, path, caption)``; ``build_all(results)`` runs
them all and also writes the CSV exports the report appendix links. House style
only (`capstone.viz`) - no seaborn, no per-chart restyling.

``results`` is ``{"A": ModelResult, "B": ModelResult}`` from
``capstone.modeling.train_model``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capstone import modeling as M
from capstone import viz

HERE = Path(__file__).resolve().parent
CHART_DIR = HERE / "output" / "charts"
OUTPUT_DIR = HERE / "output"
PHASE2_OUT = HERE.parent / "phase2_eda" / "output"

SRC = ("Source: Phase 3 training on the Phase 2 as-of feature frame - "
       "25,000 visits, time-split on visit_date (9 mo train / 1 val / 2 test).")

_CANDIDATE_LABEL = {
    "majority": "Majority\nclass",
    "simple_rule": "Simple\nrule",
    "logreg": "Logistic\nregression",
    "gbm": "Gradient\nboosting",
}


# --------------------------------------------------------------------------
# 1. target class balance, train vs test
# --------------------------------------------------------------------------
def chart_target_balance(results: dict[str, M.ModelResult]) -> tuple[str, Path, str]:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (key, res) in zip(axes, results.items()):
        y_test = res.extras["y_test"]
        test_share = y_test.value_counts(normalize=True).reindex(res.classes).fillna(0) * 100
        train_share = res.extras["train_class_share"].reindex(res.classes).fillna(0) * 100
        x = np.arange(len(res.classes))
        w = 0.38
        b1 = ax.bar(x - w / 2, train_share.values, w, label="Train",
                    color=viz.CATEGORICAL[0])
        b2 = ax.bar(x + w / 2, test_share.values, w, label="Test",
                    color=viz.CATEGORICAL[1])
        viz.label_bars(ax, b1, train_share.values, fmt="{:.0f}%")
        viz.label_bars(ax, b2, test_share.values, fmt="{:.0f}%")
        ax.set_xticks(x, labels=res.classes)
        ax.set_ylim(0, 78)
        ax.set_ylabel("Share of rows (%)")
        ax.text(0.5, 0.98, f"Model {key} - {res.target.replace('target_', '')}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color=viz.INK_SECONDARY)
        if key == "B":
            ax.legend(loc="upper right")
        rows.append(pd.DataFrame({"model": key, "class": res.classes,
                                  "train_pct": train_share.values.round(1),
                                  "test_pct": test_share.values.round(1)}))
    pd.concat(rows).to_csv(OUTPUT_DIR / "target_balance.csv", index=False)
    return "target_balance", viz.finalize(
        fig, axes[0],
        title="Targets are skewed to the cheap class and stable across the time split",
        subtitle="Class shares in the training window and the held-out test window match closely - "
                 "no temporal drift in the labels. Both minority classes (High risk, Rejected claims) "
                 "sit near 15-20%, so recall on them is the metric that matters.",
        source=SRC, out_path=CHART_DIR / "target_balance.png",
    ), "Target class balance, train vs test, for both models."


# --------------------------------------------------------------------------
# 2. candidate comparison (one per model)
# --------------------------------------------------------------------------
def _candidate_chart(res: M.ModelResult, key: str) -> tuple[str, Path, str]:
    mf = res.metrics_frame()
    order = ["majority", "simple_rule", "logreg", "gbm"]
    mf = mf.set_index("candidate").reindex(order).reset_index()
    recall_col = [c for c in mf.columns if c.startswith("recall_")][0]
    costly = res.extras["costly_class"]

    fig, ax = viz.new_figure(9.0, 4.6)
    x = np.arange(len(order))
    w = 0.26
    series = [
        ("Balanced accuracy", mf["balanced_accuracy"].values, viz.CATEGORICAL[0]),
        ("Macro F1", mf["macro_f1"].values, viz.CATEGORICAL[2]),
        (f"Recall on {costly}", mf[recall_col].values, viz.CATEGORICAL[1]),
    ]
    for i, (lab, vals, col) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
        viz.label_bars(ax, bars, vals, fmt="{:.2f}")
    # chance line for balanced accuracy on a 3-class problem; the majority bar
    # sits on it and is value-labelled, and the subtitle calls it out, so the
    # dashed line needs no in-plot annotation to compete with the bars
    ax.axhline(1 / 3, color=viz.INK_MUTED, linestyle="--", linewidth=1, label="chance (0.33)")
    ax.set_xticks(x, labels=[_CANDIDATE_LABEL[o] for o in order])
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(0.75, float(np.nanmax([v for _, vs, _ in series for v in vs])) + 0.12))
    ax.legend(loc="upper left", ncol=4)
    chosen = res.chosen
    if key == "A":
        title = "No trained model separates visit-risk classes - Model A ships as a base-rate monitor"
        sub = ("Logistic regression and gradient boosting both sit at chance on balanced accuracy; "
               "the higher macro-F1 only reflects the majority classifier scoring zero on two "
               "classes. Selected: the calibrated majority baseline (Phase 2 found no feature "
               "above the noise floor).")
    else:
        title = "Gradient boosting beats both baselines on the metrics that matter for triage"
        sub = ("Balanced accuracy 0.44 (majority 0.33), macro-F1 0.37, and 66% of rejected claims "
               "recovered vs 0% for the majority classifier and 62% for the blunt billed-band "
               "rule. Selected for Model B; raw accuracy is lower by design (minority-recall "
               "trade-off, tuned in Phase 4).")
    mf.to_csv(OUTPUT_DIR / f"candidate_metrics_{key.lower()}.csv", index=False)
    return f"candidate_comparison_{key.lower()}", viz.finalize(
        fig, ax, title=title, subtitle=sub, source=SRC,
        out_path=CHART_DIR / f"candidate_comparison_{key.lower()}.png",
    ), f"Model {key}: candidate metrics on the held-out test window (chosen: {chosen})."


def chart_candidate_comparison_a(results):
    return _candidate_chart(results["A"], "A")


def chart_candidate_comparison_b(results):
    return _candidate_chart(results["B"], "B")


# --------------------------------------------------------------------------
# 3. recall on the costly class - chosen model vs baselines
# --------------------------------------------------------------------------
def chart_costly_class_recall(results: dict[str, M.ModelResult]) -> tuple[str, Path, str]:
    fig, ax = viz.new_figure(8.5, 4.4)
    groups, rows = [], []
    labels = ["Majority\nclass", "Simple\nrule", "Chosen\nmodel"]
    x = np.arange(len(results))
    w = 0.26
    for i, (name, col) in enumerate(zip(labels, [viz.CATEGORICAL[0], viz.CATEGORICAL[1], viz.CATEGORICAL[2]])):
        vals = []
        for key, res in results.items():
            k = {"Majority\nclass": "majority", "Simple\nrule": "simple_rule",
                 "Chosen\nmodel": res.chosen}[name]
            costly = res.extras["costly_class"]
            v = res.eval_of(k)[f"recall_{costly.lower()}"]
            vals.append(v)
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name.replace("\n", " "), color=col)
        viz.label_bars(ax, bars, vals, fmt="{:.0%}")
    for key, res in results.items():
        costly = res.extras["costly_class"]
        rows.append({"model": key, "costly_class": costly,
                     "recall_majority": round(res.extras["costly_recall_majority"], 4),
                     "recall_simple_rule": round(res.extras["costly_recall_simple_rule"], 4),
                     "recall_chosen": round(res.extras["costly_recall_chosen"], 4),
                     "chosen": res.chosen})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "costly_class_recall.csv", index=False)
    ax.set_xticks(x, labels=[f"Model {k}\n(recall on {res.extras['costly_class']})"
                             for k, res in results.items()])
    ax.set_ylabel("Recall on the costly class")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return "costly_class_recall", viz.finalize(
        fig, ax,
        title="Model B recovers two-thirds of rejected claims; Model A cannot flag high risk at all",
        subtitle="Recall on the business-critical class. Model B's gradient-boosted model catches "
                 "66% of the claims that go on to be rejected (majority classifier: 0%). Model A has "
                 "no signal to work with, so no trained model beats the 0% of the base-rate monitor.",
        source=SRC, out_path=CHART_DIR / "costly_class_recall.png",
    ), "Recall on the costly class (High / Rejected): baselines vs the chosen model."


# --------------------------------------------------------------------------
# 4. confusion matrices
# --------------------------------------------------------------------------
def _confusion_chart(res: M.ModelResult, key: str, candidate: str, note: str) -> tuple[str, Path, str]:
    ev = res.eval_of(candidate)
    cm = np.array(ev["confusion_matrix"])
    pd.DataFrame(cm, index=res.classes, columns=res.classes).to_csv(
        OUTPUT_DIR / f"confusion_matrix_{key.lower()}.csv")
    fig, ax = viz.new_figure(5.6, 4.8)
    viz.heatmap(ax, cm, res.classes, res.classes, cbar_label="Row share")
    ax.set_xlabel(f"Predicted {res.target.replace('target_', '')}")
    ax.set_ylabel(f"Actual {res.target.replace('target_', '')}")
    acc, bal = ev["accuracy"], ev["balanced_accuracy"]
    return f"confusion_matrix_{key.lower()}", viz.finalize(
        fig, ax,
        title=f"Model {key} test confusion - {note}",
        subtitle=(f"{len(res.extras['y_test']):,} test rows. Cell colour is the row-normalised rate, "
                  f"the number is the count. Accuracy {acc:.0%}, balanced accuracy {bal:.0%}."),
        source=SRC, out_path=CHART_DIR / f"confusion_matrix_{key.lower()}.png",
    ), f"Model {key} confusion matrix ({candidate})."


def chart_confusion_matrix_a(results):
    return _confusion_chart(results["A"], "A", "logreg",
                            "even a trained classifier lands near chance")


def chart_confusion_matrix_b(results):
    return _confusion_chart(results["B"], "B", results["B"].chosen,
                            "rejections are found at the cost of Paid precision")


# --------------------------------------------------------------------------
# 5. calibration reliability, Model B
# --------------------------------------------------------------------------
def chart_calibration_b(results: dict[str, M.ModelResult]) -> tuple[str, Path, str]:
    res = results["B"]
    rc = M.reliability_curves(res)
    rc.to_csv(OUTPUT_DIR / "calibration_b.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), sharex=True, sharey=True)
    for ax, cls in zip(axes, res.classes):
        ax.plot([0, 1], [0, 1], "--", color=viz.INK_MUTED, linewidth=1)
        for variant, col in (("uncalibrated", viz.CATEGORICAL[0]), ("calibrated", viz.CATEGORICAL[1])):
            d = rc[(rc.cls == cls) & (rc.variant == variant)]
            ax.plot(d["mean_predicted"], d["observed_freq"], "o-", color=col,
                    markersize=4, linewidth=1.8, label=variant)
        ax.text(0.5, 1.02, cls, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=viz.INK_SECONDARY)
        ax.set_xlabel("Mean predicted prob.")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="upper left", fontsize=8)
    return "calibration_b", viz.finalize(
        fig, axes[0],
        title="Sigmoid calibration pulls Model B's probabilities onto the diagonal",
        subtitle="Reliability curves on the test window, per class, before and after calibration on "
                 "the validation month. Calibrated probabilities are what Phase 5 thresholds against "
                 "for the pre-submission triage cut.",
        source=SRC, out_path=CHART_DIR / "calibration_b.png",
    ), "Model B reliability curves, uncalibrated vs calibrated, per class."


# --------------------------------------------------------------------------
# 6. Phase 2 feature-signal recap
# --------------------------------------------------------------------------
def chart_feature_signal_recap(results: dict[str, M.ModelResult]) -> tuple[str, Path, str]:
    sig = pd.read_csv(PHASE2_OUT / "feature_signal.csv")
    sig = sig[sig["model_allowed"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))
    for ax, (mkey, mlabel) in zip(axes, [("Model A", "risk_score"), ("Model B", "claim_status")]):
        d = sig[sig["model"].str.startswith(mkey)].nlargest(6, "mutual_info").iloc[::-1]
        y = np.arange(len(d))
        h = 0.38
        sig_bars = ax.barh(y + h / 2, d["mutual_info"].values, h,
                           color=viz.CATEGORICAL[0], label="signal (mutual info)")
        ax.barh(y - h / 2, d["noise_floor"].values, h,
                color=viz.INK_MUTED, label="permuted-target noise floor")
        viz.label_bars(ax, sig_bars, d["mutual_info"].values, fmt="{:.3f}", horizontal=True)
        ax.set_yticks(y, labels=d["feature"].values, fontsize=8)
        ax.set_xlabel("Mutual information with target")
        ax.text(0.5, 1.03, f"{mkey} ({mlabel})", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10, color=viz.INK_SECONDARY)
        if mkey == "Model B":
            ax.legend(loc="lower right", fontsize=8)
    return "feature_signal_recap", viz.finalize(
        fig, axes[0],
        title="The performance ceiling was set in Phase 2: signal exists only for billed amount",
        subtitle="Top eligible features by mutual information with each target vs a permuted-target "
                 "noise floor (note the 10x smaller x-axis for Model A). Model A: nothing clears "
                 "the floor. Model B: billed_amount and its band - which is what Model B learns.",
        source="Source: Phase 2 mutual-information screen (phase2_eda/output/feature_signal.csv).",
        out_path=CHART_DIR / "feature_signal_recap.png",
    ), "Phase 2 feature-signal screen: why the models can only go so far."


# --------------------------------------------------------------------------
# build all + CSV exports
# --------------------------------------------------------------------------
def _export_predictions_and_summary(results: dict[str, M.ModelResult]) -> None:
    summ = []
    for key, res in results.items():
        res.test_predictions.to_csv(
            OUTPUT_DIR / f"model_{key.lower()}_test_predictions.csv", index=False)
        ev = res.eval_of(res.chosen)
        maj = res.eval_of("majority")
        summ.append({
            "model": f"Model {key} ({res.target.replace('target_', '')})",
            "chosen_estimator": res.chosen,
            "train_acc": round(res.split_accuracy["train"], 4),
            "val_acc": round(res.split_accuracy["val"], 4),
            "test_acc": round(res.split_accuracy["test"], 4),
            "majority_acc": round(maj["accuracy"], 4),
            "test_balanced_acc": round(ev["balanced_accuracy"], 4),
            "majority_balanced_acc": round(maj["balanced_accuracy"], 4),
            "test_macro_f1": round(ev["macro_f1"], 4),
            "costly_class": res.extras["costly_class"],
            "costly_class_recall": round(res.extras["costly_recall_chosen"], 4),
            "beats_majority_balanced_acc": res.beats_majority_balanced_accuracy,
        })
    pd.DataFrame(summ).to_csv(OUTPUT_DIR / "model_metrics_summary.csv", index=False)


def build_all(results: dict[str, M.ModelResult]) -> list[tuple[str, Path, str]]:
    viz.apply_house_style()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    charts = [
        chart_target_balance(results),
        chart_feature_signal_recap(results),
        chart_candidate_comparison_a(results),
        chart_candidate_comparison_b(results),
        chart_costly_class_recall(results),
        chart_confusion_matrix_a(results),
        chart_confusion_matrix_b(results),
        chart_calibration_b(results),
    ]
    _export_predictions_and_summary(results)
    return charts


if __name__ == "__main__":
    df = pd.read_parquet(PHASE2_OUT / "feature_frame.parquet")
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    sp = M.time_split(df)
    res = {k: M.train_model(sp, k) for k in ("A", "B")}
    for key, path, cap in build_all(res):
        print(f"  {key:28s} -> {path.relative_to(HERE)}")
