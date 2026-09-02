"""Phase 4 :: reusable model-evaluation, explainability and fairness logic.

Phase 3 shipped two calibrated, time-validated classifiers
(`phase3_models/models/`). Phase 4 loads them **as-is** (nothing is retrained or
re-tuned) and:

* scores the technical metrics - per-class precision/recall/F1, confusion
  matrices, ROC and PR curves, calibration;
* turns Model B into an operational decision - it sweeps the calibrated
  ``P(Rejected)`` and picks the threshold that **maximises net recovered denial
  leakage** (recovered leakage minus manual-review cost), chosen on the
  validation window and applied unchanged to test;
* explains Model B - sklearn permutation importance plus SHAP (global summary +
  local force/waterfall), with a permutation-only fallback if SHAP cannot
  explain the pipeline;
* verifies the Phase 2 leakage register **by ablation** - retrains throwaway
  variants with features dropped or with forbidden post-outcome fields injected,
  and shows the metric only moves when a leak is introduced;
* measures **fairness** - selection rate, recall, false-positive rate and
  calibration gap parity across gender, age band, city and insurance provider.

The notebook (`phase4_eval/phase4.ipynb`) stays thin and imports from here.

Money note: ``billed_amount`` / ``approved_amount`` / ``leakage_amount`` /
``payment_days`` are **post-outcome or reporting-only** fields. They are used
here for business-impact estimation and for the deliberate leakage-ablation
variants only - never as inputs to the shipped models.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from capstone import features as feat
from capstone import modeling as M

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

EVAL_VERSION = "1.0.0"

# Assumptions for the operating-threshold objective (maximise net recovered
# leakage). Both are documented so the threshold choice is reproducible and can
# be re-run under different numbers; units match billed_amount (INR-like).
#
#   REVIEW_COST    - fully loaded cost of a claims specialist pulling and
#                    re-working one flagged claim before submission.
#   RECOVERY_RATE  - share of a correctly-flagged rejection's leakage that is
#                    actually recoverable by pre-submission rework (the rest is
#                    denied on the merits regardless). A haircut, not 100%.
REVIEW_COST = 1_500.0
RECOVERY_RATE = 0.40

PHASE3_MODELS_DIR = Path(__file__).resolve().parents[2] / "phase3_models" / "models"

FAIRNESS_GROUPS = ("gender", "age_band", "city", "insurance_provider")

# post-outcome / reporting-only fields pulled from the spine for money maths and
# for the deliberate leakage-ablation variants (never model inputs otherwise)
_OUTCOME_COLS = ["billed_amount", "approved_amount", "leakage_amount", "payment_days"]


# --------------------------------------------------------------------------
# artefact loading & prediction frames
# --------------------------------------------------------------------------

def load_phase3(models_dir: str | Path | None = None) -> dict[str, Any]:
    """Reload the Phase 3 pipelines, calibrated wrappers and training manifest."""
    models_dir = Path(models_dir) if models_dir else PHASE3_MODELS_DIR
    manifest = json.loads((models_dir / "training_manifest.json").read_text())
    out: dict[str, Any] = {"manifest": manifest, "models_dir": models_dir}
    for key in ("A", "B"):
        out[key] = {
            "pipeline": M.load_model(models_dir, key, calibrated=False),
            "calibrated": M.load_model(models_dir, key, calibrated=True),
        }
    return out


def _feature_frame_for(split_df: pd.DataFrame, model: str):
    numeric, categorical = M.feature_lists(split_df, model)
    return split_df[numeric + categorical], numeric, categorical


def prediction_frame(splits: M.Splits, model: str, artifacts: dict,
                     spine: pd.DataFrame) -> pd.DataFrame:
    """Per-row **test-window** predictions for ``model`` ("A"/"B").

    Columns: ``visit_id``, ``visit_date``, ``actual``, ``pred_pipeline`` (the
    uncalibrated pipeline label - the Phase 3 default decision), ``prob_<class>``
    (calibrated), the fairness group columns, and the reporting-only money
    fields joined from the spine.
    """
    target = M.TARGETS[model]
    labels = M.CLASS_ORDER[model]
    test = splits.test
    X, _, _ = _feature_frame_for(test, model)

    pipe = artifacts[model]["pipeline"]
    cal = artifacts[model]["calibrated"]
    proba = cal.predict_proba(X)
    proba_classes = list(cal.classes_)

    out = pd.DataFrame({
        "visit_id": test["visit_id"].to_numpy(),
        "visit_date": pd.to_datetime(test["visit_date"].to_numpy()),
        "actual": test[target].astype(str).to_numpy(),
        "pred_pipeline": pipe.predict(X),
    })
    for cls in labels:
        j = proba_classes.index(cls) if cls in proba_classes else None
        out[f"prob_{cls.lower()}"] = proba[:, j] if j is not None else 0.0

    for g in FAIRNESS_GROUPS:
        if g in test.columns:
            out[g] = test[g].astype(str).to_numpy()

    money = spine[["visit_id", *[c for c in _OUTCOME_COLS if c in spine.columns]]]
    out = out.merge(money, on="visit_id", how="left")
    return out


# --------------------------------------------------------------------------
# technical metrics
# --------------------------------------------------------------------------

def per_class_metrics(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    return pd.DataFrame({
        "class": labels,
        "precision": prec.round(4),
        "recall": rec.round(4),
        "f1": f1.round(4),
        "support": support.astype(int),
    })


def roc_pr_frame(y_true, proba: np.ndarray, proba_classes: list[str],
                 labels: list[str]) -> pd.DataFrame:
    """One-vs-rest ROC and PR points per class as one long DataFrame.

    For a degenerate constant-probability model (Model A) every score is equal,
    so ``roc_auc`` lands at ~0.5 and the curve is the diagonal - recorded rather
    than hidden.
    """
    y_true = np.asarray(y_true)
    rows = []
    for cls in labels:
        if cls not in proba_classes:
            continue
        p = proba[:, proba_classes.index(cls)]
        pos = (y_true == cls).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue
        fpr, tpr, _ = roc_curve(pos, p)
        auc = roc_auc_score(pos, p)
        for a, b in zip(fpr, tpr):
            rows.append({"cls": cls, "curve": "roc", "x": float(a), "y": float(b),
                         "auc": float(auc), "ap": np.nan, "base_rate": float(pos.mean())})
        precision, recall, _ = precision_recall_curve(pos, p)
        ap = average_precision_score(pos, p)
        for rc, pr in zip(recall, precision):
            rows.append({"cls": cls, "curve": "pr", "x": float(rc), "y": float(pr),
                         "auc": np.nan, "ap": float(ap), "base_rate": float(pos.mean())})
    return pd.DataFrame(rows)


def reliability_frame(pipeline: Pipeline, calibrated, X: pd.DataFrame, y,
                      classes: list[str], *, n_bins: int = 8) -> pd.DataFrame:
    """Per-class reliability (mean predicted prob vs observed frequency) on the
    test window, uncalibrated pipeline vs calibrated wrapper - the Phase 4
    calibration chart input. Standalone version of
    ``capstone.modeling.reliability_curves`` that takes reloaded artefacts.
    """
    from sklearn.calibration import calibration_curve

    y = np.asarray(y)
    rows = []
    for variant, est in (("uncalibrated", pipeline), ("calibrated", calibrated)):
        proba = est.predict_proba(X)
        est_classes = list(est.classes_)
        for cls in classes:
            if cls not in est_classes:
                continue
            pos = (y == cls).astype(int)
            if pos.sum() < n_bins:
                continue
            frac_pos, mean_pred = calibration_curve(
                pos, proba[:, est_classes.index(cls)], n_bins=n_bins, strategy="quantile",
            )
            for mp, fp in zip(mean_pred, frac_pos):
                rows.append({"variant": variant, "cls": cls,
                             "mean_predicted": float(mp), "observed_freq": float(fp)})
    return pd.DataFrame(rows)


def confusion_at_threshold(pred_df: pd.DataFrame, labels: list[str], *,
                           costly: str, threshold: float) -> np.ndarray:
    """3-class confusion where ``costly`` is assigned whenever its calibrated
    probability clears ``threshold``; otherwise the argmax of the remaining
    classes. This is the operating-point decision, not the Phase 3 argmax.
    """
    prob_costly = pred_df[f"prob_{costly.lower()}"].to_numpy()
    others = [c for c in labels if c != costly]
    other_p = pred_df[[f"prob_{c.lower()}" for c in others]].to_numpy()
    pred = np.where(
        prob_costly >= threshold,
        costly,
        np.array(others)[other_p.argmax(axis=1)],
    )
    return confusion_matrix(pred_df["actual"], pred, labels=labels)


# --------------------------------------------------------------------------
# operating threshold - maximise net recovered leakage
# --------------------------------------------------------------------------

def threshold_sweep(pred_df: pd.DataFrame, *, costly: str = "Rejected",
                    grid: np.ndarray | None = None,
                    review_cost: float = REVIEW_COST,
                    recovery_rate: float = RECOVERY_RATE) -> pd.DataFrame:
    """Sweep the calibrated ``P(costly)`` cut-off.

    For each threshold: recall / precision on the costly class, how many claims
    are flagged, the recoverable denial leakage on the caught costly claims
    (``recovery_rate`` haircut applied), the review cost incurred, and the net.
    ``net_recovered`` is the objective the operating threshold maximises.
    """
    if grid is None:
        # calibrated P(Rejected) rarely exceeds ~0.3 on this data (Phase 3: the
        # calibrated head reverts toward the 60/25/15 prior), so the useful
        # operating range is low.
        grid = np.round(np.linspace(0.02, 0.40, 39), 4)
    p = pred_df[f"prob_{costly.lower()}"].to_numpy()
    is_costly = (pred_df["actual"].to_numpy() == costly)
    leak = pred_df["leakage_amount"].fillna(0.0).to_numpy()
    n = len(pred_df)

    rows = []
    for t in grid:
        flagged = p >= t
        tp = flagged & is_costly
        rec = tp.sum() / is_costly.sum() if is_costly.sum() else 0.0
        prec = tp.sum() / flagged.sum() if flagged.sum() else 0.0
        recovered = float(recovery_rate * leak[tp].sum())
        cost = float(flagged.sum() * review_cost)
        rows.append({
            "threshold": float(t),
            "recall": float(rec),
            "precision": float(prec),
            "flagged_n": int(flagged.sum()),
            "flagged_rate": float(flagged.sum() / n),
            "leakage_recovered": recovered,
            "review_cost_total": cost,
            "net_recovered": recovered - cost,
        })
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame) -> pd.Series:
    """The ``net_recovered``-maximising row of a sweep (ties -> higher threshold,
    i.e. fewer alerts)."""
    best = sweep["net_recovered"].max()
    cand = sweep[np.isclose(sweep["net_recovered"], best)]
    return cand.sort_values("threshold").iloc[-1]


def business_summary(pred_df: pd.DataFrame, threshold: float, *,
                     costly: str = "Rejected",
                     review_cost: float = REVIEW_COST,
                     recovery_rate: float = RECOVERY_RATE) -> dict[str, Any]:
    p = pred_df[f"prob_{costly.lower()}"].to_numpy()
    is_costly = pred_df["actual"].to_numpy() == costly
    leak = pred_df["leakage_amount"].fillna(0.0).to_numpy()
    flagged = p >= threshold
    tp = flagged & is_costly

    span_days = (pred_df["visit_date"].max() - pred_df["visit_date"].min()).days or 1
    months = span_days / 30.44
    recovered = float(recovery_rate * leak[tp].sum())

    return {
        "threshold": float(threshold),
        "test_rows": int(len(pred_df)),
        "costly_total": int(is_costly.sum()),
        "costly_caught": int(tp.sum()),
        "recall": float(tp.sum() / is_costly.sum()) if is_costly.sum() else 0.0,
        "precision": float(tp.sum() / flagged.sum()) if flagged.sum() else 0.0,
        "flagged_n": int(flagged.sum()),
        "alerts_per_month": float(flagged.sum() / months),
        "leakage_recovered": recovered,
        "leakage_flagged_gross": float(leak[tp].sum()),
        "leakage_total_costly": float(leak[is_costly].sum()),
        "review_cost_total": float(flagged.sum() * review_cost),
        "net_recovered": recovered - float(flagged.sum() * review_cost),
        "months_in_window": float(months),
    }


# --------------------------------------------------------------------------
# leakage verification by ablation
# --------------------------------------------------------------------------

_PATIENT_HISTORY = ["prior_visit_count", "prior_high_risk_count", "prior_rejection_count",
                    "days_since_last_visit", "prior_rejection_rate", "is_first_visit",
                    "has_prior_visit"]
_PROVIDER_HISTORY = ["provider_prior_claim_count", "provider_prior_rejection_rate"]


_ABLATION_VARIANTS = {
    "clean (shipped)": {},
    "- risk_score": {"drop": ["risk_score"]},
    "- provider history": {"drop": _PROVIDER_HISTORY},
    "- patient history": {"drop": _PATIENT_HISTORY},
    "- billed_amount": {"drop": ["billed_amount", "log_billed_amount", "billed_band"]},
    "LEAK + approved_amount": {"add_outcome": ["approved_amount"]},
    "LEAK + payment_days": {"add_outcome": ["payment_days"]},
}


def leakage_ablation(splits: M.Splits, spine: pd.DataFrame, *,
                     model: str = "B") -> pd.DataFrame:
    """Retrain Model B's Phase 3 learner (``gbm``) per feature-set variant, score
    on test.

    ``clean (shipped)`` should reproduce the Phase 3 test metrics; every
    ``LEAK +`` variant injects a forbidden post-outcome field and should show a
    clear jump - the evidence that the leakage register is load-bearing.

    Only Model B is ablated: Model A ships as a majority-class baseline that
    ignores every feature, so it is structurally incapable of leaking (its
    zero permutation importance, reported separately, is the direct evidence).
    """
    if model != "B":
        raise ValueError("leakage_ablation is Model-B only (see docstring)")
    target = M.TARGETS[model]
    labels = M.CLASS_ORDER[model]
    costly = M.COSTLY_CLASS[model]
    base_est = M.learned_estimators()["gbm"]

    numeric0, categorical0 = M.feature_lists(splits.train, model)
    outcome = spine[["visit_id", *_OUTCOME_COLS]]

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.merge(outcome, on="visit_id", how="left", suffixes=("", "_spine"))
        d["claim_status_feat"] = df[M.TARGETS["B"]].astype(str).to_numpy()
        return d

    tr, va, te = _prep(splits.train), _prep(splits.val), _prep(splits.test)
    y_tr = tr[target].astype(str)
    y_te = te[target].astype(str)

    rows = []
    for name, spec in _ABLATION_VARIANTS.items():
        num = [c for c in numeric0 if c not in spec.get("drop", [])]
        cat = [c for c in categorical0 if c not in spec.get("drop", [])]
        for col in spec.get("add_outcome", []):
            num = [*num, col]
        if spec.get("add_target_status"):
            cat = [*cat, "claim_status_feat"]
        feats = num + cat

        pipe = Pipeline([
            ("pre", M.build_preprocessor(num, cat)),
            ("clf", clone(base_est)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(tr[feats], y_tr)
        pred = pipe.predict(te[feats])
        ev = M.evaluate(y_te, pred, model=model, name=name)
        rows.append({
            "variant": name,
            "is_leak": name.startswith("LEAK"),
            "n_features": len(feats),
            "accuracy": round(ev["accuracy"], 4),
            "balanced_accuracy": round(ev["balanced_accuracy"], 4),
            "macro_f1": round(ev["macro_f1"], 4),
            f"recall_{costly.lower()}": round(ev[f"recall_{costly.lower()}"], 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# explainability
# --------------------------------------------------------------------------

def permutation_importance_frame(pipeline: Pipeline, X: pd.DataFrame, y,
                                 *, scoring: str = "balanced_accuracy",
                                 n_repeats: int = 10,
                                 random_state: int = 42) -> pd.DataFrame:
    """Permutation importance on the **raw** feature columns (whole pipeline).

    For Model A's constant classifier every column scores ~0 - reported, not
    hidden.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(pipeline, X, y, scoring=scoring,
                                   n_repeats=n_repeats, random_state=random_state)
    return (pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True))


@dataclass
class ShapResult:
    values: np.ndarray            # (n_rows, n_transformed_features) for the target class
    feature_names: list[str]
    base_value: float
    data: np.ndarray              # transformed feature matrix (same shape as values)
    target_class: str
    X_raw: pd.DataFrame

    def mean_abs(self) -> pd.DataFrame:
        return (pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(self.values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def shap_summary(pipeline: Pipeline, X_sample: pd.DataFrame, *,
                 target_class: str = "Rejected") -> ShapResult | None:
    """TreeExplainer SHAP for ``P(target_class)`` on the transformed matrix.

    Returns ``None`` (never raises) if SHAP cannot explain this pipeline, so the
    caller can fall back to permutation importance and say so.
    """
    try:
        import shap

        pre = pipeline[:-1]
        clf = pipeline[-1]
        Xt = np.asarray(pre.transform(X_sample))
        names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        classes = list(clf.classes_)
        if target_class not in classes:
            return None
        ci = classes.index(target_class)
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(Xt)
        exp_val = expl.expected_value
        if isinstance(sv, list):
            vals = sv[ci]
            base = float(np.ravel(exp_val)[ci])
        else:
            sv = np.asarray(sv)
            vals = sv[:, :, ci] if sv.ndim == 3 else sv
            base = float(np.ravel(exp_val)[ci]) if np.ndim(exp_val) else float(exp_val)
        return ShapResult(np.asarray(vals), names, base, Xt, target_class, X_sample)
    except Exception:  # noqa: BLE001 - explainability is best-effort
        return None


def shap_local_rows(pred_df: pd.DataFrame, shap_res: "ShapResult | None" = None,
                    *, costly: str = "Rejected") -> dict[str, int]:
    """Illustrative test claims for local SHAP plots, as **positional indices into
    ``shap_res.values``** (falls back to ``pred_df`` positions if no SHAP).

    Picks a caught rejection, a false alarm and a clear Paid, restricted to the
    rows SHAP actually explained.
    """
    p = pred_df[f"prob_{costly.lower()}"].to_numpy()
    actual = pred_df["actual"].to_numpy()

    if shap_res is not None:
        universe = list(shap_res.X_raw.index)          # pred_df positions, in SHAP order
        pos_of = {row: i for i, row in enumerate(universe)}
    else:
        universe = list(range(len(pred_df)))
        pos_of = {i: i for i in universe}
    allowed = np.array(universe)

    def _pick(mask, want_high):
        cand = allowed[mask[allowed]]
        if not len(cand):
            return None
        chosen = cand[np.argmax(p[cand])] if want_high else cand[np.argmin(p[cand])]
        return pos_of[int(chosen)]

    out: dict[str, int] = {}
    for label, mask, hi in (
        ("caught rejection", (actual == costly) & (p >= 0.20), True),
        ("false alarm (Paid)", (actual == "Paid") & (p >= 0.20), True),
        ("clear Paid", (actual == "Paid") & (p < 0.06), False),
    ):
        idx = _pick(mask, hi)
        if idx is not None:
            out[label] = idx
    return out


# --------------------------------------------------------------------------
# fairness
# --------------------------------------------------------------------------

def fairness_frame(pred_df: pd.DataFrame, group_col: str, *,
                   costly: str = "Rejected", threshold: float = 0.5) -> pd.DataFrame:
    """Per-subgroup operating-point parity metrics for the costly class."""
    p = pred_df[f"prob_{costly.lower()}"].to_numpy()
    is_costly = pred_df["actual"].to_numpy() == costly
    flagged = p >= threshold
    g = pred_df[group_col].to_numpy()

    rows = []
    for level in pd.unique(g):
        m = g == level
        pos, neg = m & is_costly, m & ~is_costly
        rows.append({
            "group": group_col,
            "level": str(level),
            "n": int(m.sum()),
            "base_rate": float(is_costly[m].mean()),
            "selection_rate": float(flagged[m].mean()),
            "recall": float((flagged & pos).sum() / pos.sum()) if pos.sum() else np.nan,
            "fpr": float((flagged & neg).sum() / neg.sum()) if neg.sum() else np.nan,
            "precision": float((flagged & pos).sum() / (flagged & m).sum()) if (flagged & m).sum() else np.nan,
            "calibration_gap": float(p[m].mean() - is_costly[m].mean()),
        })
    return pd.DataFrame(rows).sort_values("level").reset_index(drop=True)


def parity_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """max-min spread and min/max ratio on the parity-sensitive columns."""
    out: dict[str, Any] = {"group": frame["group"].iloc[0], "levels": len(frame)}
    for col in ("selection_rate", "recall", "fpr"):
        s = frame[col].dropna()
        if s.empty:
            continue
        out[f"{col}_gap"] = float(s.max() - s.min())
        out[f"{col}_ratio"] = float(s.min() / s.max()) if s.max() else np.nan
    out["four_fifths_pass"] = bool(out.get("selection_rate_ratio", 1.0) >= 0.8)
    return out


def fairness_all(pred_df: pd.DataFrame, *, groups: tuple[str, ...] = FAIRNESS_GROUPS,
                 costly: str = "Rejected", threshold: float = 0.5) -> dict[str, Any]:
    frames = {g: fairness_frame(pred_df, g, costly=costly, threshold=threshold)
              for g in groups if g in pred_df.columns}
    summary = pd.DataFrame([parity_summary(f) for f in frames.values()])
    return {"frames": frames, "summary": summary}


# --------------------------------------------------------------------------
# evaluation context (assembled by the notebook, consumed by charts + report)
# --------------------------------------------------------------------------

@dataclass
class EvalContext:
    manifest: dict
    review_cost: float
    preds: dict[str, pd.DataFrame]                 # "A"/"B" -> prediction_frame
    per_class: dict[str, pd.DataFrame]
    roc_pr: dict[str, pd.DataFrame]
    sweep_val: pd.DataFrame
    sweep_test: pd.DataFrame
    threshold: float
    business: dict[str, Any]
    ablation: dict[str, pd.DataFrame]              # {"B": leakage_ablation} (Model A cannot leak)
    perm_importance: dict[str, pd.DataFrame]
    reliability: dict[str, pd.DataFrame]           # "A"/"B" -> M.reliability_curves style
    fairness: dict[str, Any]                       # "B" -> fairness_all output
    shap: Any = None                               # ShapResult | None (Model B)
    shap_local: dict[str, int] = field(default_factory=dict)
    extras: dict = field(default_factory=dict)
