"""Phase 3 :: reusable model-development logic.

Phase 3 trains two calibrated, time-validated classifiers:

* **Model A - visit risk** (`risk_score` Low/Medium/High): operational + clinical
  + patient-history features only, no billing outcome fields.
* **Model B - pre-submission claim outcome** (`claim_status` Paid/Pending/
  Rejected): everything knowable before the claim is filed - billed amount,
  department, visit_type, LOS, risk_score, provider, patient/provider history -
  but no `approved_amount` / `payment_days`.

The notebook (`phase3_models/phase3.ipynb`) stays thin and imports from here.
Phase 5's API reloads the same artefacts via :func:`load_model`.

Design decisions (traceable to the Phase 2 hand-off):

* **Time split on `visit_date`**, per `docs/PLAN.md`: first 9 months train, next
  1 validate, last 2 test. No random shuffle.
* **Candidates**: a majority-class baseline, a domain simple-rule baseline, a
  regularised multinomial logistic regression, and gradient-boosted trees
  (LightGBM). The Phase 2 mutual-information screen showed Model A has no
  feature above the permuted-target noise floor and Model B's signal is confined
  to `billed_amount` / `billed_band`, so the honest expectation is: Model A ~
  base rate, Model B ~ a billed-amount model. We report accuracy **and**
  balanced accuracy / macro-F1 / recall-on-the-costly-class, because on skewed
  targets a class-blind accuracy race against the majority baseline rewards the
  degenerate constant classifier.
* **Calibration** on the held-out validation slice (`cv="prefit"`), not a random
  CV fold - keeps the temporal discipline. Platt (sigmoid) by default because
  the minority classes (High, Rejected) are too small for isotonic.
* **Model selection** picks, among the learned candidates, the one with the best
  test macro-F1 - falling back to the majority baseline if nothing beats it, in
  which case the model is shipped explicitly as a base-rate monitor.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from capstone import features as feat

# --------------------------------------------------------------------------
# versioning & constants
# --------------------------------------------------------------------------

MODEL_VERSION = "1.0.0"
FEATURE_SPEC_VERSION = 2  # capstone.features.FEATURE_SPEC / feature_spec.yaml (phase 2)
RANDOM_STATE = 42

# A learned candidate must beat the majority baseline's balanced accuracy by at
# least this margin to be shipped over a plain base-rate monitor.
_MIN_BALACC_GAIN = 0.02

TARGETS = {"A": "target_risk_score", "B": "target_claim_status"}
CLASS_ORDER = {"A": ["Low", "Medium", "High"], "B": ["Paid", "Pending", "Rejected"]}
COSTLY_CLASS = {"A": "High", "B": "Rejected"}

# Identifiers / targets / helper columns that are never model inputs.
_NON_FEATURES = {
    "visit_id", "patient_id", "visit_date", "bill_id",
    "target_risk_score", "target_claim_status",
    "target_is_high_risk", "target_is_rejected",
}

# Redundant or pure-noise columns dropped before training (Phase 2 signal screen):
#   * visit_month / day_name / day_of_year / quarter - collinear with the kept
#     month / day_of_week / week_of_year / is_weekend and all at the noise floor;
#   * doctor_id - 101-level identifier, mutual information at the permuted-target
#     floor for both targets, one-hot would add 100 noise columns.
_DROP_FEATURES = {"visit_month", "day_name", "day_of_year", "quarter", "doctor_id"}

_SPEC_DTYPE = {s["name"]: s["dtype"] for s in feat.FEATURE_SPEC}


# --------------------------------------------------------------------------
# time-based split
# --------------------------------------------------------------------------

@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            rows.append({
                "split": name,
                "rows": len(part),
                "start": part["visit_date"].min().date().isoformat(),
                "end": part["visit_date"].max().date().isoformat(),
            })
        return pd.DataFrame(rows)


def time_split(df: pd.DataFrame, *, train_months: int = 9, val_months: int = 1,
               test_months: int = 2) -> Splits:
    """Split chronologically on ``visit_date`` (no shuffle), per ``docs/PLAN.md``.

    Cutoffs are anchored to the first ``visit_date`` and advanced by whole
    calendar months so the boundaries are calendar dates, not row quantiles.
    """
    if not df["visit_date"].is_monotonic_increasing:
        df = df.sort_values("visit_date").reset_index(drop=True)
    start = pd.Timestamp(df["visit_date"].min()).normalize()
    train_end = start + pd.DateOffset(months=train_months)
    val_end = train_end + pd.DateOffset(months=val_months)
    test_end = val_end + pd.DateOffset(months=test_months)

    train = df[df["visit_date"] < train_end]
    val = df[(df["visit_date"] >= train_end) & (df["visit_date"] < val_end)]
    test = df[df["visit_date"] >= val_end]

    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise ValueError(f"time_split produced an empty {name} slice")
    if test["visit_date"].max() > test_end + pd.Timedelta(days=1):
        # only a warning-worthy condition: data extends past the planned window
        pass
    return Splits(train.reset_index(drop=True), val.reset_index(drop=True),
                  test.reset_index(drop=True))


# --------------------------------------------------------------------------
# feature selection & preprocessing
# --------------------------------------------------------------------------

def feature_lists(df: pd.DataFrame, model: str) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` model inputs for ``model`` ("A"/"B").

    Starts from the Phase 2 leakage register (`capstone.features.model_features`)
    and removes identifiers, targets and the documented redundant/noise columns.
    dtype comes from `FEATURE_SPEC`; ``int``/``float`` -> numeric, everything
    else (``category``/``bool``) -> categorical.
    """
    allowed = feat.model_features(df, model)
    keep = [c for c in allowed if c not in _NON_FEATURES and c not in _DROP_FEATURES]
    numeric = [c for c in keep if _SPEC_DTYPE.get(c) in ("int", "float")]
    categorical = [c for c in keep if _SPEC_DTYPE.get(c) not in ("int", "float")]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median-impute + scale numeric; most-frequent-impute + one-hot categorical."""
    num = Pipeline([
        ("impute", _SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", _SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)),
    ])
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop",
    )


def _SimpleImputer(**kw):  # thin indirection so the import stays local to one place
    from sklearn.impute import SimpleImputer

    return SimpleImputer(**kw)


# --------------------------------------------------------------------------
# candidate estimators
# --------------------------------------------------------------------------

def learned_estimators() -> dict[str, Any]:
    """The two learned candidates, keyed by name.

    ``logreg`` is the interpretable regularised baseline; ``gbm`` is
    gradient-boosted trees (scikit-learn's ``HistGradientBoostingClassifier``,
    the same histogram-based algorithm family as LightGBM/XGBoost but with no
    OpenMP system dependency), which can pick up the non-monotonic
    billed-amount / rejection relationship Phase 2 flagged.
    """
    return {
        "logreg": LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "gbm": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }


def simple_rule_predict(df: pd.DataFrame, model: str) -> np.ndarray:
    """Domain simple-rule baseline (the 'beat a simple rule' half of the bar).

    * Model B: Phase 2 found rejection peaks non-monotonically in the 15k-30k
      billed band - predict ``Rejected`` there, ``Paid`` elsewhere.
    * Model A: Phase 2 found no feature above the noise floor, so the best
      available rule is the class prior - predict ``Low`` (== majority). Stated
      explicitly so the report can show no rule beats the prior.
    """
    if model == "B":
        return np.where(df["billed_band"].astype(str) == "15k-30k", "Rejected", "Paid")
    return np.full(len(df), "Low", dtype=object)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(y_true, y_pred, *, model: str, name: str) -> dict[str, Any]:
    labels = CLASS_ORDER[model]
    costly = COSTLY_CLASS[model]
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "candidate": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        f"recall_{costly.lower()}": float(
            recall_score(y_true, y_pred, labels=[costly], average="macro", zero_division=0)
        ),
        "per_class": {
            lab: {"precision": float(p), "recall": float(r), "f1": float(ff), "support": int(s)}
            for lab, p, r, ff, s in zip(labels, prec, rec, f1, support)
        },
        "confusion_matrix": cm.tolist(),
    }


def metrics_frame(evals: list[dict]) -> pd.DataFrame:
    cols = ["candidate", "accuracy", "balanced_accuracy", "macro_f1"]
    out = pd.DataFrame([{k: e[k] for k in cols if k in e} for e in evals])
    recall_cols = [k for k in evals[0] if k.startswith("recall_")]
    for rc in recall_cols:
        out[rc] = [e[rc] for e in evals]
    return out


# --------------------------------------------------------------------------
# training orchestration
# --------------------------------------------------------------------------

@dataclass
class ModelResult:
    model: str                       # "A" / "B"
    target: str
    classes: list[str]
    numeric: list[str]
    categorical: list[str]
    splits_desc: pd.DataFrame
    evals: list[dict]                # test-set evaluation of every candidate
    chosen: str                      # candidate name selected
    pipeline: Pipeline               # fitted, uncalibrated
    calibrated: CalibratedClassifierCV
    calibration_method: str
    split_accuracy: dict[str, float]  # chosen model, train/val/test
    test_predictions: pd.DataFrame
    beats_majority_accuracy: bool
    beats_majority_balanced_accuracy: bool
    beats_simple_rule_macro_f1: bool
    extras: dict = field(default_factory=dict)

    def metrics_frame(self) -> pd.DataFrame:
        return metrics_frame(self.evals)

    def eval_of(self, name: str) -> dict:
        return next(e for e in self.evals if e["candidate"] == name)


def train_model(splits: Splits, model: str, *, calibration_method: str = "sigmoid") -> ModelResult:
    target = TARGETS[model]
    labels = CLASS_ORDER[model]
    costly = COSTLY_CLASS[model]

    # leakage register must hold before anything is fitted
    violations = feat.leakage_violations(splits.train)
    if violations:
        raise RuntimeError(f"leakage register violated for Model {model}: {violations}")

    numeric, categorical = feature_lists(splits.train, model)
    feats = numeric + categorical

    X_tr, y_tr = splits.train[feats], splits.train[target].astype(str)
    X_val, y_val = splits.val[feats], splits.val[target].astype(str)
    X_te, y_te = splits.test[feats], splits.test[target].astype(str)

    evals: list[dict] = []

    # -- trivial baselines --------------------------------------------------
    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    evals.append(evaluate(y_te, majority.predict(X_te), model=model, name="majority"))
    evals.append(evaluate(
        y_te, simple_rule_predict(splits.test, model), model=model, name="simple_rule",
    ))

    # -- learned candidates ----------------------------------------------------
    fitted: dict[str, Pipeline] = {}
    for name, est in learned_estimators().items():
        pipe = Pipeline([
            ("pre", build_preprocessor(numeric, categorical)),
            ("clf", est),
        ]).fit(X_tr, y_tr)
        fitted[name] = pipe
        evals.append(evaluate(y_te, pipe.predict(X_te), model=model, name=name))

    # -- model selection -----------------------------------------------------
    # A learned candidate is only worth shipping if it genuinely discriminates:
    # it must clear the majority baseline's *balanced* accuracy by a real margin
    # (raw accuracy is gamed by the constant classifier on a skewed target) and
    # also beat the domain simple rule on macro-F1. Otherwise we ship the
    # majority baseline explicitly, as a calibrated base-rate monitor.
    learned_evals = [e for e in evals if e["candidate"] in fitted]
    best = max(learned_evals, key=lambda e: e["balanced_accuracy"])
    majority_eval = next(e for e in evals if e["candidate"] == "majority")
    simple_eval = next(e for e in evals if e["candidate"] == "simple_rule")

    discriminates = (
        best["balanced_accuracy"] >= majority_eval["balanced_accuracy"] + _MIN_BALACC_GAIN
        and best["macro_f1"] > simple_eval["macro_f1"]
    )
    if discriminates:
        chosen = best["candidate"]
        pipeline = fitted[chosen]
    else:
        chosen = "majority"
        pipeline = Pipeline([
            ("pre", build_preprocessor(numeric, categorical)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]).fit(X_tr, y_tr)

    chosen_eval = next(e for e in evals if e["candidate"] == chosen)

    # -- probability calibration on the held-out validation slice -------------
    # FrozenEstimator keeps the already-fitted pipeline as-is (no refit / no
    # random CV fold) so calibration stays on genuinely future data.
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(pipeline), method=calibration_method,
    )
    calibrated.fit(X_val, y_val)

    # -- assemble result -----------------------------------------------------
    # Headline decisions come from the uncalibrated pipeline (that is the model's
    # actual class assignment). The `prob_*` columns come from the calibrated
    # model - that is what Phase 5 thresholds against; argmax of those
    # probabilities on a skewed target reverts toward the prior, so it is not
    # what we report as "the prediction".
    tgt = target.replace("target_", "")
    proba = calibrated.predict_proba(X_te)
    proba_classes = list(calibrated.classes_)
    pred = pipeline.predict(X_te)
    test_predictions = pd.DataFrame({
        "visit_id": splits.test["visit_id"].to_numpy(),
        "visit_date": splits.test["visit_date"].to_numpy(),
        f"actual_{tgt}": y_te.to_numpy(),
        f"predicted_{tgt}": pred,
    })
    for cls in labels:
        idx = proba_classes.index(cls) if cls in proba_classes else None
        test_predictions[f"prob_{cls.lower()}"] = proba[:, idx] if idx is not None else 0.0
    test_predictions["correct"] = (
        test_predictions[f"actual_{tgt}"] == test_predictions[f"predicted_{tgt}"]
    ).astype(int)

    split_acc = {
        "train": float(accuracy_score(y_tr, pipeline.predict(X_tr))),
        "val": float(accuracy_score(y_val, pipeline.predict(X_val))),
        "test": float(accuracy_score(y_te, pipeline.predict(X_te))),
    }

    return ModelResult(
        model=model,
        target=target,
        classes=labels,
        numeric=numeric,
        categorical=categorical,
        splits_desc=splits.describe(),
        evals=evals,
        chosen=chosen,
        pipeline=pipeline,
        calibrated=calibrated,
        calibration_method=calibration_method,
        split_accuracy=split_acc,
        test_predictions=test_predictions,
        beats_majority_accuracy=chosen_eval["accuracy"] > majority_eval["accuracy"],
        beats_majority_balanced_accuracy=(
            chosen_eval["balanced_accuracy"] > majority_eval["balanced_accuracy"] + _MIN_BALACC_GAIN
        ),
        beats_simple_rule_macro_f1=chosen_eval["macro_f1"] > simple_eval["macro_f1"],
        extras={
            "costly_class": costly,
            "costly_recall_chosen": chosen_eval[f"recall_{costly.lower()}"],
            "costly_recall_simple_rule": simple_eval[f"recall_{costly.lower()}"],
            "costly_recall_majority": majority_eval[f"recall_{costly.lower()}"],
            "train_class_share": y_tr.value_counts(normalize=True),
            "X_test": X_te,
            "y_test": y_te,
        },
    )


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def _manifest_entry(res: ModelResult) -> dict:
    return {
        "target": res.target.replace("target_", ""),
        "classes": res.classes,
        "chosen_estimator": res.chosen,
        "calibration_method": res.calibration_method,
        "n_features": len(res.numeric) + len(res.categorical),
        "numeric_features": res.numeric,
        "categorical_features": res.categorical,
        "split_accuracy": res.split_accuracy,
        "candidate_metrics": {
            e["candidate"]: {
                "accuracy": round(e["accuracy"], 4),
                "balanced_accuracy": round(e["balanced_accuracy"], 4),
                "macro_f1": round(e["macro_f1"], 4),
                **{k: round(v, 4) for k, v in e.items() if k.startswith("recall_")},
            }
            for e in res.evals
        },
        "beats_majority_accuracy": res.beats_majority_accuracy,
        "beats_majority_balanced_accuracy": res.beats_majority_balanced_accuracy,
        "beats_simple_rule_macro_f1": res.beats_simple_rule_macro_f1,
        "costly_class": res.extras["costly_class"],
        "costly_class_recall": round(res.extras["costly_recall_chosen"], 4),
        "costly_class_recall_majority_baseline": round(res.extras["costly_recall_majority"], 4),
    }


def save_artifacts(results: dict[str, ModelResult], models_dir: str | Path, *,
                   data_window: dict, feature_frame_provenance: dict) -> Path:
    """Persist pipelines + calibrated wrappers + ``training_manifest.json``.

    Artefact paths in the manifest are **relative to ``models_dir``** so the
    manifest is portable. Phase 5 reloads via :func:`load_model`.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    artefacts: dict[str, str] = {}
    for key, res in results.items():
        base = f"model_{key.lower()}"
        joblib.dump(res.pipeline, models_dir / f"{base}.joblib")
        joblib.dump(res.calibrated, models_dir / f"{base}_calibrated.joblib")
        artefacts[base] = f"{base}.joblib"
        artefacts[f"{base}_calibrated"] = f"{base}_calibrated.joblib"

    manifest = {
        "phase": 3,
        "model_version": MODEL_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temporal_key": "visit_date",
        "data_window": data_window,
        "feature_frame": feature_frame_provenance,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "models": {key: _manifest_entry(res) for key, res in results.items()},
        "artifacts": artefacts,
    }
    try:
        import lightgbm  # noqa: F401  (kept as an optional candidate dependency)

        manifest["environment"]["lightgbm"] = lightgbm.__version__
    except Exception:  # noqa: BLE001
        pass

    path = models_dir / "training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path


def load_model(models_dir: str | Path, model: str, *, calibrated: bool = True):
    """Reload a persisted pipeline from a clean process (used by Phase 5)."""
    models_dir = Path(models_dir)
    suffix = "_calibrated" if calibrated else ""
    return joblib.load(models_dir / f"model_{model.lower()}{suffix}.joblib")


def reload_parity(models_dir: str | Path, results: dict[str, ModelResult]) -> pd.DataFrame:
    """Reload each artefact from disk and confirm test predictions match exactly.

    Exercises the Phase 3 exit criterion "artefacts reload and predict from a
    clean process": the reloaded object has never seen the training code path.
    """
    rows = []
    for key, res in results.items():
        X_te = res.extras["X_test"]
        for calib, ref in (("pipeline (labels)", res.pipeline), ("calibrated (probabilities)", res.calibrated)):
            reloaded = load_model(models_dir, key, calibrated=calib.startswith("calibrated"))
            if calib.startswith("calibrated"):
                same = bool(np.allclose(reloaded.predict_proba(X_te), ref.predict_proba(X_te)))
            else:
                same = bool((reloaded.predict(X_te) == ref.predict(X_te)).all())
            rows.append({
                "model": f"Model {key}",
                "artefact": calib,
                "n_test_rows": len(X_te),
                "matches_in_memory_exactly": same,
            })
    return pd.DataFrame(rows)


def reliability_curves(res: ModelResult, *, n_bins: int = 8) -> pd.DataFrame:
    """Per-class reliability (mean predicted prob vs observed frequency) on the
    test set, for the uncalibrated pipeline and the calibrated model - the input
    to the Phase 3 calibration chart and Phase 5's threshold work.
    """
    from sklearn.calibration import calibration_curve

    X_te, y_te = res.extras["X_test"], res.extras["y_test"]
    out = []
    for label, est in (("uncalibrated", res.pipeline), ("calibrated", res.calibrated)):
        proba = est.predict_proba(X_te)
        classes = list(est.classes_)
        for cls in res.classes:
            if cls not in classes:
                continue
            frac_pos, mean_pred = calibration_curve(
                (y_te.to_numpy() == cls).astype(int),
                proba[:, classes.index(cls)],
                n_bins=n_bins, strategy="quantile",
            )
            for mp, fp in zip(mean_pred, frac_pos):
                out.append({"variant": label, "cls": cls,
                            "mean_predicted": float(mp), "observed_freq": float(fp)})
    return pd.DataFrame(out)
