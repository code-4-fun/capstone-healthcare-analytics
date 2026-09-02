"""Phase 5 :: reusable model-serving logic.

Phase 3 persisted two calibrated, time-validated pipelines
(``phase3_models/models/``); Phase 4 evaluated them and chose Model B's operating
threshold on ``P(Rejected)``. Phase 5's FastAPI app (``phase5_api/app/``) is thin
and imports everything here:

* :data:`DOMAINS` - the allowed categorical values (Phase 1 CHECK-constraint
  domains), so request validation and the feature assembly agree on one list.
* :func:`build_model_row` - turn a validated request payload into the exact
  one-row feature frame the Phase 3 pipeline expects, deriving the same
  transforms as :func:`capstone.features.build_feature_frame` and filling the
  as-of history aggregates with a documented no-history profile when the caller
  omits them.
* :func:`load_serving_bundle` / :func:`predict_claim_outcome` /
  :func:`predict_visit_risk` - reload the artefacts once and score a request.
* :func:`derive_operating_threshold` - rebuild the Phase 4 threshold choice so
  ``serving_config.json`` is reproducible rather than a copied constant.

Leakage discipline is inherited from Phase 2/3: Model A (visit risk) never sees a
billing or LOS field, Model B (pre-submission claim outcome) never sees a
post-outcome field. The request schemas enforce this at the edge; this module
enforces it again by only ever assembling the columns in each model's manifest
feature list.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from capstone import modeling as M

log = logging.getLogger("capstone.serving")

# --------------------------------------------------------------------------
# versioning
# --------------------------------------------------------------------------

SERVING_VERSION = "1.0.0"
THRESHOLD_VERSION = "1.0.0"

# Fallback for Model B's operating threshold when the DB (needed to rebuild the
# Phase 4 choice from leakage_amount) is unreachable. This is the value recorded
# in phase4_eval/model_card_B.md.
DEFAULT_OPERATING_THRESHOLD = 0.19
THRESHOLD_BASIS = (
    "calibrated P(Rejected) cut-off maximising net recoverable denial leakage on "
    "the validation month (Rs. 1,500/review, 40% of a caught rejection recoverable); "
    "Phase 4"
)

MODEL_A_MONITOR_NOTICE = (
    "Model A is a calibrated base-rate monitor, not a per-visit predictor. It has "
    "no signal above the class prior on the pilot year (Phase 2 mutual-information "
    "screen, Phase 3 model selection) and predicts 'Low' for every visit. Use it "
    "only to track the risk-mix distribution over time; do not use its output to "
    "prioritise or staff an individual visit. See phase4_eval/model_card_A.md."
)

# --------------------------------------------------------------------------
# categorical domains (Phase 1 CHECK-constraint values)
# --------------------------------------------------------------------------

DOMAINS: dict[str, list[str]] = {
    "department": ["Cardiology", "ER", "General", "ICU", "Neurology", "Orthopedics"],
    "visit_type": ["ER", "ICU", "OPD"],
    "gender": ["F", "M"],
    "city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"],
    "insurance_provider": ["CareOne", "HealthPlus", "MediCareX", "SecureLife"],
    "risk_score": ["Low", "Medium", "High"],
}

# As-of history aggregates. The API request accepts these as optional; when the
# caller does not supply them (a first-time integration without the Phase 1
# analytics layer wired in) we assume "no prior history" and record which fields
# were defaulted in the response so the caller knows the prediction is a
# no-history estimate.
HISTORY_FIELDS: tuple[str, ...] = (
    "prior_visit_count",
    "prior_high_risk_count",
    "prior_rejection_count",
    "days_since_last_visit",
    "prior_rejection_rate",
    "doctor_load_30d",
    "provider_prior_claim_count",
    "provider_prior_rejection_rate",
)

HISTORY_DEFAULTS: dict[str, float] = {
    "prior_visit_count": 0.0,
    "prior_high_risk_count": 0.0,
    "prior_rejection_count": 0.0,
    "days_since_last_visit": np.nan,   # median-imputed by the fitted pipeline
    "prior_rejection_rate": np.nan,
    "doctor_load_30d": 0.0,
    "provider_prior_claim_count": 0.0,
    "provider_prior_rejection_rate": np.nan,
}

PHASE5_DIR = Path(__file__).resolve().parents[2] / "phase5_api"
PHASE3_MODELS_DIR = Path(__file__).resolve().parents[2] / "phase3_models" / "models"
PHASE2_PARQUET = Path(__file__).resolve().parents[2] / "phase2_eda" / "output" / "feature_frame.parquet"


# --------------------------------------------------------------------------
# feature assembly
# --------------------------------------------------------------------------

def _age_band(age: int) -> str:
    """Same bands as ``v_visit_billing`` / ``feature_spec.yaml``."""
    if age < 18:
        return "0-17"
    if age < 35:
        return "18-34"
    if age < 50:
        return "35-49"
    if age < 65:
        return "50-64"
    return "65+"


def _billed_band(amount: float) -> str:
    # identical cut points to capstone.features._billed_band / capstone.eda
    if amount < 5000:
        return "<5k"
    if amount < 15000:
        return "5k-15k"
    if amount < 30000:
        return "15k-30k"
    return "30k+"


def _seasonality(d: date) -> dict[str, Any]:
    ts = pd.Timestamp(d)
    return {
        "month": ts.month,
        "day_of_week": ts.dayofweek,
        "week_of_year": int(ts.isocalendar().week),
        "is_weekend": ts.dayofweek >= 5,
    }


def resolve_history(payload: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Return ``(history_values, defaulted_field_names)``.

    ``prior_rejection_rate`` is derived from the counts when the caller gives the
    counts but not the rate, matching :func:`capstone.features.build_feature_frame`.
    """
    hist: dict[str, float] = {}
    defaulted: list[str] = []
    for f in HISTORY_FIELDS:
        v = payload.get(f)
        if v is None:
            hist[f] = HISTORY_DEFAULTS[f]
            defaulted.append(f)
        else:
            hist[f] = float(v)

    if payload.get("prior_rejection_rate") is None:
        pvc = hist["prior_visit_count"]
        hist["prior_rejection_rate"] = (
            hist["prior_rejection_count"] / pvc if pvc > 0 else np.nan
        )
    return hist, defaulted


def build_model_row(payload: dict[str, Any], model: str,
                    feature_lists: dict[str, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    """One-row feature frame for ``model`` ("A"/"B"), plus the list of history
    fields that were defaulted.

    ``feature_lists`` is ``{"numeric": [...], "categorical": [...]}`` from
    ``serving_config.json`` (copied verbatim from the Phase 3 training manifest),
    so the row carries exactly the columns the persisted pipeline was fitted on -
    no more (which would risk leakage) and no fewer.
    """
    hist, defaulted = resolve_history(payload)
    vd = payload["visit_date"]
    if isinstance(vd, str):
        vd = date.fromisoformat(vd)

    row: dict[str, Any] = {}
    row.update(hist)
    row.update(_seasonality(vd))
    row["age"] = int(payload["age"])
    row["age_band"] = _age_band(int(payload["age"]))
    row["chronic_flag"] = int(bool(payload["chronic_flag"]))
    row["gender"] = payload["gender"]
    row["city"] = payload["city"]
    row["department"] = payload["department"]
    row["visit_type"] = payload["visit_type"]
    row["insurance_provider"] = payload["insurance_provider"]
    row["is_first_visit"] = hist["prior_visit_count"] == 0
    row["has_prior_visit"] = hist["prior_visit_count"] > 0

    # Model B-only pre-submission fields (never assembled for Model A)
    if model == "B":
        billed = float(payload["billed_amount"])
        los = float(payload["length_of_stay_hours"])
        row["billed_amount"] = billed
        row["log_billed_amount"] = float(np.log1p(billed))
        row["billed_band"] = _billed_band(billed)
        row["billed_at_floor"] = bool(np.isclose(billed, 500.0))
        row["length_of_stay_hours"] = los
        row["los_at_floor"] = bool(np.isclose(los, 0.5))
        row["risk_score"] = payload["risk_score"]

    cols = feature_lists["numeric"] + feature_lists["categorical"]
    missing = [c for c in cols if c not in row]
    if missing:
        raise KeyError(f"feature assembly for Model {model} is missing {missing}")

    frame = pd.DataFrame([{c: row[c] for c in cols}])
    for c in feature_lists["numeric"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
    return frame, defaulted


# --------------------------------------------------------------------------
# serving bundle
# --------------------------------------------------------------------------

class ServingBundle:
    """Loaded-once container: the two pipelines, their calibrated wrappers, the
    training manifest and the serving config."""

    def __init__(self, models_dir: Path, config: dict[str, Any], manifest: dict[str, Any]):
        self.models_dir = models_dir
        self.config = config
        self.manifest = manifest
        self.pipelines = {k: M.load_model(models_dir, k, calibrated=False) for k in ("A", "B")}
        self.calibrated = {k: M.load_model(models_dir, k, calibrated=True) for k in ("A", "B")}

    def feature_lists(self, model: str) -> dict[str, list[str]]:
        m = self.config["models"][model]
        return {"numeric": m["numeric_features"], "categorical": m["categorical_features"]}

    @property
    def operating_threshold(self) -> float:
        return float(self.config["models"]["B"]["operating_threshold"])


def load_serving_bundle(models_dir: str | Path | None = None,
                        serving_config_path: str | Path | None = None) -> ServingBundle:
    models_dir = Path(models_dir) if models_dir else PHASE3_MODELS_DIR
    cfg_path = Path(serving_config_path) if serving_config_path else (PHASE5_DIR / "serving_config.json")
    manifest = json.loads((models_dir / "training_manifest.json").read_text())
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text())
    else:  # first boot before run_phase5.py has generated it - fall back to the manifest
        log.warning("serving_config.json not found at %s; deriving a default config from the "
                    "training manifest with threshold %.2f", cfg_path, DEFAULT_OPERATING_THRESHOLD)
        config = default_serving_config(manifest, DEFAULT_OPERATING_THRESHOLD, threshold_from="fallback")
    return ServingBundle(models_dir, config, manifest)


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------

def _probabilities(calibrated, X: pd.DataFrame, classes: list[str]) -> dict[str, float]:
    proba = calibrated.predict_proba(X)[0]
    order = list(calibrated.classes_)
    return {c: float(proba[order.index(c)]) if c in order else 0.0 for c in classes}


def predict_claim_outcome(bundle: ServingBundle, payload: dict[str, Any]) -> dict[str, Any]:
    fl = bundle.feature_lists("B")
    X, defaulted = build_model_row(payload, "B", fl)
    classes = M.CLASS_ORDER["B"]
    label = str(bundle.pipelines["B"].predict(X)[0])
    probs = _probabilities(bundle.calibrated["B"], X, classes)
    threshold = bundle.operating_threshold
    p_rej = probs["Rejected"]
    flagged = p_rej >= threshold
    return {
        "model": "B",
        "model_version": bundle.manifest["model_version"],
        "feature_spec_version": bundle.manifest["feature_spec_version"],
        "predicted_class": label,
        "probabilities": probs,
        "decision": {
            "action": "review" if flagged else "submit",
            "flagged_for_review": bool(flagged),
            "p_rejected": p_rej,
            "threshold": threshold,
            "threshold_version": bundle.config["models"]["B"].get("threshold_version", THRESHOLD_VERSION),
        },
        "defaults_applied": defaulted,
    }


def predict_visit_risk(bundle: ServingBundle, payload: dict[str, Any]) -> dict[str, Any]:
    fl = bundle.feature_lists("A")
    X, defaulted = build_model_row(payload, "A", fl)
    classes = M.CLASS_ORDER["A"]
    label = str(bundle.pipelines["A"].predict(X)[0])
    probs = _probabilities(bundle.calibrated["A"], X, classes)
    return {
        "model": "A",
        "model_version": bundle.manifest["model_version"],
        "feature_spec_version": bundle.manifest["feature_spec_version"],
        "predicted_class": label,
        "probabilities": probs,
        "monitor_notice": MODEL_A_MONITOR_NOTICE,
        "defaults_applied": defaulted,
    }


# --------------------------------------------------------------------------
# serving config (written by run_phase5.py, read by the app)
# --------------------------------------------------------------------------

def default_serving_config(manifest: dict[str, Any], threshold: float, *,
                           threshold_from: str) -> dict[str, Any]:
    from datetime import datetime, timezone

    models: dict[str, Any] = {}
    for k, entry in manifest["models"].items():
        m = {
            "target": entry["target"],
            "classes": entry["classes"],
            "numeric_features": entry["numeric_features"],
            "categorical_features": entry["categorical_features"],
            "chosen_estimator": entry["chosen_estimator"],
            "calibration_method": entry["calibration_method"],
        }
        if k == "A":
            m["monitor_only"] = entry["chosen_estimator"] == "majority"
            m["monitor_notice"] = MODEL_A_MONITOR_NOTICE
        if k == "B":
            m["operating_threshold"] = round(float(threshold), 4)
            m["threshold_version"] = THRESHOLD_VERSION
            m["threshold_basis"] = THRESHOLD_BASIS
            m["threshold_source"] = threshold_from
        models[k] = m

    return {
        "serving_version": SERVING_VERSION,
        "model_version": manifest["model_version"],
        "feature_spec_version": manifest["feature_spec_version"],
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temporal_key": manifest["temporal_key"],
        "data_window": manifest["data_window"],
        "categorical_domains": DOMAINS,
        "models": models,
    }


def derive_operating_threshold(models_dir: str | Path | None = None,
                               parquet: str | Path | None = None) -> tuple[float, str]:
    """Rebuild Phase 4's Model B operating-threshold choice.

    Returns ``(threshold, source)`` where source is ``"rederived"`` when the
    Phase 4 sweep ran, or ``"fallback"`` when the DB / parquet was unavailable
    and :data:`DEFAULT_OPERATING_THRESHOLD` was used.
    """
    models_dir = Path(models_dir) if models_dir else PHASE3_MODELS_DIR
    parquet = Path(parquet) if parquet else PHASE2_PARQUET
    try:
        from capstone import eda
        from capstone import evaluation as E

        df = pd.read_parquet(parquet)
        df["visit_date"] = pd.to_datetime(df["visit_date"])
        splits = M.time_split(df)
        art = E.load_phase3(models_dir)
        spine = eda.load_spine()
        val_split = M.Splits(splits.train, splits.val, splits.val)
        pred_val = E.prediction_frame(val_split, "B", art, spine)
        sweep = E.threshold_sweep(pred_val)
        chosen = E.choose_threshold(sweep)
        return float(chosen["threshold"]), "rederived"
    except Exception as exc:  # noqa: BLE001 - reproducibility is best-effort
        log.warning("could not re-derive the operating threshold (%s); using the documented "
                    "fallback %.2f", exc, DEFAULT_OPERATING_THRESHOLD)
        return DEFAULT_OPERATING_THRESHOLD, "fallback"


def write_serving_config(path: str | Path | None = None,
                         models_dir: str | Path | None = None,
                         parquet: str | Path | None = None) -> Path:
    models_dir = Path(models_dir) if models_dir else PHASE3_MODELS_DIR
    path = Path(path) if path else (PHASE5_DIR / "serving_config.json")
    manifest = json.loads((models_dir / "training_manifest.json").read_text())
    threshold, source = derive_operating_threshold(models_dir, parquet)
    config = default_serving_config(manifest, threshold, threshold_from=source)
    path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    return path
