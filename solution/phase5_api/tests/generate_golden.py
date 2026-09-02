"""Regenerate the golden-prediction fixtures.

    uv run python phase5_api/tests/generate_golden.py

Picks a spread of test-window visits from the Phase 2 feature frame, reconstructs
the API request payload for each, and records the **persisted Phase 3
predictions** (`phase3_models/output/model_{a,b}_test_predictions.csv`) as the
expected output. `test_golden_regression.py` then asserts the Phase 5 serving
path reproduces those exactly - the contract that the API's feature assembly
equals the training-time assembly.

Run this only when the Phase 3 models are deliberately retrained; commit the
result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from capstone import modeling as M
from capstone import serving as S

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
PHASE3_OUT = HERE.parents[1] / "phase3_models" / "output"
N_PER_MODEL = 40
RANDOM_STATE = 20260902

_HISTORY_COLS = {
    "prior_visit_count": int,
    "prior_high_risk_count": int,
    "prior_rejection_count": int,
    "prior_rejection_rate": float,
    "days_since_last_visit": float,
    "doctor_load_30d": int,
    "provider_prior_claim_count": float,
    "provider_prior_rejection_rate": float,
}


def _opt(value, cast):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return cast(value)


def _payload(row: pd.Series, model: str) -> dict:
    p = {
        "visit_date": pd.Timestamp(row["visit_date"]).date().isoformat(),
        "department": row["department"],
        "visit_type": row["visit_type"],
        "age": int(row["age"]),
        "gender": row["gender"],
        "city": row["city"],
        "insurance_provider": row["insurance_provider"],
        "chronic_flag": bool(row["chronic_flag"]),
    }
    for col, cast in _HISTORY_COLS.items():
        p[col] = _opt(row.get(col), cast)
    if model == "B":
        p["billed_amount"] = float(row["billed_amount"])
        p["length_of_stay_hours"] = float(row["length_of_stay_hours"])
        p["risk_score"] = row["risk_score"]
    return p


def build() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(S.PHASE2_PARQUET)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    test = M.time_split(df).test

    for model, tgt in (("A", "risk_score"), ("B", "claim_status")):
        preds = pd.read_csv(PHASE3_OUT / f"model_{model.lower()}_test_predictions.csv")
        classes = [c.lower() for c in M.CLASS_ORDER[model]]
        sample = preds.sample(N_PER_MODEL, random_state=RANDOM_STATE).sort_values("visit_id")
        rows = []
        for _, pr in sample.iterrows():
            src = test.loc[test["visit_id"] == pr["visit_id"]].iloc[0]
            rows.append({
                "visit_id": int(pr["visit_id"]),
                "payload": _payload(src, model),
                "expected": {
                    "predicted_class": str(pr[f"predicted_{tgt}"]),
                    "probabilities": {c.capitalize(): float(pr[f"prob_{c}"]) for c in classes},
                },
            })
        out = GOLDEN / f"model_{model.lower()}_golden.json"
        out.write_text(json.dumps({
            "model": model,
            "source": f"phase3_models/output/model_{model.lower()}_test_predictions.csv",
            "model_version": M.MODEL_VERSION,
            "n": len(rows),
            "cases": rows,
        }, indent=2) + "\n")
        print("wrote", out.relative_to(HERE.parents[1]), f"({len(rows)} cases)")


if __name__ == "__main__":
    build()
