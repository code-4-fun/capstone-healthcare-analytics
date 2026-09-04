"""feature_drift + prediction_drift over assembled model-input frames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from capstone import monitoring as mon

from conftest import sample_reference_rows


def _served(bundle, model, rows):
    return mon.assemble_frame(
        (mon.feature_row_to_payload(r, model) for _, r in rows.iterrows()), bundle, model)


def test_unperturbed_replay_shows_no_significant_drift(bundle, reference):
    rows = sample_reference_rows("B", n=500)
    ref_served = mon.reference_served(bundle, "B", reference)
    cur_served = _served(bundle, "B", rows)
    numeric, categorical = mon.monitored_features(bundle, "B")
    fd = mon.feature_drift(ref_served, cur_served, numeric, categorical)
    assert not fd["drifted"].any(), fd[fd["drifted"]]


def test_injected_billed_amount_shift_is_flagged(bundle, reference):
    rows = sample_reference_rows("B", n=500).copy()
    rows["billed_amount"] = rows["billed_amount"] * 1.8
    ref_served = mon.reference_served(bundle, "B", reference)
    cur_served = _served(bundle, "B", rows)
    numeric, categorical = mon.monitored_features(bundle, "B")
    fd = mon.feature_drift(ref_served, cur_served, numeric, categorical).set_index("feature")

    assert fd.loc["billed_amount", "drifted"]
    assert fd.loc["billed_amount", "band"] == "significant"
    # a feature that was not touched stays put
    assert not fd.loc["gender", "drifted"]


def test_model_a_feature_drift_never_touches_billing(bundle):
    numeric, categorical = mon.monitored_features(bundle, "A")
    for banned in ("billed_amount", "log_billed_amount", "length_of_stay_hours", "risk_score"):
        assert banned not in numeric + categorical


def test_prediction_drift_detects_mix_shift():
    classes = ["Paid", "Pending", "Rejected"]
    ref = pd.Series(["Paid"] * 600 + ["Pending"] * 250 + ["Rejected"] * 150)
    drifted = pd.Series(["Paid"] * 300 + ["Pending"] * 200 + ["Rejected"] * 500)
    pd_out = mon.prediction_drift(ref, drifted, classes)
    assert pd_out["mix_psi"].iloc[0] > mon.ALERT_RULES["prediction_psi"]
    rej = pd_out.set_index("predicted_class").loc["Rejected"]
    assert rej["share_delta"] > 0.2
