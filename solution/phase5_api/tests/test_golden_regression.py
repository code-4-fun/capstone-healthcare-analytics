"""Golden-prediction regression.

Each fixture case is a reconstructed API payload plus the **persisted Phase 3
prediction** for that visit. The Phase 5 serving path must reproduce the class
label exactly and the calibrated probabilities to 1e-6 - proof that the API's
feature assembly is identical to the training-time assembly (same transforms,
same as-of rules, same column order).
"""
from __future__ import annotations

import pytest

TOL = 1e-6


@pytest.mark.parametrize("model, endpoint", [
    ("A", "/predict/visit-risk"),
    ("B", "/predict/claim-outcome"),
])
def test_serving_reproduces_phase3(client, golden, model, endpoint):
    cases = golden[model]["cases"]
    assert len(cases) >= 20
    mismatches = []
    for case in cases:
        r = client.post(endpoint, json=case["payload"])
        assert r.status_code == 200, (case["visit_id"], r.json())
        body = r.json()
        exp = case["expected"]
        if body["predicted_class"] != exp["predicted_class"]:
            mismatches.append((case["visit_id"], "label", body["predicted_class"], exp["predicted_class"]))
        for cls, p in exp["probabilities"].items():
            if abs(body["probabilities"][cls] - p) > TOL:
                mismatches.append((case["visit_id"], cls, body["probabilities"][cls], p))
    assert not mismatches, mismatches
