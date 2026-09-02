"""Prediction responses: schema, decision logic, defaults, versions, latency."""
from __future__ import annotations

from conftest import valid_claim_payload, valid_visit_payload

HISTORY = [
    "prior_visit_count", "prior_high_risk_count", "prior_rejection_count",
    "days_since_last_visit", "prior_rejection_rate", "doctor_load_30d",
    "provider_prior_claim_count", "provider_prior_rejection_rate",
]


def test_claim_outcome_response_shape(client, bundle):
    r = client.post("/predict/claim-outcome", json=valid_claim_payload())
    assert r.status_code == 200
    b = r.json()
    assert b["model"] == "B"
    assert b["model_version"] == bundle.manifest["model_version"]
    assert b["feature_spec_version"] == bundle.manifest["feature_spec_version"]
    assert b["predicted_class"] in ("Paid", "Pending", "Rejected")
    assert abs(sum(b["probabilities"].values()) - 1.0) < 1e-6
    assert set(b["probabilities"]) == {"Paid", "Pending", "Rejected"}
    assert b["latency_ms"] > 0
    assert b["request_id"]


def test_claim_outcome_decision_tracks_threshold(client, bundle):
    thr = bundle.operating_threshold
    # a large ICU claim in the high-rejection band -> likely flagged
    r = client.post("/predict/claim-outcome", json=valid_claim_payload(
        billed_amount=24000, visit_type="ICU", department="ICU", risk_score="High"))
    d = r.json()["decision"]
    assert d["threshold"] == thr
    assert d["p_rejected"] == r.json()["probabilities"]["Rejected"]
    assert d["flagged_for_review"] is (d["p_rejected"] >= thr)
    assert d["action"] == ("review" if d["flagged_for_review"] else "submit")


def test_defaults_applied_lists_omitted_history(client):
    r = client.post("/predict/claim-outcome", json=valid_claim_payload())
    assert sorted(r.json()["defaults_applied"]) == sorted(HISTORY)


def test_supplied_history_not_defaulted(client):
    payload = valid_claim_payload(
        prior_visit_count=4, prior_high_risk_count=1, prior_rejection_count=2,
        prior_rejection_rate=0.5, days_since_last_visit=30, doctor_load_30d=12,
        provider_prior_claim_count=900, provider_prior_rejection_rate=0.17)
    r = client.post("/predict/claim-outcome", json=payload)
    assert r.json()["defaults_applied"] == []


def test_visit_risk_is_a_monitor(client):
    r = client.post("/predict/visit-risk", json=valid_visit_payload())
    assert r.status_code == 200
    b = r.json()
    assert b["model"] == "A"
    assert b["predicted_class"] == "Low"  # base-rate monitor: constant
    assert "base-rate monitor" in b["monitor_notice"]
    assert abs(sum(b["probabilities"].values()) - 1.0) < 1e-6
