"""Request validation: the typed 422 error body and the leakage-safe schemas."""
from __future__ import annotations

import pytest

from conftest import valid_claim_payload, valid_visit_payload


def _fields(resp):
    body = resp.json()
    assert body["error"] == "validation_error"
    return {e["field"] for e in body["detail"]}


def test_valid_minimal_claim_payload_ok(client):
    r = client.post("/predict/claim-outcome", json=valid_claim_payload())
    assert r.status_code == 200


@pytest.mark.parametrize("override, bad_field", [
    ({"department": "Dermatology"}, "department"),
    ({"visit_type": "Daycare"}, "visit_type"),
    ({"gender": "X"}, "gender"),
    ({"city": "Kolkata"}, "city"),
    ({"insurance_provider": "NoSuchInsurer"}, "insurance_provider"),
    ({"risk_score": "Severe"}, "risk_score"),
    ({"billed_amount": -1}, "billed_amount"),
    ({"length_of_stay_hours": -0.5}, "length_of_stay_hours"),
    ({"age": 200}, "age"),
    ({"age": -3}, "age"),
    ({"prior_rejection_rate": 1.5}, "prior_rejection_rate"),
])
def test_bad_values_rejected(client, override, bad_field):
    r = client.post("/predict/claim-outcome", json=valid_claim_payload(**override))
    assert r.status_code == 422
    assert bad_field in _fields(r)


def test_missing_required_field_rejected(client):
    p = valid_claim_payload()
    del p["billed_amount"]
    r = client.post("/predict/claim-outcome", json=p)
    assert r.status_code == 422
    assert "billed_amount" in _fields(r)


def test_visit_risk_rejects_billing_and_los_and_risk(client):
    """Model A's leakage register - the schema must not accept these at all."""
    for leak in ("billed_amount", "length_of_stay_hours", "risk_score"):
        r = client.post("/predict/visit-risk", json=valid_visit_payload(**{leak: 1}))
        assert r.status_code == 422, leak
        assert leak in _fields(r)


def test_unknown_field_rejected(client):
    r = client.post("/predict/visit-risk", json=valid_visit_payload(surprise=1))
    assert r.status_code == 422
