"""The prediction log: every served prediction lands with version metadata."""
from __future__ import annotations

import pytest

from conftest import valid_claim_payload, valid_visit_payload


@pytest.fixture()
def _require_db(_db_available):
    if not _db_available:
        pytest.skip("Postgres not reachable - prediction-log tests skipped")


def _fetch(request_id):
    from capstone.db import connect

    with connect(autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT endpoint, model, model_version, feature_spec_version, serving_version, "
            "predicted_class, probabilities, decision, defaults_applied, latency_ms "
            "FROM capstone_solution.prediction_log WHERE request_id = %s", (request_id,))
        return cur.fetchone()


def test_claim_outcome_is_logged(client, _require_db, bundle):
    r = client.post("/predict/claim-outcome", json=valid_claim_payload())
    body = r.json()
    row = _fetch(body["request_id"])
    assert row is not None
    endpoint, model, mv, fsv, sv, pc, probs, decision, defaults, latency = row
    assert endpoint == "/predict/claim-outcome"
    assert model == "B"
    assert mv == bundle.manifest["model_version"]
    assert fsv == bundle.manifest["feature_spec_version"]
    assert sv == bundle.config["serving_version"]
    assert pc == body["predicted_class"]
    assert abs(probs["Rejected"] - body["probabilities"]["Rejected"]) < 1e-9
    assert decision["threshold"] == bundle.operating_threshold
    assert float(latency) > 0


def test_visit_risk_is_logged(client, _require_db):
    r = client.post("/predict/visit-risk", json=valid_visit_payload())
    row = _fetch(r.json()["request_id"])
    assert row is not None
    assert row[1] == "A"
    assert row[7] is None  # no decision block for the monitor


def test_logging_can_be_disabled(client, _require_db, monkeypatch):
    import dataclasses

    from app import routes

    monkeypatch.setattr(routes, "CONFIG", dataclasses.replace(routes.CONFIG, log_predictions=False))
    r = client.post("/predict/visit-risk", json=valid_visit_payload())
    assert _fetch(r.json()["request_id"]) is None
