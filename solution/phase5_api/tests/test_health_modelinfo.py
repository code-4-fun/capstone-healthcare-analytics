"""/health and /model-info: shape, versions and the Model A monitor notice."""
from __future__ import annotations

import json

from capstone import serving as S

MANIFEST = json.loads((S.PHASE3_MODELS_DIR / "training_manifest.json").read_text())


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == {"A": True, "B": True}
    assert body["model_version"] == MANIFEST["model_version"]
    assert body["uptime_seconds"] >= 0


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/docs"


def test_model_info_matches_manifest(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == MANIFEST["model_version"]
    assert body["feature_spec_version"] == MANIFEST["feature_spec_version"]
    assert set(body["models"]) == {"A", "B"}

    a, b = body["models"]["A"], body["models"]["B"]
    assert a["n_features"] == MANIFEST["models"]["A"]["n_features"]
    assert b["n_features"] == MANIFEST["models"]["B"]["n_features"]

    # Model B ships an operating threshold; Model A ships the monitor notice
    assert b["operating_threshold"] is not None
    assert 0 < b["operating_threshold"] < 0.5
    assert b["threshold_version"]
    assert a["monitor_only"] is True
    assert "base-rate monitor" in a["monitor_notice"]

    # domains are advertised for client-side validation
    assert body["categorical_domains"]["department"] == S.DOMAINS["department"]
