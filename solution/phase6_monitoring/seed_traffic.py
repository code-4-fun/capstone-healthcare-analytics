"""Phase 6 :: seed the prediction log with two comparable traffic windows.

``prediction_log`` only fills from live API calls, so a fresh environment has
nothing to monitor. This module replays real Phase 3 **test-window** visits
through the exact serving path the API uses
(:func:`capstone.serving.predict_claim_outcome` / :func:`predict_visit_risk`)
and writes the resulting log rows with controlled timestamps and a
``client_host`` tag:

* ``phase6-seed:baseline`` - an un-perturbed window (synthetic "last week"),
* ``phase6-seed:drift``    - the same kind of traffic with deliberate drift
  injected (synthetic "this week"): billed amounts inflated, the department mix
  shifted toward ER, patients older, one insurer over-represented, and a
  volume spike.

The drift job then compares the two windows against the Phase 3 test-window
reference and raises alerts. Seeding is idempotent - existing ``phase6-seed:*``
rows are removed first.

Faithful: Phase 5's golden-regression test already proves ``serving.predict_*``
reproduces the Phase 3 training-time assembly to 1e-6; here we only control the
``ts`` and the request volume.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from capstone import modeling as M
from capstone import monitoring as mon
from capstone import serving as S

log = logging.getLogger("capstone.phase6.seed")

BASELINE_TAG = "phase6-seed:baseline"
DRIFT_TAG = "phase6-seed:drift"

#: The un-perturbed request payload for a feature-frame row (shared with the
#: reference assembly in :mod:`capstone.monitoring`).
payload_from_row = mon.feature_row_to_payload

_INSERT = """
INSERT INTO capstone_solution.prediction_log
    (ts, request_id, endpoint, model, model_version, feature_spec_version, serving_version,
     operating_threshold, predicted_class, probabilities, decision, defaults_applied,
     request_payload, latency_ms, client_host)
VALUES
    (%(ts)s, %(request_id)s, %(endpoint)s, %(model)s, %(model_version)s, %(feature_spec_version)s,
     %(serving_version)s, %(operating_threshold)s, %(predicted_class)s, %(probabilities)s,
     %(decision)s, %(defaults_applied)s, %(request_payload)s, %(latency_ms)s, %(client_host)s)
"""

_ENDPOINT = {"A": "/predict/visit-risk", "B": "/predict/claim-outcome"}
_LATENCY_MS = {"A": (5.5, 1.2), "B": (20.0, 5.0)}   # (mean, sd) - synthetic, plausible


# --------------------------------------------------------------------------
# payloads
# --------------------------------------------------------------------------

def inject_drift(payload: dict, rng: np.random.Generator) -> dict:
    """Deliberate covariate + prior shift, mirroring a plausible real-world
    change in the patient / billing mix."""
    p = dict(payload)
    p["age"] = int(min(120, p["age"] + 12))                      # ageing catchment
    if p["department"] != "ER" and rng.random() < 0.45:          # ER-mix shift
        p["department"] = "ER"
    if rng.random() < 0.55:                                      # insurer concentration
        p["insurance_provider"] = "MediCareX"
    if "billed_amount" in p:                                     # tariff inflation
        p["billed_amount"] = round(float(p["billed_amount"]) * 1.6, 2)
    return p


# --------------------------------------------------------------------------
# scoring + insert
# --------------------------------------------------------------------------

def _score(bundle: S.ServingBundle, model: str, payload: dict) -> dict:
    if model == "B":
        return S.predict_claim_outcome(bundle, payload)
    return S.predict_visit_risk(bundle, payload)


def _log_row(model: str, payload: dict, resp: dict, *, ts: datetime, tag: str,
             rng: np.random.Generator) -> dict:
    mean, sd = _LATENCY_MS[model]
    decision = resp.get("decision")
    return {
        "ts": ts,
        "request_id": str(uuid.uuid4()),
        "endpoint": _ENDPOINT[model],
        "model": model,
        "model_version": resp["model_version"],
        "feature_spec_version": resp["feature_spec_version"],
        "serving_version": S.SERVING_VERSION,
        "operating_threshold": decision["threshold"] if decision else None,
        "predicted_class": resp["predicted_class"],
        "probabilities": json.dumps(resp["probabilities"]),
        "decision": json.dumps(decision) if decision else None,
        "defaults_applied": json.dumps(resp["defaults_applied"]),
        "request_payload": json.dumps(payload, default=str),
        "latency_ms": round(float(max(0.5, rng.normal(mean, sd))), 2),
        "client_host": tag,
    }


def _window_rows(sample: pd.DataFrame, bundle: S.ServingBundle, *, tag: str,
                 ts_start: datetime, ts_end: datetime, drift: bool,
                 rng: np.random.Generator) -> list[dict]:
    span = (ts_end - ts_start).total_seconds()
    rows: list[dict] = []
    for i, (_, r) in enumerate(sample.iterrows()):
        ts = ts_start + timedelta(seconds=span * (i + 1) / (len(sample) + 1))
        for model in ("A", "B"):
            payload = payload_from_row(r, model)
            if drift:
                payload = inject_drift(payload, rng)
            resp = _score(bundle, model, payload)
            rows.append(_log_row(model, payload, resp, ts=ts, tag=tag, rng=rng))
    return rows


def clear_seed_rows(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM capstone_solution.prediction_log "
                    "WHERE client_host LIKE 'phase6-seed:%'")
        n = cur.rowcount
    conn.commit()
    return n


def seed_all(conn, *, n_baseline: int = 900, n_drift: int = 1400,
             seed: int = 20260904, serving_config=None) -> dict:
    """Replace any prior seed rows with a fresh baseline + drifted window.
    Returns a summary dict. ``n_drift > n_baseline`` gives the volume spike."""
    rng = np.random.default_rng(seed)
    bundle = S.load_serving_bundle(serving_config_path=serving_config)

    df = pd.read_parquet(S.PHASE2_PARQUET)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    test = M.time_split(df).test.reset_index(drop=True)

    base_idx = rng.choice(len(test), size=min(n_baseline, len(test)), replace=False)
    drift_idx = rng.choice(len(test), size=min(n_drift, len(test)), replace=True)

    now = datetime.now(timezone.utc)
    removed = clear_seed_rows(conn)

    baseline = _window_rows(
        test.iloc[base_idx], bundle, tag=BASELINE_TAG,
        ts_start=now - timedelta(days=14), ts_end=now - timedelta(days=7),
        drift=False, rng=rng)
    drifted = _window_rows(
        test.iloc[drift_idx], bundle, tag=DRIFT_TAG,
        ts_start=now - timedelta(days=7), ts_end=now - timedelta(hours=1),
        drift=True, rng=rng)

    with conn.cursor() as cur:
        cur.executemany(_INSERT, baseline)
        cur.executemany(_INSERT, drifted)
    conn.commit()

    summary = {
        "removed_prior": removed,
        "baseline_requests": len(baseline), "drift_requests": len(drifted),
        "baseline_visits": len(base_idx), "drift_visits": len(drift_idx),
        "model_version": bundle.manifest["model_version"],
    }
    log.info("seeded prediction_log: %s", summary)
    return summary
