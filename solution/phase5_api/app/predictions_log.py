"""Prediction-log persistence (Postgres ``capstone_solution.prediction_log``).

Writes are **best-effort**: a prediction is still returned to the caller if the
database is unreachable - the failure is logged and surfaced on ``/health``
(``db_reachable: false``), never raised into the request path. The table is the
Phase 6 drift baseline and the governance audit trail.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from capstone.db import connect

log = logging.getLogger("capstone.api.prediction_log")

_DDL_PATH = Path(__file__).resolve().parent.parent / "sql" / "prediction_log.sql"

_INSERT = """
INSERT INTO capstone_solution.prediction_log
    (request_id, endpoint, model, model_version, feature_spec_version, serving_version,
     operating_threshold, predicted_class, probabilities, decision, defaults_applied,
     request_payload, latency_ms, client_host)
VALUES
    (%(request_id)s, %(endpoint)s, %(model)s, %(model_version)s, %(feature_spec_version)s,
     %(serving_version)s, %(operating_threshold)s, %(predicted_class)s, %(probabilities)s,
     %(decision)s, %(defaults_applied)s, %(request_payload)s, %(latency_ms)s, %(client_host)s)
"""


def ensure_table() -> bool:
    """Apply the idempotent DDL. Returns True on success, False if the DB is down."""
    try:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(_DDL_PATH.read_text())
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("could not ensure prediction_log table: %s", exc)
        return False


def db_reachable() -> bool:
    try:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False


def log_prediction(*, endpoint: str, response: dict[str, Any], serving_version: str,
                   request_payload: dict[str, Any] | None, client_host: str | None) -> bool:
    """Append one prediction. Best-effort - returns False (and logs) on any error."""
    decision = response.get("decision")
    params = {
        "request_id": response["request_id"],
        "endpoint": endpoint,
        "model": response["model"],
        "model_version": response["model_version"],
        "feature_spec_version": response["feature_spec_version"],
        "serving_version": serving_version,
        "operating_threshold": decision["threshold"] if decision else None,
        "predicted_class": response["predicted_class"],
        "probabilities": json.dumps(response["probabilities"]),
        "decision": json.dumps(decision) if decision else None,
        "defaults_applied": json.dumps(response["defaults_applied"]),
        "request_payload": json.dumps(request_payload, default=str) if request_payload is not None else None,
        "latency_ms": response["latency_ms"],
        "client_host": client_host,
    }
    try:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(_INSERT, params)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("prediction_log insert failed (%s): %s", response.get("request_id"), exc)
        return False
