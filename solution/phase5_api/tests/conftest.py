"""Shared fixtures for the Phase 5 API test suite.

Ensures the Phase 3 artefacts and the serving config exist, exposes a
lifespan-managed ``TestClient``, and cleans up any ``prediction_log`` rows the
tests write.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PHASE5_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = PHASE5_DIR.parent
if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

from capstone import serving as S  # noqa: E402


def _ensure_artefacts() -> None:
    manifest = S.PHASE3_MODELS_DIR / "training_manifest.json"
    if not manifest.exists():
        subprocess.run([sys.executable, str(SOLUTION_DIR / "phase3_models" / "run_phase3.py")], check=True)
    cfg = PHASE5_DIR / "serving_config.json"
    if not cfg.exists():
        S.write_serving_config(cfg)


@pytest.fixture(scope="session")
def bundle():
    _ensure_artefacts()
    return S.load_serving_bundle()


@pytest.fixture(scope="session")
def _db_available() -> bool:
    from app import predictions_log as plog

    return plog.db_reachable()


@pytest.fixture()
def client(bundle):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app import predictions_log as plog

    hi_before = _max_log_id()
    with TestClient(create_app()) as c:
        yield c
    # tidy: drop only the rows this test appended
    if hi_before is not None:
        try:
            from capstone.db import connect

            with connect(autocommit=True) as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM capstone_solution.prediction_log WHERE id > %s", (hi_before,))
        except Exception:  # noqa: BLE001
            pass


def _max_log_id():
    try:
        from capstone.db import connect

        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(id), 0) FROM capstone_solution.prediction_log")
            return cur.fetchone()[0]
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="session")
def golden():
    g = PHASE5_DIR / "tests" / "golden"
    return {
        "A": json.loads((g / "model_a_golden.json").read_text()),
        "B": json.loads((g / "model_b_golden.json").read_text()),
    }


def valid_claim_payload(**overrides):
    p = {
        "visit_date": "2025-12-01",
        "department": "Cardiology",
        "visit_type": "OPD",
        "age": 54,
        "gender": "M",
        "city": "Pune",
        "insurance_provider": "CareOne",
        "chronic_flag": True,
        "billed_amount": 22000.0,
        "length_of_stay_hours": 12.5,
        "risk_score": "Medium",
    }
    p.update(overrides)
    return p


def valid_visit_payload(**overrides):
    p = {
        "visit_date": "2025-12-01",
        "department": "ICU",
        "visit_type": "ICU",
        "age": 70,
        "gender": "F",
        "city": "Mumbai",
        "insurance_provider": "SecureLife",
        "chronic_flag": False,
    }
    p.update(overrides)
    return p
