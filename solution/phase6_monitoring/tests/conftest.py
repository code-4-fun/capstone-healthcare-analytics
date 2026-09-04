"""Shared fixtures for the Phase 6 monitoring test suite."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PHASE6_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = PHASE6_DIR.parent
if str(PHASE6_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE6_DIR))

from capstone import monitoring as mon  # noqa: E402
from capstone import serving as S  # noqa: E402


def _ensure_artefacts() -> None:
    if not (S.PHASE3_MODELS_DIR / "training_manifest.json").exists():
        subprocess.run([sys.executable, str(SOLUTION_DIR / "phase3_models" / "run_phase3.py")], check=True)
    if not (SOLUTION_DIR / "phase5_api" / "serving_config.json").exists():
        S.write_serving_config(SOLUTION_DIR / "phase5_api" / "serving_config.json")


@pytest.fixture(scope="session")
def bundle():
    _ensure_artefacts()
    return S.load_serving_bundle()


@pytest.fixture(scope="session")
def reference():
    _ensure_artefacts()
    return mon.reference_frame()


def _db_reachable() -> bool:
    try:
        from capstone.db import connect

        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False


db_required = pytest.mark.skipif(not _db_reachable(), reason="Postgres unreachable")


@pytest.fixture(scope="session")
def conn():
    from capstone.db import connect

    with connect() as c:
        mon.ensure_tables(c)
        yield c


@pytest.fixture(scope="session")
def seeded(conn):
    """Seed a small baseline + drift window once for the DB-backed tests."""
    from phase6_monitoring import seed_traffic

    summary = seed_traffic.seed_all(conn, n_baseline=150, n_drift=220)
    yield summary
    seed_traffic.clear_seed_rows(conn)


def sample_reference_rows(model: str, n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ref = mon.reference_frame()
    idx = rng.choice(len(ref), size=min(n, len(ref)), replace=False)
    return ref.iloc[idx].reset_index(drop=True)
