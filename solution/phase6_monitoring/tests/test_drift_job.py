"""End-to-end drift run over seeded prediction_log windows (needs Postgres)."""
from __future__ import annotations

from capstone import monitoring as mon

from conftest import db_required

pytestmark = db_required


def test_baseline_window_is_quiet(conn, bundle, reference, seeded):
    res = mon.run_drift(conn, bundle, model="B", window="baseline", reference=reference)
    assert res.window_n > 0
    assert res.status == "OK"
    assert res.alerts.empty
    assert not res.feature["drifted"].any()


def test_drifted_window_alerts_on_the_injected_features(conn, bundle, reference, seeded):
    res = mon.run_drift(conn, bundle, model="B", window="drift", reference=reference)
    assert res.status == "ALERT"

    drifted = set(res.feature[res.feature["drifted"]]["feature"])
    assert {"billed_amount", "department", "insurance_provider"} <= drifted

    alert_kinds = set(res.alerts["metric_kind"])
    assert "feature_psi" in alert_kinds
    assert "perf_recall_costly" in alert_kinds     # Model B recall drop, actuals joined


def test_drift_run_persists_and_is_repeatable(conn, bundle, reference, seeded):
    r1 = mon.run_drift(conn, bundle, model="B", window="drift", reference=reference)
    n1 = mon.write_drift_report(conn, r1)
    r2 = mon.run_drift(conn, bundle, model="B", window="drift", reference=reference)
    n2 = mon.write_drift_report(conn, r2)

    assert n1 == n2 > 0
    assert r1.run_id != r2.run_id
    assert r1.status == r2.status == "ALERT"

    with conn.cursor() as cur:
        cur.execute("SELECT count(DISTINCT run_id) FROM capstone_solution.drift_report "
                    "WHERE run_id IN (%s, %s)", (r1.run_id, r2.run_id))
        assert cur.fetchone()[0] == 2


def test_model_a_runs_without_billing_features(conn, bundle, reference, seeded):
    res = mon.run_drift(conn, bundle, model="A", window="drift", reference=reference)
    assert set(res.feature["feature"]) & {"billed_amount", "risk_score"} == set()
