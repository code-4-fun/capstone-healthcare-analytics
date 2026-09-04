"""drift_report / prediction_override / v_prediction_audit + append-only trigger."""
from __future__ import annotations

import uuid

import psycopg
import pytest

from capstone import monitoring as mon

from conftest import db_required

pytestmark = db_required


def test_tables_and_view_exist(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('capstone_solution.drift_report'), "
                    "to_regclass('capstone_solution.prediction_override'), "
                    "to_regclass('capstone_solution.v_prediction_audit')")
        assert all(cur.fetchone())


def test_drift_report_is_append_only(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM capstone_solution.drift_report ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    if row is None:
        pytest.skip("no drift_report rows yet")
    with conn.cursor() as cur, pytest.raises(psycopg.errors.RaiseException):
        cur.execute("UPDATE capstone_solution.drift_report SET value = 0 WHERE id = %s", (row[0],))
    conn.rollback()
    with conn.cursor() as cur, pytest.raises(psycopg.errors.RaiseException):
        cur.execute("DELETE FROM capstone_solution.drift_report WHERE id = %s", (row[0],))
    conn.rollback()


def test_prediction_log_blocks_update_but_allows_delete(conn):
    """A logged prediction is never edited; retention / cleanup still needs DELETE."""
    rid = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO capstone_solution.prediction_log
               (request_id, endpoint, model, model_version, feature_spec_version,
                serving_version, predicted_class, probabilities, latency_ms, client_host)
               VALUES (%s, '/x', 'B', '1.0.0', 2, '1.0.0', 'Paid', '{}'::jsonb, 1.0,
                       'phase6-test')""", (rid,))
        conn.commit()
    with conn.cursor() as cur, pytest.raises(psycopg.errors.RaiseException):
        cur.execute("UPDATE capstone_solution.prediction_log SET predicted_class = 'Rejected' "
                    "WHERE request_id = %s", (rid,))
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM capstone_solution.prediction_log WHERE request_id = %s", (rid,))
        conn.commit()


def test_audit_view_pairs_a_prediction_with_its_override(conn, seeded):
    # prediction_override is append-only too, so stage the row inside a
    # transaction, read it back through the view, and roll back.
    with conn.cursor() as cur:
        cur.execute("""SELECT request_id::text, predicted_class FROM capstone_solution.prediction_log
                       WHERE client_host = 'phase6-seed:drift' AND model = 'B' LIMIT 1""")
        rid, orig = cur.fetchone()
        cur.execute("""INSERT INTO capstone_solution.prediction_override
            (request_id, model, original_class, override_class, actor, reason)
            VALUES (%s, 'B', %s, 'Rejected', 'tester', 'unit test')""", (rid, orig))

        cur.execute("""SELECT predicted_class, override_class, override_actor, was_overridden
                       FROM capstone_solution.v_prediction_audit WHERE request_id = %s""", (rid,))
        pred_class, override_class, actor, was = cur.fetchone()
    conn.rollback()

    assert pred_class == orig
    assert override_class == "Rejected"
    assert actor == "tester"
    assert was is True


def test_alert_rules_are_the_single_source(conn):
    assert set(mon.ALERT_RULES) == {
        "feature_psi", "prediction_psi", "recall_costly_drop_pts", "gate_fail_rate"}
