"""Phase 6 :: the scheduler sidecar.

Runs inside ``docker compose`` alongside Postgres, the API and Grafana. On
start it waits for the database, applies the DDL, and seeds the two demo
traffic windows if the log is empty; then it runs the drift job on a fixed
interval so ``drift_report`` (and the Grafana dashboard) keep filling.

Config (env):
  * ``DRIFT_INTERVAL_SECONDS`` - seconds between drift runs (default 21600 = 6h;
    the compose file sets a short value for the demo).
  * ``DRIFT_WINDOW``           - window spec passed to the job (default ``drift``).
  * ``SEED_ON_START``          - ``"true"`` (default) to seed when the log is empty.
"""
from __future__ import annotations

import logging
import os
import time

from capstone import monitoring as mon
from capstone.db import connect

from phase6_monitoring import drift_job, seed_traffic

log = logging.getLogger("capstone.phase6.scheduler")

INTERVAL = int(os.environ.get("DRIFT_INTERVAL_SECONDS", "21600"))
WINDOW = os.environ.get("DRIFT_WINDOW", "drift")
SEED_ON_START = os.environ.get("SEED_ON_START", "true").lower() == "true"


def _wait_for_db(attempts: int = 60, delay: float = 2.0) -> None:
    for i in range(attempts):
        try:
            with connect(autocommit=True) as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            log.info("database reachable")
            return
        except Exception as exc:  # noqa: BLE001
            log.info("waiting for database (%d/%d): %s", i + 1, attempts, exc)
            time.sleep(delay)
    raise SystemExit("database never became reachable")


def _log_is_empty(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM capstone_solution.prediction_log")
        return cur.fetchone()[0] == 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _wait_for_db()

    with connect() as conn:
        mon.ensure_tables(conn)
        if SEED_ON_START and _log_is_empty(conn):
            log.info("prediction_log empty - seeding baseline + drift windows")
            seed_traffic.seed_all(conn)

    log.info("scheduler up - drift job every %ds on window %r", INTERVAL, WINDOW)
    while True:
        try:
            drift_job.run(models=["A", "B"], window=WINDOW)
        except Exception:  # noqa: BLE001 - a scheduler must not die on one bad run
            log.exception("drift run failed; will retry next interval")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
