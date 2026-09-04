"""Phase 1 :: DB-only bootstrap (no charts, no findings report).

Applies exactly the DDL + load steps documented in the Phase 1 outcome:

  1. sql/01_schema.sql        - create the capstone_solution schema
  2. sql/02_tables.sql        - typed, constrained core tables
  3. load_data.py             - direct typed load of the CSVs
  4. sql/03_indexes.sql       - indexing strategy + ANALYZE
  5. sql/04_views.sql         - the 10 business-intelligence views
  6. sql/05_data_quality.sql  - v_data_quality_report (12 checks)

This is the fast path the `bootstrap` service in `docker-compose.yml` runs
against a freshly-started, empty Postgres container so the analytics layer is
ready the moment the container is healthy - Phase 2 reads `v_visit_billing`
straight away, no manual step in between.

    uv run python phase1_sql_analytics/bootstrap_db.py

Idempotent, same as `run_phase1.py` (which this intentionally does not call -
it is the interactive, full pipeline: same DDL + load, plus CSV exports,
charts and `PHASE1_FINDINGS.md`). Run that instead when you want the reviewed
findings, not just a ready database.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capstone.db import SETTINGS, connect  # noqa: E402
from load_data import load  # noqa: E402

HERE = Path(__file__).resolve().parent
SQL_DIR = HERE / "sql"

SQL_STEPS = [
    "01_schema.sql",
    "02_tables.sql",
]
SQL_STEPS_POST_LOAD = [
    "03_indexes.sql",
    "04_views.sql",
    "05_data_quality.sql",
]


def run_sql_file(cur, path: Path) -> None:
    print(f"  - {path.name}")
    cur.execute(path.read_text())


def main() -> None:
    print(f"[1] schema + tables  (db={SETTINGS.database}, schema={SETTINGS.schema})")
    with connect(search_path=False) as conn, conn.cursor() as cur:
        for name in SQL_STEPS:
            run_sql_file(cur, SQL_DIR / name)
        conn.commit()

    print(f"[2] direct typed load  (source={SETTINGS.data_dir})")
    load_summary = load()
    for table, s in load_summary.items():
        print(f"      {table:<10} loaded={s['loaded']:>6}  rejected={s['rejected']:>4}")

    print("[3] indexes + views + data quality")
    with connect() as conn, conn.cursor() as cur:
        for name in SQL_STEPS_POST_LOAD:
            run_sql_file(cur, SQL_DIR / name)
        conn.commit()

    print(f"\nDB bootstrap complete - '{SETTINGS.schema}' is ready for Phase 2/3.")
    print("For charts, CSV exports and PHASE1_FINDINGS.md, run run_phase1.py instead.")


if __name__ == "__main__":
    main()
