"""Phase 1 orchestrator: build the SQL analytics layer end to end.

    uv run phase1_sql_analytics/run_phase1.py

Steps:
  1. 01_schema.sql        - create the capstone_solution schema
  2. 02_tables.sql        - typed, constrained core tables
  3. load_data.py         - direct typed load of the CSVs
  4. 03_indexes.sql       - indexing strategy + ANALYZE
  5. 04_views.sql         - business intelligence views
  6. 05_data_quality.sql  - automated data-quality report view
  7. charts               - one C-suite chart per finding (capstone.viz)
  8. export               - dump views to phase1_sql_analytics/output/*.csv and
                            write PHASE1_FINDINGS.md (findings + charts + tables)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capstone.db import SETTINGS, connect, engine  # noqa: E402
from load_data import load  # noqa: E402
import make_charts  # noqa: E402

HERE = Path(__file__).resolve().parent
SQL_DIR = HERE / "sql"
OUT_DIR = HERE / "output"

SQL_STEPS = [
    "01_schema.sql",
    "02_tables.sql",
]
SQL_STEPS_POST_LOAD = [
    "03_indexes.sql",
    "04_views.sql",
    "05_data_quality.sql",
]

EXPORT_VIEWS = [
    "v_department_performance",
    "v_insurance_provider_behavior",
    "v_revenue_realization_monthly",
    "v_patient_flow_monthly",
    "v_claim_rejection_analysis",
    "v_doctor_workload",
    "v_data_quality_report",
]


def run_sql_file(cur, path: Path) -> None:
    print(f"  - {path.name}")
    cur.execute(path.read_text())


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)

    print(f"[1] schema + tables  (db={SETTINGS.database}, schema={SETTINGS.schema})")
    with connect(search_path=False) as conn, conn.cursor() as cur:
        for name in SQL_STEPS:
            run_sql_file(cur, SQL_DIR / name)
        conn.commit()

    print("[2] direct typed load")
    load_summary = load()
    for table, s in load_summary.items():
        print(f"      {table:<10} loaded={s['loaded']:>6}  rejected={s['rejected']:>4}")

    print("[3] indexes + views + data quality")
    with connect() as conn, conn.cursor() as cur:
        for name in SQL_STEPS_POST_LOAD:
            run_sql_file(cur, SQL_DIR / name)
        conn.commit()

    print("[4] export view outputs -> phase1_sql_analytics/output/")
    exports: dict[str, pd.DataFrame] = {}
    eng = engine()
    for view in EXPORT_VIEWS:
        df = pd.read_sql(f"SELECT * FROM {view}", eng)
        df.to_csv(OUT_DIR / f"{view}.csv", index=False)
        exports[view] = df
        print(f"      {view:<32} {len(df):>5} rows")
    eng.dispose()

    print("[5] charts -> phase1_sql_analytics/output/charts/")
    charts = make_charts.build_all()

    _write_report(started, load_summary, exports, charts)
    print(f"\nDone. Findings: {HERE / 'PHASE1_FINDINGS.md'}")


def _img(charts: dict, key: str) -> str:
    path, caption = charts[key]
    rel = path.relative_to(HERE).as_posix()
    return f"![{caption}]({rel})\n\n*{caption}*"


def _write_report(started, load_summary, exports, chart_list) -> None:
    charts = {key: (path, caption) for key, path, caption in chart_list}
    dq = exports["v_data_quality_report"]
    dept = exports["v_department_performance"]
    prov = exports["v_insurance_provider_behavior"]
    rev = exports["v_revenue_realization_monthly"]

    total_billed = float(rev["billed_amount"].sum())
    total_collected = float(rev["collected_amount"].sum())
    total_leakage = float(rev["leakage_amount"].sum())
    total_pending = float(rev["pending_amount"].sum())
    pct = lambda v: f"{100 * v / total_billed:.1f}%"

    L: list[str] = []
    L.append("# Phase 1 — SQL Analytics Layer :: Findings\n")
    L.append("*Hospital Operations & Revenue Risk Intelligence Platform — Business Intelligence Foundation*\n")
    L.append(f"- Generated: {started.isoformat(timespec='seconds')}")
    L.append(f"- Source: `{SETTINGS.database}` / schema `{SETTINGS.schema}` — "
             f"{load_summary['visits']['loaded']:,} visits, "
             f"{load_summary['billing']['loaded']:,} claims, "
             f"{load_summary['patients']['loaded']:,} patients (calendar 2025)")
    L.append("- Every finding below is backed by a chart in `output/charts/`; supporting "
             "tables are in the appendix.\n")

    L.append("## Executive summary\n")
    L.append(f"1. **Revenue realization is {pct(total_collected)}** — of "
             f"{total_billed:,.0f} billed, only {total_collected:,.0f} is collected. "
             f"{total_leakage:,.0f} ({pct(total_leakage)}) is lost to adjudicated denials "
             f"and {total_pending:,.0f} ({pct(total_pending)}) is stuck pending.")
    L.append("2. **The leakage is structural, not localised** — realization is ~58% in "
             "every department, every insurer and every month. There is no single "
             "counterparty to renegotiate or quarter to blame.")
    L.append("3. **Denial drivers are not the obvious dimensions** — rejection rate is "
             "flat (~15%) across department, provider, visit type and risk band, but "
             "peaks at 22.7% in the mid-value (15k–30k) billed band and is *lower* for "
             "the largest claims.")
    L.append("4. **Two source fields cannot be trusted as a pair** — 95% of Pending "
             "claims already carry an `approved_amount`; 817 Paid claims carry none.")
    L.append("5. **Timelines are unreliable** — ~49% of records have billing before the "
             "visit or the visit before patient registration. Phase 3 time-based "
             "validation must key on `visit_date` only.\n")

    L.append("---\n\n## 1. Revenue realization\n")
    L.append(_img(charts, "revenue_waterfall"))
    L.append("\n" + _img(charts, "realization_trend"))
    L.append("\n" + _img(charts, "department_billed_collected"))

    L.append("\n---\n\n## 2. Insurance provider behaviour\n")
    L.append(_img(charts, "provider_claim_mix"))

    L.append("\n---\n\n## 3. Where claim rejections concentrate\n")
    L.append(_img(charts, "rejection_by_billed_band"))
    L.append("\n" + _img(charts, "rejection_flat_across_dims"))

    L.append("\n---\n\n## 4. Patient flow\n")
    L.append(_img(charts, "patient_flow_monthly"))

    L.append("\n---\n\n## 5. Data quality\n")
    L.append(_img(charts, "data_quality"))
    L.append("\n" + _img(charts, "status_vs_approved"))
    L.append("\n" + _img(charts, "distribution_floors"))
    errors = dq[(dq["severity"] == "ERROR") & (dq["records_flagged"] > 0)]
    L.append(f"\n**{len(errors)} ERROR-level check(s) with flagged rows.** "
             "Full check list in the appendix; rules live in `sql/05_data_quality.sql`.")

    L.append("\n---\n\n## Appendix — supporting tables\n")
    L.append("### Load summary\n")
    L.append("| Table | Read | Loaded | Rejected |")
    L.append("|---|--:|--:|--:|")
    for t, s in load_summary.items():
        L.append(f"| {t} | {s['read']:,} | {s['loaded']:,} | {s['rejected']} |")
    L.append("\n### Department performance\n")
    L.append(dept.to_markdown(index=False))
    L.append("\n### Insurance provider behaviour\n")
    L.append(prov.to_markdown(index=False))
    L.append("\n### Data quality report\n")
    L.append(dq.to_markdown(index=False))

    (HERE / "PHASE1_FINDINGS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
