# Phase 1 — SQL Analytics Layer

Transforms the three raw CSVs into a trusted, indexed analytics layer in the
Postgres schema **`capstone_solution`** (inside database
`capstone_hospital_analytics`).

## Run it

```bash
# from solution/
cp .env.example .env          # first time only; defaults match the docker Postgres
docker compose up -d          # starts + bootstraps Postgres (see ../docker-compose.yml)
uv sync
uv run python phase1_sql_analytics/run_phase1.py
```

The runner is idempotent — it drops/recreates tables, reloads, and rebuilds
every view. View extracts land in `output/*.csv`, charts in `output/charts/`,
and the C-suite write-up in `PHASE1_FINDINGS.md` (findings by theme, each with
its chart embedded; supporting tables in an appendix).

`docker compose up -d` already applies this phase's DDL via the `bootstrap`
service (`bootstrap_db.py` — the same `sql/01_schema.sql`.. `05_data_quality.sql`
+ `load_data.py` steps, without the reporting pipeline), so a fresh clone needs
no externally-set-up Postgres to get here.

## What gets built

| Step | File | Result |
|---|---|---|
| 1 | `sql/01_schema.sql` | `capstone_solution` schema |
| 2 | `sql/02_tables.sql` | typed core tables + `load_rejects` |
| 3 | `load_data.py` | direct typed load with row validation |
| 4 | `sql/03_indexes.sql` | indexes for the analytics access patterns |
| 5 | `sql/04_views.sql` | 10 business-intelligence views |
| 6 | `sql/05_data_quality.sql` | `v_data_quality_report` (12 checks) |
| 7 | `make_charts.py` | 10 C-suite charts → `output/charts/` (house style: `capstone.viz`) |
| — | `sql/06_business_queries.sql` | 10 leadership questions (run ad hoc) |

## Data model

```
patients (patient_id PK)
   │  1
   │  ∞
visits (visit_id PK, patient_id FK)
   │  1
   │  1
billing (bill_id PK, visit_id FK UNIQUE)
```

Integrity enforced in-database: PKs everywhere, both FKs, `UNIQUE(visit_id)` on
billing (1:1), and CHECK constraints on `age`, `gender`, `chronic_flag`,
`visit_type`, `risk_score`, `claim_status`, all amounts (`>= 0`),
`approved_amount <= billed_amount`.

## Views

| View | Purpose |
|---|---|
| `v_visit_billing` | **Spine** — one row per visit, visit+patient+billing enriched with derived fields (age band, `collected_amount`, `leakage_amount`, `pending_amount`, `billing_lag_days`, status flags). Downstream phases read this. |
| `v_department_performance` | Per-department volume, LOS, acuity mix, billed/collected/leakage, realization & rejection rates |
| `v_patient_flow_daily` | Daily volume, ER/OPD/ICU split, high-risk count, avg LOS |
| `v_patient_flow_monthly` | Monthly flow & revenue by department |
| `v_insurance_provider_behavior` | Per-insurer outcome mix, approved ratio, payment speed (avg & p90), realization |
| `v_revenue_realization_monthly` | Monthly revenue waterfall: billed → approved → collected, pending, leakage |
| `v_high_risk_visits` | All High-risk visits with patient + billing context |
| `v_claim_rejection_analysis` | Rejection rate by department / provider / visit_type / risk band / billed band |
| `v_doctor_workload` | Per-doctor load, acuity mix, rejection rate |
| `v_data_quality_report` | 12 automated checks, severity ERROR/WARN/INFO |

## Headline results (this run)

- **Revenue:** 521.8M billed → 302.3M collected (**57.9%** realization);
  **91.8M** denial leakage (17.6%); **127.7M** pending (24.5%).
- **Rejections** peak in the 15k–30k billed band (**22.7%**), non-monotonic in
  amount, and are nearly flat across department, provider and risk band —
  denial drivers are not the obvious ones.
- **Data quality:** 1 ERROR check fires — 817 Paid claims (5.5%) with no
  `approved_amount`. 95% of Pending claims already carry an `approved_amount`.
  LOS floors at 0.5h; `billed_amount` floors at 500. ~49% of rows have
  billing-before-visit or visit-before-registration — **dates are approximate**.

These feed directly into the Phase 2 EDA and the Phase 3 leakage register
(see `../docs/PLAN.md`).
