# Hospital Operations & Revenue Risk Intelligence Platform

Capstone solution — an end-to-end healthcare analytics + AI system: a trusted
SQL analytics layer, EDA & data-quality work, two classification models
(visit risk, claim outcome), a FastAPI serving layer, and monitoring/governance.

Full phased plan: **[`docs/PLAN.md`](docs/PLAN.md)**.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | SQL analytics layer (Postgres) | ✅ built — `phase1_sql_analytics/` |
| 2 | EDA & data quality (Python) | planned |
| 3 | Model development (classification) | planned |
| 4 | Evaluation & explainability | planned |
| 5 | Deployment & API (FastAPI) | planned |
| 6 | Monitoring, drift & governance | planned |
| — | Executive presentation | planned |

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and the Docker Postgres container
(`capstone-project-postgres`, port 5432, trust auth).

```bash
uv sync
cp .env.example .env        # defaults already match the docker Postgres
docker start capstone-project-postgres
```

## Run Phase 1

```bash
uv run python phase1_sql_analytics/run_phase1.py
```

Builds schema `capstone_solution` in database `capstone_hospital_analytics`:
typed constrained tables → validated load → indexes → 10 BI views →
data-quality report → CSV exports → 10 C-suite charts →
`phase1_sql_analytics/PHASE1_FINDINGS.md` (findings by theme, each with its
chart; supporting tables in an appendix).

See [`phase1_sql_analytics/README.md`](phase1_sql_analytics/README.md).

## Reporting standard

Every quantitative finding, in every phase, is backed by a chart built with the
shared house style in `src/capstone/viz.py` (form chosen by the data's job,
CVD-validated palette, takeaway titles, one y-axis, direct labels, table view
alongside). See [`CLAUDE.md`](CLAUDE.md).

## Layout

```
CLAUDE.md                 agent instructions (uv, DB, reporting standard)
src/capstone/db.py        shared Postgres connection / SQLAlchemy engine (reads .env)
src/capstone/viz.py       shared charting house style (all phases)
docs/PLAN.md              phased build plan + Phase 1 findings
phase1_sql_analytics/     Phase 1 (built)
phase2_eda/ ...           later phases
```

## Configuration

All connection detail lives in `.env` (see `.env.example`):
`PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE`, `CAPSTONE_SCHEMA`,
`CAPSTONE_DATA_DIR`. Code reads it via `capstone.db.SETTINGS`.
