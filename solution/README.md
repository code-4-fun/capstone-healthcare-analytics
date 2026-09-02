# Hospital Operations & Revenue Risk Intelligence Platform

Capstone solution — an end-to-end healthcare analytics + AI system: a trusted
SQL analytics layer, EDA & data-quality work, two classification models
(visit risk, claim outcome), a FastAPI serving layer, and monitoring/governance.

Full phased plan: **[`docs/PLAN.md`](docs/PLAN.md)**.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | SQL analytics layer (Postgres) | ✅ built — `phase1_sql_analytics/` |
| 2 | EDA & data quality (Python) | ✅ built — `phase2_eda/` |
| 3 | Model development (classification) | ✅ built — `phase3_models/` |
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

## Run Phase 2

```bash
uv run python phase2_eda/run_phase2.py     # regenerates + executes phase2_eda/phase2.ipynb
```

From Phase 2 on the deliverable is a **Jupyter notebook** (`phase2_eda/phase2.ipynb`)
that runs top-to-bottom and renders its charts and tables inline; reusable logic
lives in `src/capstone/` (`eda`, `features`, `data_quality`). The notebook reads
`v_visit_billing` and produces the modelling-readiness artefacts: profiling +
business-analysis CSVs, `feature_spec.yaml` (feature catalogue + per-model leakage
register), `output/feature_frame.parquet` (as-of feature matrix), a leakage
self-check, 19 house-style charts, and `phase2_eda/PHASE2_FINDINGS.md`.
`run_phase2.py` is idempotent and rebuilds the Phase 1 layer first if it is
unreachable.

Headline: Model A (visit risk) has no learnable signal in the available data;
Model B (claim outcome) is driven almost entirely by `billed_amount` (rejections
peak non-monotonically in the 15k–30k band). `visit_date` is the only trusted
temporal key. See [`phase2_eda/README.md`](phase2_eda/README.md).

## Run Phase 3

```bash
uv run python phase3_models/run_phase3.py   # regenerates + executes phase3_models/phase3.ipynb
```

Same notebook-first pattern: `phase3.ipynb` is the deliverable, reusable logic
lives in `src/capstone/modeling.py`. Reads `phase2_eda/output/feature_frame.parquet`,
splits it on `visit_date` (9 months train / 1 validate / 2 test), and trains two
models against four candidates each (majority baseline, domain simple-rule
baseline, logistic regression, gradient-boosted trees), then calibrates the
selected model (sigmoid, on the validation month) and persists it with a
versioned `training_manifest.json`. Outputs: `models/`, test predictions +
metrics in `output/`, 8 house-style charts, and `phase3_models/PHASE3_FINDINGS.md`.

Headline: **Model A ships as a calibrated base-rate monitor** — no candidate
beats chance on balanced accuracy (Phase 2 predicted this). **Model B (gradient
boosting) works for pre-submission triage** — it recovers 66% of claims that go
on to be rejected (vs 0% for a majority classifier) and beats the majority and
simple-rule baselines on balanced accuracy and macro-F1. Neither beats the
majority baseline on raw accuracy — the wrong bar for a skewed target. See
[`phase3_models/README.md`](phase3_models/README.md).

## Reporting standard

Every quantitative finding, in every phase, is backed by a chart built with the
shared house style in `src/capstone/viz.py` (form chosen by the data's job,
CVD-validated palette, takeaway titles, one y-axis, direct labels, table view
alongside). See [`CLAUDE.md`](CLAUDE.md).

## Layout

```
CLAUDE.md                    agent instructions (uv, DB, reporting standard, notebook format)
src/capstone/db.py           shared Postgres connection / SQLAlchemy engine (reads .env)
src/capstone/viz.py          shared charting house style (all phases)
src/capstone/eda.py          Phase 2 profiling + business analyses
src/capstone/features.py     as-of feature builder + FEATURE_SPEC + leakage register
src/capstone/data_quality.py reusable data-quality validators
src/capstone/modeling.py     Phase 3 split / pipelines / training / calibration / persistence
docs/PLAN.md                 phased build plan + Phase 1 findings
phase1_sql_analytics/        Phase 1 (built, script-based)
phase2_eda/                  Phase 2 (built, notebook: phase2.ipynb)
phase3_models/               Phase 3 (built, notebook: phase3.ipynb)
phase4_eval/ ...             later phases
```

## Configuration

All connection detail lives in `.env` (see `.env.example`):
`PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE`, `CAPSTONE_SCHEMA`,
`CAPSTONE_DATA_DIR`. Code reads it via `capstone.db.SETTINGS`.
