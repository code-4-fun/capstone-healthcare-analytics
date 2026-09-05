# Hospital Operations & Revenue Risk Intelligence Platform

**Live executive presentation: https://code-4-fun.github.io/capstone-healthcare-analytics/**

A capstone project: turn one year of a multi-specialty, multi-city hospital
network's visit and claims data into a trusted analytics layer, two
calibrated decision models, a served API, and a monitored, governed
production loop — one connected platform, not six disconnected exercises.

## The problem

The network was losing money and operational control to three coupled
failures: **patient-flow blindness** (no forward view of demand or acuity, so
staffing stays reactive), **revenue leakage** (claims rejected or
under-approved after the fact — 17.6% of billed value lost to denials, another
24.5% pending), and **no predictive layer** (decisions made on lagging
reports, not risk-scored signals). The [executive presentation
site](https://code-4-fun.github.io/capstone-healthcare-analytics/) walks
through the full story, in business terms, with every claim backed by a chart.

## Repository layout

```
capstone-healthcare-analytics/
├── objectives/          the assignment brief + raw data — given, never edited
│   ├── Capstone-Statement.pdf
│   └── data/             patients.csv, visits.csv, billing.csv
│                          (5,000 patients · 25,000 visits · 25,000 claims)
└── solution/             all solution work — start here to run anything
    ├── README.md          setup + how to run every phase
    ├── docs/PLAN.md        the full phased plan and exit criteria
    ├── docs/CAPSTONE_OUTCOMES.md   the brief's acceptance bar, verbatim
    ├── src/capstone/        shared package (DB, chart house style, per-phase
    │                        utilities) reused across every phase
    ├── phase1_sql_analytics/   SQL analytics layer (Postgres)
    ├── phase2_eda/             EDA & data quality
    ├── phase3_models/          two time-validated classifiers
    ├── phase4_eval/            evaluation, explainability, fairness
    ├── phase5_api/             FastAPI serving layer
    ├── phase6_monitoring/      drift detection & governance
    └── final_presentation/     the site above — architecture, results,
                                 business case, governance, in one page
```

Each `solution/phase<n>_*/` folder is self-contained: its own `README.md`,
its own runnable entrypoint (`run_phase<n>.py`, or a notebook it regenerates
and executes), and its own findings report (`PHASE<n>_FINDINGS.md`) with every
number backed by a chart. **Start with [`solution/README.md`](solution/README.md)**
for setup and how to run each phase end to end; **start with the
[executive presentation](https://code-4-fun.github.io/capstone-healthcare-analytics/)**
for the results without running anything.

## Status

All six phases are built, verified end to end, and reproducible from a clean
checkout — `docker compose up -d` (solution root) brings up a self-managed
Postgres, and each phase's own entrypoint rebuilds that phase's artefacts. See
[`solution/README.md`](solution/README.md#status) for the phase-by-phase table.
