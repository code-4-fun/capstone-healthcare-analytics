# Hospital Operations & Revenue Risk Intelligence Platform — Phased Build Plan

## 1. The business problem

A multi-specialty, multi-city hospital network is losing money and operational
control to three coupled failures:

1. **Patient-flow blindness** — no forward view of where demand, acuity and
   length-of-stay pressure will land, so staffing and bed planning are reactive.
2. **Revenue leakage** — insurance claims are rejected or under-approved after
   the fact; ~18% of billed value is lost to denials and another ~25% sits in
   pending limbo (measured in Phase 1).
3. **No predictive layer** — decisions are made on lagging reports, not on
   risk-scored, forward-looking signals.

The platform delivers a single, trusted data foundation plus two decision
models (visit risk, claim outcome), served through an API and governed for
healthcare use.

## 2. Source data

Three CSVs in `objectives/data/` (one calendar year, 2025-01-20 → 2026-01-20):

| File | Grain | Rows | Key columns |
|---|---|--:|---|
| `patients.csv` | one patient | 5,000 | `patient_id`, age, gender, city, `insurance_provider`, `chronic_flag`, `registration_date` |
| `visits.csv` | one visit | 25,000 | `visit_id`, `patient_id`→patient, `visit_date`, department, `visit_type` (ER/OPD/ICU), `length_of_stay_hours`, `risk_score` (Low/Med/High), `doctor_id` |
| `billing.csv` | one claim (1:1 with visit) | 25,000 | `bill_id`, `visit_id`→visit, `billed_amount`, `approved_amount`, `claim_status` (Paid/Pending/Rejected), `payment_days`, `billing_date` |

Relationships: `visits.patient_id → patients`, `billing.visit_id → visits`
(1:1). Referential integrity is clean in the raw files; the data-quality
issues are in **values and timelines**, not links (see Phase 1 findings).

## 2a. Key outcomes — the acceptance bar

`CAPSTONE_OUTCOMES.md` (in this folder) records, verbatim from the brief, the
**Key Learning Outcomes** and the **"How to approach this capstone"** notes. The
platform is judged against them: end-to-end connected phases, models judged on
business impact, data-quality and leakage risks handled, predictions expressed
as operational/financial decisions, MLOps deployment + governance, and one
cohesive platform proposable to hospital leadership. Every phase is framed
against that file, and hawk-eye checks the phase's work against it before OKAY.

## 3. Guiding principles

- **Each phase produces an artefact the next phase consumes.** Phase 1's
  `v_visit_billing` view is the modelling spine; Phase 2's feature definitions
  become Phase 3's feature pipeline; Phase 3's models become Phase 5's API;
  Phase 5's prediction log becomes Phase 6's drift baseline.
- **Business metric first, technical metric second.** Recall on High-risk
  visits and on Rejected claims are the headline numbers.
- **Every finding is backed by a chart.** This is a C-suite deliverable; a
  quantitative claim in prose without an appropriate visual is not a finished
  finding. Charts use the shared house style in `capstone.viz` (form chosen by
  the data's job, CVD-validated palette, takeaway titles, one y-axis, direct
  labels, table view alongside). Full rules in `CLAUDE.md`.
- **Leakage-aware.** `registration_date` is not a reliable temporal anchor
  (48% of visits precede it) and `billing_date` ordering is noisy — time-based
  validation in Phase 3 uses `visit_date`, and no post-outcome field
  (`approved_amount`, `payment_days`, `claim_status`) may feed Model A or the
  pre-submission Model B.
- **Everything reproducible.** One command rebuilds each phase; config lives in
  `.env`; dependencies are locked by `uv`.
- **Notebooks are the phase artefact from Phase 2 on.** Each phase from Phase 2
  onward delivers its main artefact and outputs as a Jupyter notebook that runs
  top-to-bottom and renders its charts/tables inline. Reusable logic lives in
  `src/capstone/` (adding utility modules there is expected); the notebook stays
  thin and imports from `capstone`. Phase 1 keeps its script form.

## 4. Phases

### Phase 1 — SQL Analytics Layer  *(BUILT — see `phase1_sql_analytics/`)*

**Goal:** turn the raw CSVs into a trusted, queryable analytics layer in
Postgres schema `capstone_solution`.

**Deliverables**
- Typed, constrained core tables (`patients`, `visits`, `billing`) — PKs, FKs,
  CHECK constraints on every known domain, 1:1 billing↔visit enforced.
- Direct typed load (`load_data.py`) with row-level validation; rejects parked
  in `load_rejects` (JSONB + reason).
- Indexing strategy for the analytics access patterns (time, department,
  provider, risk band, claim status, FK joins).
- 10 business-intelligence views: `v_visit_billing` (spine),
  `v_department_performance`, `v_patient_flow_daily` / `_monthly`,
  `v_insurance_provider_behavior`, `v_revenue_realization_monthly`,
  `v_high_risk_visits`, `v_claim_rejection_analysis`, `v_doctor_workload`.
- `v_data_quality_report` — 12 automated checks (ERROR/WARN/INFO).
- `sql/06_business_queries.sql` — 10 leadership questions answered off the views.
- `make_charts.py` — 10 C-suite charts (one per finding) via the `capstone.viz`
  house style, written to `output/charts/`.
- `run_phase1.py` — one-command rebuild + CSV exports + charts +
  `PHASE1_FINDINGS.md` (findings organised by theme, each with its chart embedded
  and supporting tables in an appendix).

**Key findings handed to Phase 2**
- Revenue: 521.8M billed → 302.3M collected (57.9% realization), 91.8M denial
  leakage (17.6%), 127.7M pending (24.5%).
- Rejection rate is highest in the 15k–30k billed band (22.7%) and *not*
  monotonic in amount; barely varies by department, provider, or risk band.
- 95% of Pending claims already carry an `approved_amount`; 5.5% of Paid claims
  have none — `claim_status` and `approved_amount` are partly decoupled.
- `length_of_stay_hours` has a hard floor at 0.5h (1.2%); `billed_amount` floors
  at 500 (1.0%).
- Temporal fields are internally inconsistent (~49% billing-before-visit,
  ~49% visit-before-registration) → treat dates as approximate.

**Exit criteria:** all views materialise, `run_phase1.py` is idempotent, DQ
report reviewed, 0 unexplained ERROR-level checks.

---

### Phase 2 — Exploratory Data Analysis & Data Quality  *(Python)*

**Goal:** understand operations and financial dynamics; lock down feature
definitions and leakage rules before modelling.

**Work**
- Pull the joined dataset from `v_visit_billing` into pandas; profile
  distributions, missingness, outliers per column.
- Formalise the Phase 1 DQ findings: quantify the 0.5h / 500-amount floors,
  the status↔approved_amount decoupling, the temporal inconsistency; decide
  imputation / exclusion rules.
- Business analyses: patient-flow seasonality, LOS drivers, department acuity
  mix, provider payment behaviour, denial cohort analysis, revenue waterfall.
- **Feature engineering catalogue** (business-driven, leakage-safe):
  - patient history as-of `visit_date` (prior visit count, prior high-risk
    count, prior rejection count, days since last visit)
  - visit context (department, visit_type, doctor load, chronic_flag, age band)
  - billing context available pre-submission (billed_amount band, provider,
    provider historical rejection rate as-of date)
  - seasonality (month, day-of-week, week-of-year)
- Explicit **leakage register**: fields forbidden per model.

**Deliverables:** `phase2_eda/` — notebook(s) + `PHASE2_FINDINGS.md` (every
finding backed by a `capstone.viz` chart) + `feature_spec.yaml` +
`data_quality_rules.py` (reusable validators).

**Exit criteria:** every feature has a definition, a source, an as-of rule and a
leakage verdict; target distributions and class balance documented.

---

### Phase 3 — Model Development (Classification)  *(BUILT — see `phase3_models/`)*

**Goal:** two calibrated classifiers, time-validated.

**Model A — Visit Risk (Low / Medium / High)**
- Target: `visits.risk_score`. Features: operational + clinical + patient
  history only — **no billing outcome fields**.
- Use: staffing, bed planning, patient prioritisation.

**Model B — Claim Outcome (Paid / Pending / Rejected)**
- Target: `billing.claim_status`. Features: everything knowable **before claim
  submission** — billed amount, department, visit_type, LOS, risk_score,
  provider, patient/provider history. **No `approved_amount`, `payment_days`.**
- Use: pre-submission triage to cut denial leakage.

**Shared method**
- Time-based split on `visit_date` (e.g. first 9 months train, next 1 validate,
  last 2 test); no random shuffle.
- Pipeline = sklearn `ColumnTransformer` + model; candidates: regularised
  logistic regression (baseline, interpretable), gradient-boosted trees
  (LightGBM/XGBoost). Class weighting / threshold tuning for the costly class.
- Probability calibration (isotonic / Platt) — needed for Phase 5 thresholds.
- Persist: `model_a.joblib`, `model_b.joblib` + `feature_pipeline` + a
  `training_manifest.json` (data window, feature list, versions, metrics).

**Deliverables:** `phase3_models/` — training scripts, `models/` artefacts,
CV/holdout metric tables.

**Exit criteria:** both models beat a majority-class and a simple-rule baseline
on the time-held-out test set; no leakage (verified by ablation); artefacts
reload and predict from a clean process.

---

### Phase 4 — Model Evaluation & Explainability  *(BUILT — see `phase4_eval/`)*

**Goal:** prove the models are interpretable, reliable and safe.

**Work**
- Technical metrics: precision / recall / F1 per class, confusion matrices,
  ROC & PR curves, calibration plots.
- **Business metrics:** recall on High-risk visits; recall on Rejected claims;
  projected leakage recovered at the chosen operating threshold; alert volume.
- Explainability: global (permutation importance, SHAP summary) and local
  (SHAP force plots for sample decisions).
- Fairness: metric parity across gender, age band, city, insurance provider;
  document disparities and mitigations.
- **Model cards** for A and B: intent, data, metrics, thresholds, limitations,
  ethical considerations, retraining triggers.

**Deliverables:** `phase4_eval/` — `PHASE4_FINDINGS.md` with all metric,
calibration, explainability and fairness charts in the `capstone.viz` style,
`model_card_A.md`, `model_card_B.md`.

**Exit criteria:** business-critical recall targets met or explicitly
signed-off; fairness gaps quantified; model cards complete.

---

### Phase 5 — Deployment & API Integration (MLOps)  *(FastAPI)*

**Goal:** production-ready services, not more accuracy.

**Work**
- FastAPI app, one endpoint per model: `POST /predict/visit-risk`,
  `POST /predict/claim-outcome`; plus `/health`, `/model-info`.
- Pydantic request/response schemas mirroring the `feature_spec`; strict
  validation and typed errors.
- Load versioned artefacts at startup; embed model + data version in every
  response.
- **Prediction log** — persist every request, response, model version,
  latency and timestamp to Postgres (`capstone_solution.prediction_log`).
- Containerised (`Dockerfile`), config via env, `docker compose` alongside the
  existing Postgres.
- Tests: schema validation, golden-prediction regression, contract tests.

**Deliverables:** `phase5_api/` — app, Dockerfile, compose fragment, API docs,
tests.

**Exit criteria:** `docker compose up` serves both models; invalid payloads
rejected cleanly; predictions logged with version metadata; p95 latency noted.

---

### Phase 6 — Monitoring, Drift Detection & Governance

**Goal:** long-term reliability and compliance.

**Work**
- Data validation gate on incoming requests (range/enum/schema) reusing
  `data_quality_rules.py`.
- Drift monitoring off `prediction_log`: feature drift (PSI / KS vs the Phase 3
  training reference), prediction-distribution drift, and — once outcomes land —
  performance drift.
- Scheduled drift job + threshold alerts; `drift_report` view/table.
- Audit log: who/what/when for predictions and any manual overrides.
- Governance docs: assumptions, limitations, retraining policy (trigger +
  cadence + rollback), incident runbook.

**Deliverables:** `phase6_monitoring/` — drift job, dashboards/queries,
`governance.md`, `retraining_policy.md`.

**Exit criteria:** drift job runs on a schedule and alerts on injected drift;
audit log immutable-by-convention; governance docs reviewed.

---

### Final Phase — Executive Business Presentation

**Goal:** translate the platform into leadership decisions.

Deck covering: the operational/financial problem, end-to-end architecture and
data flow, headline SQL + EDA insights, model performance *in money and risk
terms*, revenue-optimisation potential (leakage recoverable), and the
deployment / scaling / risk-management plan.

**Deliverable:** `final_presentation/` — slides + one-page executive brief +
architecture diagram.

## 5. Repository layout

```
capstone-healthcare-analytics/     # git repo root
├── objectives/                    # assignment brief + raw data (given)
└── solution/                      # all solution work (our root)
    ├── pyproject.toml / uv.lock    # uv-managed environment
    ├── .env(.example)              # DB + data-dir config
    ├── CLAUDE.md                   # agent instructions (uv, DB, reporting standard)
    ├── src/capstone/db.py          # shared connection / engine helpers
    ├── src/capstone/viz.py         # shared charting house style (all phases)
    ├── docs/PLAN.md                # this document
    ├── phase1_sql_analytics/       # BUILT
    │   ├── sql/01..06_*.sql
    │   ├── load_data.py  make_charts.py  run_phase1.py
    │   ├── output/*.csv  output/charts/*.png  PHASE1_FINDINGS.md
    │   └── README.md
    ├── phase2_eda/                 # next
    ├── phase3_models/
    ├── phase4_eval/
    ├── phase5_api/
    ├── phase6_monitoring/
    └── final_presentation/
```

## 6. Cross-cutting conventions

- **Config:** all connection detail in `.env`; code reads `capstone.db.SETTINGS`.
- **Reporting:** every finding is backed by a `capstone.viz` chart; each phase
  emits a `PHASE<n>_FINDINGS.md` organised by theme with charts embedded and the
  supporting numbers in a table appendix.
- **Deliverable format:** from Phase 2 on, each phase's main artefact and its
  outputs are a Jupyter notebook (runs top-to-bottom, charts/tables inline);
  reusable logic goes in `src/capstone/`. Phase 1 stays script-based.
- **Reproducibility:** each phase rebuilds from scratch via its notebook
  (Phase 2+) or entrypoint script (Phase 1); outputs are regenerated, never
  hand-edited.
- **Versioning:** models and feature pipelines carry semantic versions recorded
  in a manifest and echoed in API responses and the prediction log.
- **Leakage discipline:** the Phase 2 leakage register is the contract; Phase 3
  enforces it; Phase 4 verifies it by ablation.
- **Time:** `visit_date` is the temporal key for all splits and as-of features.
