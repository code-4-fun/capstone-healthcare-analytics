# Phase 2 — Exploratory Data Analysis & Data Quality

Turns the Phase 1 analytics layer into **modelling readiness**: profiled
distributions, quantified data-quality findings with a written handling policy, a
leakage-safe feature catalogue, and reusable validators.

## Run it

```bash
# from solution/
uv sync
docker start capstone-project-postgres
uv run python phase2_eda/run_phase2.py
```

Idempotent — rebuilds every CSV, chart and document from
`capstone_solution.v_visit_billing`. If the analytics layer is unreachable it
rebuilds it first (`phase1_sql_analytics/run_phase1.py`).

## What gets built

| Step | File | Result |
|---|---|---|
| 2 | `analyses.py` | profiling + business-analysis tables → `output/*.csv` |
| 3 | `features.py` | `feature_spec.yaml` (catalogue + leakage register) + `output/feature_frame.parquet` (as-of matrix, one row per visit) |
| 4 | `data_quality_rules.py` | `output/data_quality_report.csv` — 22 importable validators |
| 5 | `run_phase2.py` | leakage self-check — asserts no post-outcome field is model-eligible |
| 6 | `make_charts.py` | 19 charts → `output/charts/` (house style: `capstone.viz`) |
| 7 | `run_phase2.py` | `PHASE2_FINDINGS.md` — findings by theme, every claim backed by a chart, tables in an appendix |

## Key findings

- **Model A (visit risk) has no learnable signal.** Every eligible feature sits
  at the permuted-target noise floor (< 0.5% of target entropy); `risk_score` is
  effectively random. Phase 3 should treat Model A as a calibrated base-rate
  baseline and document the ceiling.
- **Model B (claim outcome) is a billed-amount model.** `billed_amount` /
  `billed_band` are the only features above the noise floor. Rejection rate is
  **non-monotonic** — 22.7% at 15k–30k vs 4.5% / 6.5% at the extremes — and that
  mid band carries 70% of the 91.8M denial leakage.
- **Targets are imbalanced toward the cheap class** — 50/30/20 (risk) and
  60/25/15 (claim). Class weighting + threshold tuning mandatory; recall on
  High / Rejected is the headline metric.
- **Four Phase 1 DQ findings quantified with a handling policy**: 0.5h / 500
  capture floors (keep + flag), status↔approved_amount decoupling
  (analysis-only), ~5% MCAR missingness on the two billing outcome fields
  (impute for reporting only), temporal inconsistency (~50% billing-before-visit)
  → **`visit_date` is the only temporal key**.
- **The business is structurally uniform** — no seasonality, no weekday pattern,
  interchangeable departments and insurers, no length-of-stay drivers.

## Leakage register (contract for Phase 3)

`feature_spec.yaml` gives every field a definition, source, as-of rule, dtype and
a **per-model verdict** (`allow` / `exclude` / `target`). Enforced rules:

- `visit_date` is the sole temporal key; nothing derives from `billing_date` or
  `registration_date`.
- No post-outcome field (`approved_amount`, `payment_days`, `claim_status`,
  `collected/leakage/pending_amount`) feeds Model A or the pre-submission
  Model B.
- `length_of_stay_hours` and `billed_amount` are excluded from Model A (unknown
  at admission / not operational) but allowed for Model B.
- `risk_score` is the Model A target and a legitimate Model B input.
- Patient/provider history counts only visits with strictly earlier
  `visit_date`.

## Files

```
run_phase2.py          single entrypoint (idempotent)
analyses.py            profiling + business analyses -> tidy DataFrames / CSVs
features.py            as-of feature builder + FEATURE_SPEC + feature_spec.yaml writer
data_quality_rules.py  Rule registry, validate(), add_quality_flags(), apply_training_exclusions()
make_charts.py         one function per finding -> (key, path, caption); build_all()
feature_spec.yaml      generated feature catalogue + leakage register
PHASE2_FINDINGS.md     generated report
output/                generated CSVs, feature_frame.parquet, charts/
```
