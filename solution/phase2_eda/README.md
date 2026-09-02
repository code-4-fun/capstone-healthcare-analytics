# Phase 2 — Exploratory Data Analysis & Data Quality

Turns the Phase 1 analytics layer into **modelling readiness**: profiled
distributions, quantified data-quality findings with a written handling policy, a
leakage-safe feature catalogue, and reusable validators.

**The deliverable is the notebook** — `phase2.ipynb` — which runs top-to-bottom,
renders every chart and table inline, and writes the outputs below. Reusable
logic lives in `src/capstone/` (`eda`, `features`, `data_quality`); the notebook
stays thin and imports it.

## Run it

```bash
# from solution/
uv sync
docker start capstone-project-postgres

# headless: regenerate + execute the notebook and all outputs
uv run python phase2_eda/run_phase2.py

# or open phase2.ipynb and Run All
```

`run_phase2.py` is idempotent — it checks the Phase 1 layer (rebuilding it via
`phase1_sql_analytics/run_phase1.py` if unreachable), regenerates `phase2.ipynb`
from `notebook.py`, and executes it in place.

## Outputs

| Path | Content |
|---|---|
| `phase2.ipynb` | executed review notebook (narrative + inline charts/tables) |
| `feature_spec.yaml` | feature catalogue + per-model leakage register |
| `output/feature_frame.parquet` | as-of feature matrix, one row per visit (Phase 3 input) |
| `output/*.csv` | every analysis table (profiling, DQ report, business analyses, feature signal) |
| `output/charts/*.png` | 19 house-style charts, one per finding |
| `PHASE2_FINDINGS.md` | generated report — findings by theme, charts embedded, tables in an appendix |

## Files

```
phase2.ipynb           the deliverable (generated + executed by run_phase2.py)
notebook.py            builds phase2.ipynb from a cell list (keeps the notebook thin)
run_phase2.py          headless entrypoint: ensure layer -> build -> execute
make_charts.py         one function per finding -> (key, path, caption); build_all()
report.py              assembles PHASE2_FINDINGS.md from the tables + charts
../src/capstone/eda.py           profiling + business analyses (returns DataFrames)
../src/capstone/features.py      as-of feature builder, FEATURE_SPEC, leakage_violations()
../src/capstone/data_quality.py  Rule registry, validate(), add_quality_flags(), apply_training_exclusions()
```

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
a **per-model verdict** (`allow` / `exclude` / `target`).
`capstone.features.leakage_violations(frame)` enforces:

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
