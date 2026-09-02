# Phase 3 — Model Development (Classification)

Turns the Phase 2 feature frame into **two calibrated, time-validated
classifiers**:

- **Model A — visit risk** (`risk_score` Low/Medium/High): operational + clinical
  + patient-history features only. Use: staffing, bed planning, prioritisation.
- **Model B — pre-submission claim outcome** (`claim_status` Paid/Pending/
  Rejected): everything knowable before the claim is filed. Use: pre-submission
  denial triage.

**The deliverable is the notebook** — `phase3.ipynb` — which runs top-to-bottom,
renders every chart and table inline, and writes the artefacts below. Reusable
logic lives in `src/capstone/modeling.py`; the notebook stays thin and imports
it.

## Run it

```bash
# from solution/
uv sync

# headless: regenerate + execute the notebook and all outputs
uv run python phase3_models/run_phase3.py

# or open phase3.ipynb and Run All
```

`run_phase3.py` is idempotent — it checks for the Phase 2 feature frame
(rebuilding it via `phase2_eda/run_phase2.py` if missing), regenerates
`phase3.ipynb` from `notebook.py`, and executes it in place. Runtime ~1 minute.

## Outcome

| Model | Selected | Test balanced acc. (vs majority) | Recall on costly class | Verdict |
|---|---|---|---|---|
| A (visit risk) | majority baseline | 0.33 vs 0.33 — chance | High: 0% | **No signal** (Phase 2 predicted this). Shipped as a calibrated base-rate monitor for risk-mix tracking, not per-visit prioritisation. |
| B (claim outcome) | gradient boosting | 0.44 vs 0.33 | Rejected: **66%** (vs 0% majority, 62% simple rule) | **Works for triage.** Beats the majority and simple-rule baselines on balanced accuracy, macro-F1 and rejected-claim recall. Raw accuracy is lower by design (minority-recall trade-off); the operating threshold is tuned in Phase 4. |

Neither model beats the majority baseline on **raw accuracy** — the wrong bar on
a 50/30/20 and 60/25/15 target, where a constant "Low" / "Paid" classifier is
hard to beat on accuracy alone. `PHASE3_FINDINGS.md` documents this against the
`docs/PLAN.md` exit criteria.

## Method

- **Split:** calendar-anchored on `visit_date` — first 9 months train, next 1
  validate, last 2 test (18,665 / 2,075 / 4,260 rows). No shuffle.
- **Candidates (each model):** majority-class baseline, domain simple-rule
  baseline (Model B: flag the 15k–30k billed band), regularised multinomial
  logistic regression, gradient-boosted trees
  (`HistGradientBoostingClassifier`). Learned candidates use balanced class
  weights.
- **Selection:** ship the best learned candidate only if it beats the majority
  baseline's balanced accuracy by ≥ 0.02 **and** the simple rule's macro-F1;
  otherwise ship the majority baseline as a monitor.
- **Calibration:** Platt scaling (sigmoid) fitted on the validation month via a
  `FrozenEstimator` — no refit, no random CV fold, calibration data strictly in
  the future of training.
- **Leakage:** `capstone.features.leakage_violations()` is enforced before any
  fit.

## Outputs

| Path | Content |
|---|---|
| `phase3.ipynb` | executed review notebook (narrative + inline charts/tables) |
| `models/model_{a,b}.joblib` | fitted sklearn pipeline (`ColumnTransformer` + estimator) |
| `models/model_{a,b}_calibrated.joblib` | calibrated wrapper — Phase 5 thresholds against these |
| `models/training_manifest.json` | model version, data window, per-model feature lists, every candidate's metrics, environment (portable — relative paths) |
| `output/model_{a,b}_test_predictions.csv` | test-window predictions + calibrated probabilities |
| `output/*.csv` | candidate metrics, confusion matrices, calibration, target balance, summary |
| `output/charts/*.png` | 8 house-style charts, one per finding |
| `PHASE3_FINDINGS.md` | generated report — findings, charts embedded, exit-criteria table, appendix tables |

## Files

```
phase3.ipynb        the deliverable (generated + executed by run_phase3.py)
notebook.py         builds phase3.ipynb from a cell list (keeps the notebook thin)
run_phase3.py       headless entrypoint: ensure feature frame -> build -> execute
make_charts.py      one function per finding -> (key, path, caption); build_all(results)
report.py           assembles PHASE3_FINDINGS.md from the model results + charts
../src/capstone/modeling.py   time_split, feature_lists, build_preprocessor,
                              train_model, save_artifacts, load_model, reliability_curves
```

## Hand-off to Phase 4

`models/model_{a,b}.joblib` + `_calibrated.joblib` + `training_manifest.json`;
`output/model_{a,b}_test_predictions.csv` with calibrated probabilities. Phase 4
does explainability (SHAP, permutation importance), fairness parity, the leakage
ablation, ROC/PR curves, the operating-threshold choice, and the model cards.
