# Phase 4 — Model Evaluation & Explainability

Proves the two Phase 3 classifiers are interpretable, reliable and safe, and
turns **Model B** into an operating decision with a business case in money terms.

**The deliverable is the notebook** — `phase4.ipynb` — which runs top-to-bottom,
renders every chart and table inline, and writes the artefacts below. Reusable
logic lives in `src/capstone/evaluation.py`; the notebook stays thin and imports
it. The **persisted Phase 3 models are evaluated as-is** — nothing is retrained
or retuned (the leakage ablation trains throwaway variants for evidence only).

## Run it

```bash
# from solution/
uv sync
uv run python phase4_eval/run_phase4.py      # regenerate + execute the notebook
# or open phase4.ipynb and Run All
```

`run_phase4.py` is idempotent — it rebuilds Phase 3 (`phase3_models/run_phase3.py`)
if its artefacts are missing, regenerates `phase4.ipynb` from `notebook.py`, and
executes it in place. Runtime ~2 minutes (SHAP + permutation importance).

## Outcome

| Model | Verdict | Key numbers (test window) |
|---|---|---|
| **B — claim outcome** | **Deploy as a pre-submission triage assistant.** | Operating threshold: calibrated `P(Rejected)` ≥ 0.19, chosen on validation to maximise net recoverable leakage. Recall on Rejected ~62% (377/608), ~850 review alerts/month, ~₹15L/month recoverable denial leakage (~₹5L net of review cost). Billed-amount model — permutation importance, SHAP and the ablation all agree. |
| **A — visit risk** | **Monitor only, never per-visit.** | Predicts `Low` for every visit, 0% recall on High-risk, ROC AUC ~0.50, zero permutation importance. Retained to track the risk-mix distribution over time. |

**Leakage:** verified by ablation — dropping allowed features barely moves the
test metrics; injecting a forbidden post-outcome field (`approved_amount` →
balanced accuracy 0.96) spikes them. The shipped models sit at the clean row.

**Fairness:** no protected attribute (gender, age band, city, insurer) fails the
four-fifths selection-rate test at the operating threshold; the widest recall
gap is across cities (~0.20), tracking base-rate differences. Mitigation is
per-group monitoring in Phase 6, not a model change.

## Method

- **Operating threshold (Model B):** sweep the calibrated `P(Rejected)` cut-off;
  pick the point that maximises `recoverable leakage on caught rejections
  (40% haircut) − ₹1,500 review cost × claims flagged`, chosen on the validation
  month and applied unchanged to test. Both assumptions are documented constants
  in `capstone.evaluation` (`REVIEW_COST`, `RECOVERY_RATE`).
- **Explainability:** `sklearn.inspection.permutation_importance` (balanced-
  accuracy scoring) on both models; SHAP `TreeExplainer` on Model B's booster
  (global mean-|value| + three local explanations), with a permutation-only
  fallback if SHAP cannot explain the pipeline.
- **Leakage ablation:** retrain Model B's Phase 3 learner (`gbm`) with feature
  groups dropped or a forbidden field injected; score on test. Model A ships as a
  majority-class baseline that ignores every feature, so it cannot leak and is
  not ablated (its zero permutation importance is the direct evidence).
- **Fairness:** selection rate, recall, FPR and calibration gap per subgroup at
  the operating threshold, with a four-fifths parity summary.

## Outputs

| Path | Content |
|---|---|
| `phase4.ipynb` | executed review notebook (narrative + inline charts/tables) |
| `PHASE4_FINDINGS.md` | generated report — findings, charts embedded, exit-criteria table, appendix |
| `model_card_A.md`, `model_card_B.md` | generated model cards (intent, data, metrics, thresholds, limitations, ethical considerations, retraining triggers) |
| `output/charts/*.png` | 13 house-style charts, one per finding (12 if SHAP is unavailable) |
| `output/*.csv` | per-class metrics, threshold sweep, permutation importance, SHAP, leakage ablation, fairness tables |

## Files

```
phase4.ipynb        the deliverable (generated + executed by run_phase4.py)
notebook.py         builds phase4.ipynb from a cell list (keeps the notebook thin)
run_phase4.py       headless entrypoint: ensure Phase 3 artefacts -> build -> execute
make_charts.py      one function per finding -> (key, path, caption); build_all(ctx)
report.py           assembles PHASE4_FINDINGS.md + the two model cards
../src/capstone/evaluation.py   prediction frames, metrics, threshold sweep,
                                leakage ablation, permutation importance, SHAP, fairness
```

## Hand-off to Phase 5

The persisted models + calibrated wrappers + the **operating threshold 0.19** on
`P(Rejected)` + the two model cards. Phase 5 serves Model B behind
`POST /predict/claim-outcome`, rate-limits the review queue, and logs every
prediction with model + threshold version for Phase 6 drift and fairness
monitoring.
