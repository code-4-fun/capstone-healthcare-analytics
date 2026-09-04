# Retraining policy - Models A & B

Companion to `governance.md` and the model cards. Defines **when** the models
are retrained, **how**, and **how to roll back**.

## 1. Triggers

Retrain (and re-derive calibration and, for Model B, the operating threshold)
when **any** of the following fires. The first four are watched automatically by
the Phase 6 drift job on `capstone_solution.prediction_log`.

| # | Trigger | Threshold | Source |
|---|---|---|---|
| 1 | **Feature drift** | any monitored feature PSI > 0.25 on **two consecutive** scheduled runs | `drift_report` `metric_kind = feature_psi` |
| 2 | **Prediction-mix drift** | predicted-class-mix PSI > 0.20 sustained over a week | `drift_report` `prediction_psi` |
| 3 | **Performance drift (Model B)** | recall on Rejected at the operating threshold drops below **0.52** (> 10 points under the 0.62 Phase 4 baseline) on a trailing quarter of adjudicated claims | `drift_report` `perf_recall_costly` |
| 4 | **Calibration decay (Model B)** | reliability-curve error on `P(Rejected)` in 0.1-0.35 exceeds 0.10 | quarterly calibration check |
| 5 | **Gate failure surge** | > 1% of requests fail the validation gate for > 24 h | `drift_report` `gate_fail_rate` - investigate the upstream source first; retrain only if the new inputs are legitimate |
| 6 | **Calendar** | 12 months since the last retrain, regardless of drift | scheduler |
| 7 | **Data / process change** | a new department, city, insurer, tariff schedule, or claim-submission workflow | notified by revenue-cycle / clinical ops |

Model A is **not** retrained on performance triggers (it has no performance to
lose). It is refreshed on triggers 6-7 so its risk-mix reference stays current,
and it is **re-evaluated for signal** at every refresh - if a future data
vintage gives it lift above the class prior, it graduates from monitor to
predictor via a fresh Phase 3/4 pass and a model-card update.

## 2. Procedure

1. **Freeze a data snapshot** - extend the `v_visit_billing` window to the new
   cutoff; record the row count and date range.
2. **Re-run the pipeline end-to-end**, in order:
   - `phase2_eda/run_phase2.py` - refreshes `feature_frame.parquet`,
     `feature_spec.yaml` and the leakage register.
   - `phase3_models/run_phase3.py` - re-splits on `visit_date`, retrains the
     four candidates per model, re-selects, re-calibrates, writes a new
     `training_manifest.json` with a bumped `model_version`.
   - `phase4_eval/run_phase4.py` - re-evaluates, re-derives the Model B
     operating threshold, regenerates both model cards.
   - `phase5_api/run_phase5.py` - rewrites `serving_config.json` (new
     `model_version`, `threshold_version`).
3. **Diff the model cards** - per-class metrics, the operating threshold,
   fairness parity, the feature-importance ordering. Any material regression or
   a new fairness gap past the four-fifths line blocks promotion.
4. **Shadow** - run the candidate against a trailing month of `prediction_log`
   payloads (replay, as Phase 6 seeds) and compare its predicted-class mix and,
   where actuals exist, its recall to the incumbent. Promote only if it is at
   least as good on recall-on-costly at a comparable alert volume.
5. **Promote** - deploy the new artefacts + `serving_config.json`. The API picks
   up `model_version` at startup; it flows into every response and log row
   automatically.
6. **Record** - add a row to the change log below and reset the drift baseline
   (the reference split moves with the new Phase 3 test window).

## 3. Rollback

- The previous model artefacts (`model_a.joblib`, `model_b.joblib`, the
  calibrated wrappers) and the previous `serving_config.json` are retained for
  at least two versions.
- **To roll back:** redeploy the prior `phase3_models/models/` directory and the
  prior `serving_config.json`, restart the API. `model_version` in responses and
  the log immediately reflects the rollback, so drift monitoring and the audit
  trail stay consistent.
- No schema change is involved in a model swap, so rollback is a
  restart, not a migration.
- If a bad threshold (not a bad model) shipped, roll back `serving_config.json`
  alone - the models are unchanged.

## 4. Sign-off

| Step | Approver |
|---|---|
| Snapshot + pipeline re-run | Analytics & AI team |
| Model-card diff review | Analytics & AI team + Revenue-cycle lead (Model B) / Clinical ops lead (Model A) |
| Promotion to production | Revenue-cycle lead (Model B); Clinical ops lead (Model A) |
| Rollback | Analytics & AI team (notify the relevant lead) |

## 5. Change log

| Date | Version | Trigger | Notes |
|---|---|---|---|
| Phase 3 build | `model_version 1.0.0` | initial | 9m train / 1m val / 2m test on the pilot year; Model A shipped as a base-rate monitor, Model B as gradient-boosted triage at `P(Rejected) >= 0.19`. |
