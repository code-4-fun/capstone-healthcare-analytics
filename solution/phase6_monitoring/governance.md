# Governance - Hospital Operations & Revenue Risk Intelligence Platform

Owner: Analytics & AI team · Review cadence: quarterly, or on any retraining
event · Last reviewed: Phase 6 build.

This document states what the platform assumes, where it must not be trusted,
how it is monitored, and who is accountable. It is the companion to the two
model cards (`phase4_eval/model_card_A.md`, `model_card_B.md`) and the
`retraining_policy.md` / `runbook.md` in this folder.

## 1. What the platform is

A single data foundation (Postgres `capstone_solution`) plus two decision
models, served over FastAPI and monitored:

| Model | Decision | Status |
|---|---|---|
| **A - visit risk** | risk-mix monitoring for staffing / bed planning | **base-rate monitor only** - no signal above the class prior on the pilot year; predicts "Low" for every visit |
| **B - claim outcome** | pre-submission triage: flag claims likely to be Rejected | **in use** - recall ~62% on to-be-rejected claims at the operating threshold, ~Rs 1.8 Cr/year recoverable at ~850 review alerts/month |

## 2. Assumptions

- **`visit_date` is the only trusted temporal key.** `registration_date`
  precedes ~48% of visits and `billing_date` ordering is noisy (Phase 1). All
  time splits and as-of features use `visit_date`.
- **One pilot year of data** (2025-01-20 - 2026-01-20, 25,000 visits). Seasonal
  effects beyond one cycle are unobserved.
- **Capture floors are real**: `length_of_stay_hours` floors at 0.5 h, `billed_amount`
  at 500. These are kept, not imputed, and are modelled features.
- **The claim outcome is driven almost entirely by `billed_amount`** (Phase 2
  mutual-information screen, Phase 4 SHAP + permutation importance + ablation all
  agree). Model B is a billed-amount model with minor lift from department and
  provider history.
- **Callers supply business-known fields only.** As-of patient/provider history
  aggregates are optional; when omitted the server uses a documented no-history
  profile and lists the defaulted fields in `defaults_applied`.

## 3. Limitations and where not to trust the models

- **Model A must not drive individual-visit decisions.** It is exposed
  (`POST /predict/visit-risk`) purely so the risk-mix distribution can be
  tracked over time; every response carries the base-rate-monitor notice.
- **Model B is a review prompt, never an automated hold.** At the operating
  threshold ~78% of flags are not rejections. A missed rejection is lost
  revenue, not a patient-safety event; a false flag costs a few minutes of
  specialist time.
- **Model B calibration is reliable only to `P(Rejected)` ~0.35.** Thresholds
  above that are unsupported.
- **Pending is near-random** for Model B (ROC AUC ~0.5) - it separates Rejected
  from not-Rejected, nothing finer.
- **Fairness** was checked across gender, age band, city and insurer at the
  operating threshold (Phase 4). No group fails the four-fifths selection-rate
  test; the widest recall gap is across `city` (~0.20), driven by base rates
  rather than unequal treatment at equal risk. Phase 6 re-checks per-group
  parity on the live log (planned extension of the drift job).

## 4. Data handling

- The prediction log (`capstone_solution.prediction_log`) stores the
  caller-supplied request payload. It carries **no patient identifiers** beyond
  what the caller sends; callers integrating a `visit_id` or patient key
  **should hash it** before sending. No name, address, or contact field is
  accepted by any endpoint schema.
- Protected attributes (`gender`, `age`, `city`) are model inputs (Model B) and
  are logged; they are used for fairness monitoring, never surfaced to the
  review queue UI.
- The analytics tables are derived from the three source CSVs and hold no free
  text.
- **Retention:** `prediction_log` rows are kept for 24 months rolling (the
  drift-monitoring lookback plus a year); `drift_report` and
  `prediction_override` are kept indefinitely (audit record). `prediction_log`
  permits `DELETE` for retention pruning only; the two audit tables do not.

## 5. Monitoring design (Phase 6)

The drift job (`phase6_monitoring/drift_job.py`, scheduled by
`scheduler.py`) runs against `prediction_log` and writes
`capstone_solution.drift_report`:

| Signal | Method | Reference |
|---|---|---|
| Request validity | Phase 2 rule registry (`capstone.data_quality`) as a batch gate | fixed enum domains + pre-submission ranges |
| Feature drift | PSI (deciles) + two-sample KS per monitored feature | Phase 3 **test** window (`REFERENCE_SPLIT`) |
| Prediction drift | predicted-class-mix PSI | Phase 3 test-window predicted mix |
| Performance drift | recall on the costly class + net recoverable leakage, once actuals join on `visit_id` | Phase 4 operating baseline (`PHASE4_BASELINE`) |

The reference is the Phase 3 **test** split, not the training split: it is the
most recent window the models were validated on, and the training window's
calendar features are disjoint from any later window by construction.

### Alert rules (single source: `capstone.monitoring.ALERT_RULES`)

| Alert | Fires when |
|---|---|
| feature PSI | any monitored feature PSI > **0.25** |
| prediction PSI | predicted-class-mix PSI > **0.20** |
| performance | recall on the costly class drops > **10 points** vs the Phase 4 baseline |
| gate | > **1%** of served requests fail the validation gate |

Any alert row sets `drift_report.alert = true`; the Grafana **Hospital Drift**
dashboard and the runbook key off that flag.

## 6. Audit trail

- **`v_prediction_audit`** - every `prediction_log` row joined to any
  `prediction_override`: the who / what / when for predictions and overrides.
- **`prediction_override`** - a manual override records `actor`, `reason`,
  `original_class`, `override_class`. Overrides are advisory records; they do
  not change a served response after the fact.
- **Append-only, enforced.** `capstone_solution.forbid_mutation()` raises on any
  `UPDATE` to the three log tables, and on `DELETE` to the two audit tables.

## 7. Roles and responsibilities

| Role | Owns |
|---|---|
| Analytics & AI team | the models, the pipelines, the drift job, this doc; triage of drift alerts; retraining execution |
| Revenue-cycle lead | the Model B review queue; sign-off on the operating threshold; manual overrides |
| Clinical operations lead | how Model A's risk-mix trend feeds staffing; sign-off that Model A stays a monitor |
| Data platform / DBA | Postgres, the prediction log, retention jobs, backups |
| Compliance | annual review of data handling and fairness monitoring |

## 8. Change control

- Model / feature-pipeline versions are recorded in
  `phase3_models/models/training_manifest.json` and echoed in every API response
  and every log row (`model_version`, `feature_spec_version`, `serving_version`).
- The operating threshold is a versioned artefact in
  `phase5_api/serving_config.json` (`threshold_version`), re-derived from the
  Phase 4 net-recovery sweep on every retrain.
- Any change to a model, a threshold, the alert rules, or the leakage register
  requires a `retraining_policy.md` sign-off and an entry in the change log
  there.
