# Model Card - Pre-submission Claim Outcome (Model B)

*Hospital Operations & Revenue Risk Intelligence Platform*  

- Model version: `1.0.0`  |  feature spec: v2  |  card generated: 2026-09-02
- Artefacts: `phase3_models/models/model_b.joblib` (+ `_calibrated.joblib`), `training_manifest.json`
- Selected estimator: **gbm**; calibration: sigmoid on the validation month

## 1. Intent

**Purpose.** Flag claims that are likely to be **Rejected** *before* they are submitted, so a claims specialist can rework or correct them and avoid a denial.

**In scope.** A ranked review queue of pre-submission claims. **Out of scope.** Automatically holding, denying or re-pricing a claim; predicting the *approved amount*; any use after adjudication. Every flag is a review prompt for a human, not an action.

## 2. Data

- **Source.** `capstone_solution.v_visit_billing` (Phase 1), as-of feature matrix `phase2_eda/output/feature_frame.parquet` (Phase 2).
- **Window.** 2025-01-20 -> 2026-01-20, 25,000 visits. Split: 9 months train / 1 validate / 2 test, calendar-anchored on visit_date.
- **Features.** 29 (16 numeric + 13 categorical). Target: `claim_status` (Paid / Pending / Rejected).
- **Leakage register.** `capstone.features` - `visit_date` is the only temporal key; no `approved_amount`, `payment_days`, `claim_status` or `billing_date` derivative. Model B may use `billed_amount`, `length_of_stay_hours` and `risk_score` - all known before the claim is filed. Verified by ablation in `PHASE4_FINDINGS.md` section 5.

## 3. Metrics (test window)

| class    |   precision |   recall |     f1 |   support |
|:---------|------------:|---------:|-------:|----------:|
| Paid     |      0.6935 |   0.3934 | 0.502  |      2565 |
| Pending  |      0.2838 |   0.2585 | 0.2706 |      1087 |
| Rejected |      0.2198 |   0.6562 | 0.3293 |       608 |

- **Business (at operating threshold P(Rejected) >= 0.19):** recall on Rejected **62%** (377/608), precision 22%, ~848 alerts/month.
- **Financial:** Rs. 30.5L recoverable denial leakage over the 2-month test window (Rs. 5.0L net of review cost); ~Rs. 1.8Cr/year at this operating point.
- Balanced accuracy 0.44 vs 0.33 majority; macro-F1 0.37. Raw accuracy 39.6% is *below* the 60.2% majority rate by design - the model is tuned for minority recall.

## 4. Thresholds

- **Operating threshold: calibrated `P(Rejected)` >= 0.19.** Chosen on the validation month as the point maximising net recoverable leakage (Rs. 1,500/review, 40% of a caught rejection's leakage recoverable), applied unchanged to test.
- Do **not** use the argmax label - the calibrated argmax collapses to Paid on this class prior. The Paid/Pending split (for non-flagged claims) comes from the uncalibrated pipeline.
- The threshold is a versioned artefact: it travels with the model into Phase 5 and is re-derived whenever the model is retrained.

## 5. Limitations

- **Single-feature model.** `billed_amount` and its `15k-30k` band supply essentially all the signal; the other 26 features are close to inert (permutation importance within noise of zero). A shift in the billing-amount distribution will degrade the model directly.
- **Low precision.** At the operating threshold ~22% of flags are not rejections - the review queue is mostly Paid claims getting a second look. This is acceptable for a review prompt, not for any automated action.
- **Pending is near-random** (ROC AUC ~0.5). The model separates Rejected from not-Rejected; it does not usefully predict Pending.
- **Calibrated only to ~0.35.** Above that the reliability curve is unstable (few claims), so thresholds above ~0.35 are not supported.

## 6. Ethical considerations

- **Fairness.** Selection rate, recall and FPR were checked across gender, age band, city and insurer at the operating threshold. No group fails the four-fifths selection-rate test; the widest recall gap is across `city` (0.20), driven by base-rate differences rather than unequal treatment at equal risk (calibration gaps are small). See `PHASE4_FINDINGS.md` section 6.
- **Mitigations.** Flags are review prompts, never automated holds; specialists see the queue without protected attributes; Phase 6 monitors per-group parity on the live log and alerts on drift below the four-fifths line.
- **Clinical / financial context.** A missed rejection is lost revenue, not a patient-safety event; a false flag costs a few minutes of specialist time. The error trade-off is financial and reversible, which is why a recall-favouring threshold is appropriate.

## 7. Retraining triggers

Retrain (and re-derive the calibration and, for Model B, the operating threshold) when any of the following fires - these are the signals Phase 6 monitors on the prediction log:

- **Scheduled:** every 3 months on a rolling 12-month window.
- **Feature drift:** PSI > 0.2 on `billed_amount` / `billed_band` between the live request stream and the Phase 3 training reference (this is the feature the model depends on).
- **Outcome drift:** the observed rejection rate on adjudicated claims moves more than +/- 3 points from the ~14% base rate, or the 15k-30k band stops being the rejection peak.
- **Calibration decay:** reliability-curve error on `P(Rejected)` in the 0.1-0.35 range exceeds 0.1, or realised recall at the operating threshold drops below 55% on a trailing quarter of adjudicated claims.
- **Fairness:** any protected group's selection-rate ratio drops below 0.8 or calibration gap exceeds 0.1 on the live log.

---

*Generated by `phase4_eval/report.py` from `capstone.evaluation` on 2026-09-02T09:40:10+00:00. Do not hand-edit - re-run `phase4_eval/run_phase4.py`.*

