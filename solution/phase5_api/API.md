# Phase 5 API reference

Base URL (local): `http://localhost:8000` · Interactive docs: `/docs` ·
OpenAPI: `/openapi.json` (also committed as `phase5_api/openapi.json`).

Every prediction response echoes `model_version` and `feature_spec_version`, and
is appended to `capstone_solution.prediction_log`.

---

## `POST /predict/claim-outcome` — Model B

Predicts the pre-submission claim outcome and returns a **review / submit**
decision. A claim is flagged for review when calibrated `P(Rejected)` ≥ the
operating threshold (Phase 4; default `0.19`).

### Request

| field | type | required | notes |
|---|---|---|---|
| `visit_date` | date (`YYYY-MM-DD`) | yes | the only trusted temporal key |
| `department` | enum | yes | `Cardiology · ER · General · ICU · Neurology · Orthopedics` |
| `visit_type` | enum | yes | `ER · ICU · OPD` |
| `age` | int 0–120 | yes | |
| `gender` | enum | yes | `F · M` |
| `city` | enum | yes | `Bangalore · Chennai · Delhi · Hyderabad · Mumbai · Pune` |
| `insurance_provider` | enum | yes | `CareOne · HealthPlus · MediCareX · SecureLife` |
| `chronic_flag` | bool | yes | |
| `billed_amount` | float ≥ 0 | yes | strongest predictor of the outcome |
| `length_of_stay_hours` | float ≥ 0 | yes | known before the claim is filed |
| `risk_score` | enum | yes | `Low · Medium · High` — assigned at the visit |
| `prior_visit_count` | int ≥ 0 | no | as-of patient history; omit → no-history estimate |
| `prior_high_risk_count` | int ≥ 0 | no | |
| `prior_rejection_count` | int ≥ 0 | no | |
| `prior_rejection_rate` | float 0–1 | no | derived from the counts if omitted |
| `days_since_last_visit` | float ≥ 0 | no | |
| `doctor_load_30d` | int ≥ 0 | no | attending doctor's visits in the prior 30 days |
| `provider_prior_claim_count` | float ≥ 0 | no | |
| `provider_prior_rejection_rate` | float 0–1 | no | |

Unknown fields are rejected (`extra = "forbid"`).

```bash
curl -s -X POST http://localhost:8000/predict/claim-outcome \
  -H 'content-type: application/json' -d '{
    "visit_date": "2025-12-08", "department": "ICU", "visit_type": "ICU",
    "age": 67, "gender": "F", "city": "Pune", "insurance_provider": "HealthPlus",
    "chronic_flag": true, "billed_amount": 24500, "length_of_stay_hours": 52.0,
    "risk_score": "High", "prior_visit_count": 3, "prior_rejection_count": 1
  }'
```

### Response `200`

```json
{
  "model": "B",
  "model_version": "1.0.0",
  "feature_spec_version": 2,
  "predicted_class": "Rejected",
  "probabilities": { "Paid": 0.505, "Pending": 0.235, "Rejected": 0.260 },
  "decision": {
    "action": "review",
    "flagged_for_review": true,
    "p_rejected": 0.260,
    "threshold": 0.19,
    "threshold_version": "1.0.0"
  },
  "defaults_applied": ["prior_high_risk_count", "days_since_last_visit", "prior_rejection_rate",
                       "doctor_load_30d", "provider_prior_claim_count", "provider_prior_rejection_rate"],
  "request_id": "08bc3b0e-129c-4e3e-bb68-50dd3be84cf4",
  "latency_ms": 24.9
}
```

`predicted_class` is the uncalibrated Phase 3 pipeline label; `probabilities` are
calibrated. The **decision** thresholds `P(Rejected)` — do not use the argmax
label for the review/submit call (Phase 4: the calibrated argmax collapses to
`Paid`).

---

## `POST /predict/visit-risk` — Model A

⚠️ **Base-rate monitor, not a per-visit predictor.** Model A has no signal above
the class prior on the pilot year (Phase 2 / Phase 3) and returns `Low` for every
visit. Use it to track the risk-mix distribution over time; the response carries
an explicit `monitor_notice`. See `phase4_eval/model_card_A.md`.

### Request

Same as `claim-outcome` **without** `billed_amount`, `length_of_stay_hours` and
`risk_score` — the Phase 2 leakage register excludes those from Model A, so the
schema rejects them (`422`).

```bash
curl -s -X POST http://localhost:8000/predict/visit-risk \
  -H 'content-type: application/json' -d '{
    "visit_date": "2025-12-08", "department": "Cardiology", "visit_type": "OPD",
    "age": 41, "gender": "M", "city": "Delhi", "insurance_provider": "CareOne",
    "chronic_flag": false
  }'
```

### Response `200`

```json
{
  "model": "A",
  "model_version": "1.0.0",
  "feature_spec_version": 2,
  "predicted_class": "Low",
  "probabilities": { "Low": 0.482, "Medium": 0.309, "High": 0.209 },
  "monitor_notice": "Model A is a calibrated base-rate monitor, not a per-visit predictor...",
  "defaults_applied": ["prior_visit_count", "..."],
  "request_id": "429c3515-a577-411d-a9b6-11d876a24054",
  "latency_ms": 6.9
}
```

---

## `GET /model-info`

Versions, per-model feature counts, Model B's operating threshold, and the
categorical domains (for client-side validation).

## `GET /health`

```json
{ "status": "ok", "models_loaded": {"A": true, "B": true}, "db_reachable": true,
  "serving_version": "1.0.0", "model_version": "1.0.0", "uptime_seconds": 12.4 }
```

`status` is `degraded` if a model failed to load. `db_reachable: false` means the
prediction log is not being written (predictions are still served).

---

## Errors

Validation failures return `422` with a typed body:

```json
{
  "error": "validation_error",
  "detail": [
    { "field": "department", "message": "Input should be 'Cardiology', 'ER', ...", "type": "enum" },
    { "field": "billed_amount", "message": "Input should be greater than or equal to 0", "type": "greater_than_equal" }
  ]
}
```
