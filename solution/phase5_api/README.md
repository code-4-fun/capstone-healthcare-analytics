# Phase 5 — Deployment & API Integration (MLOps)

Serves the **persisted Phase 3 models** behind a FastAPI service — one endpoint
per model, strict Pydantic validation, versioned artefacts echoed in every
response, and a Postgres prediction log for Phase 6 drift monitoring. The goal
is operational readiness, not more accuracy.

| Endpoint | Model | Purpose |
|---|---|---|
| `POST /predict/claim-outcome` | B — pre-submission claim outcome | Flag a claim for review when calibrated `P(Rejected)` ≥ the Phase 4 operating threshold |
| `POST /predict/visit-risk` | A — visit risk | **Base-rate monitor** (no signal above the class prior); tracks risk-mix drift, not per-visit decisions |
| `GET /model-info` | — | Versions, feature counts, thresholds, categorical domains |
| `GET /health` | — | Model + database readiness |

Full request/response reference with `curl` examples: [`API.md`](API.md).

## Run it

```bash
# from solution/
uv sync

# regenerate serving_config.json, apply the DDL, run the tests, benchmark,
# and write PHASE5_FINDINGS.md + charts + openapi.json
uv run python phase5_api/run_phase5.py

# serve locally (http://localhost:8000/docs)
uv run uvicorn app.main:app --app-dir phase5_api --reload

# or the whole stack (API + its Postgres) in containers
docker compose -f phase5_api/docker-compose.yml up --build
```

`run_phase5.py` is idempotent — it rebuilds Phase 3 if the model artefacts are
missing, re-derives the Model B operating threshold from the Phase 4 net-recovery
sweep, and regenerates every output.

## How a request becomes a prediction

1. **Validate** — `app/schemas.py`. Every categorical field is bound to its
   Phase 1 CHECK-constraint domain (`capstone.serving.DOMAINS`); numerics carry
   ranges. `VisitRiskRequest` has no billing / LOS / `risk_score` field — the
   leakage register enforced at the edge. A bad payload → `422` with a typed
   `{"error": "validation_error", "detail": [...]}` body.
2. **Assemble features** — `capstone.serving.build_model_row`. Derives the same
   transforms as `capstone.features.build_feature_frame` (age band, billed band,
   `log1p`, seasonality, floor flags) and fills the optional as-of history
   aggregates with a no-history profile, recording which fields it defaulted.
3. **Predict** — the uncalibrated Phase 3 pipeline gives the class label; the
   calibrated wrapper gives the probabilities. Model B compares
   `P(Rejected)` to the operating threshold → `review` / `submit`.
4. **Log** — best-effort insert into `capstone_solution.prediction_log` with the
   model + data versions, latency and payload. A DB outage degrades `/health`
   but never fails the prediction.

## Versioning

`serving_config.json` (generated) carries `serving_version`, `model_version`,
`feature_spec_version`, and Model B's `operating_threshold` + `threshold_version`.
The app loads it at startup; every response and every log row echoes the
versions. Re-derive it whenever Phase 3 is retrained.

## Files

```
app/
  main.py            create_app(): lifespan loads the bundle, typed error handler, routes
  routes.py          /health, /model-info, /predict/{claim-outcome,visit-risk}
  schemas.py         Pydantic v2 request/response models (leakage-safe)
  predictions_log.py best-effort Postgres logging + DDL runner
  config.py          env-driven settings (Postgres via capstone.db.SETTINGS)
sql/prediction_log.sql   append-only log table (Phase 6 drift baseline)
serving_config.json      generated: versions + operating threshold
run_phase5.py            entrypoint: config -> DDL -> pytest -> benchmark -> report
make_charts.py           house-style latency + decision charts
report.py                assembles PHASE5_FINDINGS.md
tests/                    schema / golden-regression / contract / logging
Dockerfile, docker-compose.yml
../src/capstone/serving.py   reusable serving + feature assembly
```

## Hand-off to Phase 6

The running services, `capstone_solution.prediction_log` as the drift baseline,
`serving_config.json` (model + threshold versions), and `capstone.data_quality`
for the request-validation gate. Phase 6 adds drift detection, the scheduled
drift job, the audit log, and the governance / retraining-policy docs.
