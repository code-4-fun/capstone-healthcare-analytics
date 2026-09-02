"""FastAPI application factory for the Phase 5 API.

Serves the persisted Phase 3 models (Model A - visit risk, Model B -
pre-submission claim outcome) behind typed, validated endpoints, echoing the
model + data version in every response and logging every prediction to Postgres
for Phase 6 drift monitoring.

Run locally:  ``uv run uvicorn app.main:app --app-dir phase5_api --reload``
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from capstone import serving as S

from .config import CONFIG
from . import predictions_log as plog
from .routes import router
from .schemas import ErrorResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("capstone.api")

DESCRIPTION = """\
Prediction services for the Hospital Operations & Revenue Risk Intelligence Platform.

* **POST `/predict/claim-outcome`** - Model B. Flags a claim for pre-submission
  review when calibrated `P(Rejected)` clears the Phase 4 operating threshold.
* **POST `/predict/visit-risk`** - Model A. A calibrated **base-rate monitor**
  (no signal above the class prior on the pilot year); use it to track risk-mix
  drift, not to prioritise an individual visit.
* **GET `/model-info`** - versions, feature counts, thresholds, domains.
* **GET `/health`** - model + database readiness.

Every response carries `model_version` / `feature_spec_version`; every prediction
is appended to `capstone_solution.prediction_log`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading serving bundle from %s (config: %s)", CONFIG.models_dir, CONFIG.serving_config_path)
    app.state.bundle = S.load_serving_bundle(CONFIG.models_dir, CONFIG.serving_config_path)
    log.info("models loaded: A=%s B=%s | model_version=%s | threshold(B)=%.3f",
             "A" in app.state.bundle.pipelines, "B" in app.state.bundle.pipelines,
             app.state.bundle.manifest["model_version"], app.state.bundle.operating_threshold)
    if CONFIG.log_predictions:
        app.state.db_ready = plog.ensure_table()
        log.info("prediction_log ready: %s", app.state.db_ready)
    else:
        app.state.db_ready = False
        log.info("prediction logging disabled (API_LOG_PREDICTIONS=false)")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=CONFIG.title,
        version=CONFIG.version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # typed, stable error shape (FastAPI's default nests under "detail" only)
        errors = [
            {
                "field": ".".join(str(p) for p in e["loc"] if p not in ("body",)),
                "message": e["msg"],
                "type": e["type"],
            }
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content=ErrorResponse(error="validation_error", detail=errors).model_dump())

    @app.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    app.include_router(router)
    return app


app = create_app()
