"""API routes: /health, /model-info, /predict/visit-risk, /predict/claim-outcome."""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request

from capstone import serving as S

from .config import CONFIG
from . import predictions_log as plog
from .schemas import (
    ClaimOutcomeRequest,
    ClaimOutcomeResponse,
    HealthResponse,
    ModelInfoResponse,
    VisitRiskRequest,
    VisitRiskResponse,
)

router = APIRouter()

_STARTED = time.monotonic()


def _client_host(request: Request) -> str | None:
    return request.client.host if request.client else None


def _finish(request: Request, endpoint: str, result: dict[str, Any], t0: float,
            payload: dict[str, Any]) -> dict[str, Any]:
    result["request_id"] = str(uuid.uuid4())
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    if CONFIG.log_predictions:
        plog.log_prediction(
            endpoint=endpoint,
            response=result,
            serving_version=request.app.state.bundle.config["serving_version"],
            request_payload=payload if CONFIG.log_request_payload else None,
            client_host=_client_host(request),
        )
    return result


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    bundle = getattr(request.app.state, "bundle", None)
    models_loaded = {
        "A": bool(bundle and "A" in bundle.pipelines),
        "B": bool(bundle and "B" in bundle.pipelines),
    }
    db_ok = plog.db_reachable()
    ok = all(models_loaded.values())
    return HealthResponse(
        status="ok" if ok else "degraded",
        models_loaded=models_loaded,
        db_reachable=db_ok,
        serving_version=bundle.config["serving_version"] if bundle else S.SERVING_VERSION,
        model_version=bundle.manifest["model_version"] if bundle else "unknown",
        uptime_seconds=round(time.monotonic() - _STARTED, 1),
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info(request: Request) -> Any:
    cfg = request.app.state.bundle.config
    models = {}
    for k, m in cfg["models"].items():
        models[k] = {
            "target": m["target"],
            "classes": m["classes"],
            "n_features": len(m["numeric_features"]) + len(m["categorical_features"]),
            "chosen_estimator": m["chosen_estimator"],
            "calibration_method": m["calibration_method"],
            "operating_threshold": m.get("operating_threshold"),
            "threshold_version": m.get("threshold_version"),
            "threshold_basis": m.get("threshold_basis"),
            "monitor_only": m.get("monitor_only"),
            "monitor_notice": m.get("monitor_notice"),
        }
    return {
        "serving_version": cfg["serving_version"],
        "model_version": cfg["model_version"],
        "feature_spec_version": cfg["feature_spec_version"],
        "temporal_key": cfg["temporal_key"],
        "data_window": cfg["data_window"],
        "generated": cfg["generated"],
        "categorical_domains": cfg["categorical_domains"],
        "models": models,
    }


@router.post("/predict/claim-outcome", response_model=ClaimOutcomeResponse,
             tags=["predict"], summary="Model B - pre-submission claim outcome")
def predict_claim_outcome(request: Request, body: ClaimOutcomeRequest) -> Any:
    t0 = time.perf_counter()
    payload = body.model_dump(mode="json")
    result = S.predict_claim_outcome(request.app.state.bundle, payload)
    return _finish(request, "/predict/claim-outcome", result, t0, payload)


@router.post("/predict/visit-risk", response_model=VisitRiskResponse,
             tags=["predict"], summary="Model A - visit risk (base-rate monitor)")
def predict_visit_risk(request: Request, body: VisitRiskRequest) -> Any:
    t0 = time.perf_counter()
    payload = body.model_dump(mode="json")
    result = S.predict_visit_risk(request.app.state.bundle, payload)
    return _finish(request, "/predict/visit-risk", result, t0, payload)
