"""OpenAPI contract: the four routes and their response models are published."""
from __future__ import annotations

EXPECTED_ROUTES = {
    ("/health", "get"),
    ("/model-info", "get"),
    ("/predict/claim-outcome", "post"),
    ("/predict/visit-risk", "post"),
}


def test_openapi_publishes_the_four_routes(client):
    spec = client.get("/openapi.json").json()
    present = {(path, method) for path, ops in spec["paths"].items() for method in ops}
    assert EXPECTED_ROUTES <= present


def test_predict_responses_are_typed(client):
    spec = client.get("/openapi.json").json()
    for path, model in [
        ("/predict/claim-outcome", "ClaimOutcomeResponse"),
        ("/predict/visit-risk", "VisitRiskResponse"),
    ]:
        ref = spec["paths"][path]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith(f"/{model}")
    for model in ("ClaimOutcomeResponse", "VisitRiskResponse", "HealthResponse", "ModelInfoResponse"):
        assert model in spec["components"]["schemas"]


def test_validation_error_shape_is_documented(client):
    # the custom 422 body is the typed ErrorResponse, not FastAPI's default
    r = client.post("/predict/visit-risk", json={})
    assert r.status_code == 422
    body = r.json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == "validation_error"
