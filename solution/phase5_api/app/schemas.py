"""Pydantic request/response schemas.

Mirror ``phase2_eda/feature_spec.yaml`` and the Phase 2 leakage register:

* Every categorical field is constrained to its Phase 1 CHECK-constraint domain
  (``capstone.serving.DOMAINS``).
* ``VisitRiskRequest`` (Model A) has **no** ``billed_amount`` /
  ``length_of_stay_hours`` / ``risk_score`` field - those are excluded for Model
  A by the leakage register, so the schema does not even accept them.
* The as-of history aggregates are optional; when omitted the server assumes a
  no-history profile and lists the defaulted fields in ``defaults_applied``.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from capstone import serving as S

Department = Enum("Department", {v: v for v in S.DOMAINS["department"]}, type=str)
VisitType = Enum("VisitType", {v: v for v in S.DOMAINS["visit_type"]}, type=str)
Gender = Enum("Gender", {v: v for v in S.DOMAINS["gender"]}, type=str)
City = Enum("City", {v: v for v in S.DOMAINS["city"]}, type=str)
InsuranceProvider = Enum("InsuranceProvider", {v: v for v in S.DOMAINS["insurance_provider"]}, type=str)
RiskScore = Enum("RiskScore", {v: v for v in S.DOMAINS["risk_score"]}, type=str)


class _HistoryMixin(BaseModel):
    """Optional as-of patient/provider history. Supply from the Phase 1 analytics
    layer when integrated; omit for a no-history estimate."""

    prior_visit_count: int | None = Field(None, ge=0, description="Patient's earlier visits, as of visit_date.")
    prior_high_risk_count: int | None = Field(None, ge=0, description="Patient's earlier High-risk visits.")
    prior_rejection_count: int | None = Field(None, ge=0, description="Patient's earlier Rejected claims.")
    prior_rejection_rate: float | None = Field(None, ge=0, le=1, description="prior_rejection_count / prior_visit_count.")
    days_since_last_visit: float | None = Field(None, ge=0, description="Days since the patient's previous visit.")
    doctor_load_30d: int | None = Field(None, ge=0, description="Attending doctor's other visits in the prior 30 days.")
    provider_prior_claim_count: float | None = Field(None, ge=0, description="Insurer's earlier claims, as of visit_date.")
    provider_prior_rejection_rate: float | None = Field(None, ge=0, le=1, description="Insurer's historical rejection rate.")


class _VisitCore(_HistoryMixin):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    visit_date: date = Field(..., description="Date of the visit - the only trusted temporal key.")
    department: Department
    visit_type: VisitType
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    city: City
    insurance_provider: InsuranceProvider
    chronic_flag: bool = Field(..., description="1 if the patient is flagged chronic.")


class VisitRiskRequest(_VisitCore):
    """Model A input. Operational + clinical + patient-history only - no billing
    or LOS field (leakage register: excluded for Model A)."""


class ClaimOutcomeRequest(_VisitCore):
    """Model B input. Everything knowable before the claim is submitted."""

    billed_amount: float = Field(..., ge=0, description="Amount billed on the claim.")
    length_of_stay_hours: float = Field(..., ge=0, description="Recorded length of stay (floored at 0.5h in source).")
    risk_score: RiskScore = Field(..., description="Clinical risk band assigned at the visit (pre-submission input).")


# --------------------------------------------------------------------------
# responses
# --------------------------------------------------------------------------

class ClaimDecision(BaseModel):
    action: Literal["review", "submit"]
    flagged_for_review: bool
    p_rejected: float
    threshold: float
    threshold_version: str


class _PredictionBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: Literal["A", "B"]
    model_version: str
    feature_spec_version: int
    predicted_class: str
    probabilities: dict[str, float]
    defaults_applied: list[str]
    request_id: str
    latency_ms: float


class ClaimOutcomeResponse(_PredictionBase):
    decision: ClaimDecision


class VisitRiskResponse(_PredictionBase):
    monitor_notice: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: dict[str, bool]
    db_reachable: bool
    serving_version: str
    model_version: str
    uptime_seconds: float


class ModelSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    target: str
    classes: list[str]
    n_features: int
    chosen_estimator: str
    calibration_method: str
    operating_threshold: float | None = None
    threshold_version: str | None = None
    threshold_basis: str | None = None
    monitor_only: bool | None = None
    monitor_notice: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    serving_version: str
    model_version: str
    feature_spec_version: int
    temporal_key: str
    data_window: dict
    generated: str
    categorical_domains: dict[str, list[str]]
    models: dict[str, ModelSummary]


class ErrorResponse(BaseModel):
    error: str
    detail: list[dict] | str
