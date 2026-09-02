"""Reusable data-quality validators for the hospital analytics platform.

Phase 2 formalises the data-quality findings surfaced in Phase 1 into a set of
importable, row-level validators. Later phases (feature pipeline, API request
gate, drift monitor) import ``RULES`` / ``validate`` from here so the definition
of "valid" lives in exactly one place.

Each :class:`Rule` inspects a dataframe shaped like the ``v_visit_billing`` spine
(one row per visit) and returns a boolean mask of the *offending* rows. A rule
whose required columns are absent is skipped (reported as ``n/a``).

The module also carries the agreed **handling policy** for each finding
(:data:`HANDLING`) — impute, exclude-from-training, or keep-as-is — and two
convenience transforms, :func:`apply_training_exclusions` and
:func:`add_quality_flags`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# --- domain vocabulary ----------------------------------------------------
VALID_GENDER = {"M", "F"}
VALID_VISIT_TYPE = {"ER", "OPD", "ICU"}
VALID_RISK = {"Low", "Medium", "High"}
VALID_CLAIM_STATUS = {"Paid", "Pending", "Rejected"}
VALID_DEPARTMENT = {"ER", "ICU", "General", "Cardiology", "Neurology", "Orthopedics"}
VALID_CITY = {"Hyderabad", "Pune", "Bangalore", "Mumbai", "Delhi", "Chennai"}
VALID_PROVIDER = {"MediCareX", "CareOne", "HealthPlus", "SecureLife"}

# --- measured floors (Phase 2, quantified in PHASE2_FINDINGS.md) ---------
LOS_FLOOR_HOURS = 0.5
BILLED_FLOOR_AMOUNT = 500.0
AGE_MIN, AGE_MAX = 0, 120


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str  # ERROR | WARN | INFO
    scope: str
    description: str
    columns: tuple[str, ...]
    predicate: Callable[[pd.DataFrame], pd.Series]
    #: how downstream phases should treat flagged rows
    handling: str = "review"

    def applicable(self, df: pd.DataFrame) -> bool:
        return all(c in df.columns for c in self.columns)

    def flag(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask, index-aligned to ``df``, True where the row offends."""
        if not self.applicable(df):
            return pd.Series(False, index=df.index)
        mask = self.predicate(df)
        if not isinstance(mask, pd.Series):
            mask = pd.Series(np.asarray(mask), index=df.index)
        return mask.reindex(df.index, fill_value=False).fillna(False).astype(bool)


def _f(col: str) -> Callable:  # numeric coercion helper
    return lambda df: pd.to_numeric(df[col], errors="coerce")


def _dt(col: str) -> Callable:
    return lambda df: pd.to_datetime(df[col], errors="coerce")


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------
RULES: list[Rule] = [
    # -- enum domains -----------------------------------------------------
    Rule("gender_enum", "ERROR", "patient", "gender outside {M,F}",
         ("gender",), lambda df: ~df["gender"].isin(VALID_GENDER), "exclude"),
    Rule("visit_type_enum", "ERROR", "visit", "visit_type outside {ER,OPD,ICU}",
         ("visit_type",), lambda df: ~df["visit_type"].isin(VALID_VISIT_TYPE), "exclude"),
    Rule("risk_score_enum", "ERROR", "visit", "risk_score outside {Low,Medium,High}",
         ("risk_score",), lambda df: ~df["risk_score"].isin(VALID_RISK), "exclude"),
    Rule("claim_status_enum", "ERROR", "billing", "claim_status outside {Paid,Pending,Rejected}",
         ("claim_status",), lambda df: ~df["claim_status"].isin(VALID_CLAIM_STATUS), "exclude"),
    Rule("department_enum", "WARN", "visit", "department not in the known set",
         ("department",), lambda df: ~df["department"].isin(VALID_DEPARTMENT), "review"),
    Rule("city_enum", "WARN", "patient", "city not in the known set",
         ("city",), lambda df: ~df["city"].isin(VALID_CITY), "review"),
    Rule("provider_enum", "WARN", "patient", "insurance_provider not in the known set",
         ("insurance_provider",), lambda df: ~df["insurance_provider"].isin(VALID_PROVIDER), "review"),
    # -- numeric ranges ------------------------------------------------
    Rule("age_range", "ERROR", "patient", f"age outside [{AGE_MIN}, {AGE_MAX}]",
         ("age",), lambda df: ~_f("age")(df).between(AGE_MIN, AGE_MAX), "exclude"),
    Rule("los_negative", "ERROR", "visit", "length_of_stay_hours < 0",
         ("length_of_stay_hours",), lambda df: _f("length_of_stay_hours")(df) < 0, "exclude"),
    Rule("billed_negative", "ERROR", "billing", "billed_amount < 0",
         ("billed_amount",), lambda df: _f("billed_amount")(df) < 0, "exclude"),
    Rule("payment_days_negative", "WARN", "billing", "payment_days < 0",
         ("payment_days",), lambda df: _f("payment_days")(df) < 0, "impute"),
    Rule("approved_exceeds_billed", "ERROR", "billing", "approved_amount > billed_amount",
         ("approved_amount", "billed_amount"),
         lambda df: _f("approved_amount")(df) > _f("billed_amount")(df), "exclude"),
    Rule("approved_negative", "ERROR", "billing", "approved_amount < 0",
         ("approved_amount",), lambda df: _f("approved_amount")(df) < 0, "exclude"),
    # -- capture floors (Phase 1 finding, Phase 2 quantified) ----------
    Rule("los_at_floor", "INFO", "visit",
         f"length_of_stay_hours pinned at the {LOS_FLOOR_HOURS}h capture floor",
         ("length_of_stay_hours",),
         lambda df: np.isclose(_f("length_of_stay_hours")(df), LOS_FLOOR_HOURS), "flag"),
    Rule("billed_at_floor", "INFO", "billing",
         f"billed_amount pinned at the {BILLED_FLOOR_AMOUNT:.0f} capture floor",
         ("billed_amount",),
         lambda df: np.isclose(_f("billed_amount")(df), BILLED_FLOOR_AMOUNT), "flag"),
    # -- status <-> amount decoupling (Phase 1 finding) ---------------
    Rule("paid_missing_approved", "ERROR", "billing",
         "Paid claim with no approved_amount",
         ("claim_status", "approved_amount"),
         lambda df: df["claim_status"].eq("Paid") & _f("approved_amount")(df).isna(), "impute"),
    Rule("paid_missing_payment_days", "WARN", "billing",
         "Paid claim with no payment_days",
         ("claim_status", "payment_days"),
         lambda df: df["claim_status"].eq("Paid") & _f("payment_days")(df).isna(), "impute"),
    Rule("rejected_with_approved", "ERROR", "billing",
         "Rejected claim that still carries a positive approved_amount",
         ("claim_status", "approved_amount"),
         lambda df: df["claim_status"].eq("Rejected") & (_f("approved_amount")(df) > 0), "review"),
    Rule("pending_with_approved", "INFO", "billing",
         "Pending claim that already carries an approved_amount (pre-adjudication value)",
         ("claim_status", "approved_amount"),
         lambda df: df["claim_status"].eq("Pending") & _f("approved_amount")(df).notna(), "flag"),
    Rule("rejected_with_payment_days", "INFO", "billing",
         "Rejected claim that carries payment_days (payment_days is not tied to Paid)",
         ("claim_status", "payment_days"),
         lambda df: df["claim_status"].eq("Rejected") & _f("payment_days")(df).notna(), "flag"),
    # -- temporal inconsistency (Phase 1 finding) --------------------
    Rule("billing_before_visit", "WARN", "billing+visit",
         "billing_date precedes visit_date - date ordering is unreliable",
         ("billing_date", "visit_date"),
         lambda df: _dt("billing_date")(df) < _dt("visit_date")(df), "keep"),
    Rule("visit_before_registration", "WARN", "visit+patient",
         "visit_date precedes registration_date - registration_date is not a temporal anchor",
         ("visit_date", "registration_date"),
         lambda df: _dt("visit_date")(df) < _dt("registration_date")(df), "keep"),
]

RULES_BY_NAME = {r.name: r for r in RULES}


# ---------------------------------------------------------------------------
# Handling policy — the Phase 2 decision for each finding
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Handling:
    finding: str
    decision: str          # impute | exclude_from_training | keep | derive
    rule: str
    rationale: str
    detail: str = ""


HANDLING: list[Handling] = [
    Handling(
        "length_of_stay_hours == 0.5h floor (1.2% of visits)",
        "keep + flag", "los_at_floor",
        "The 0.5h value is a left-censored capture minimum, not a data error. LOS "
        "carries no measurable signal for either target, so no imputation is "
        "warranted; add a boolean `los_at_floor` flag and keep the row.",
        "Do NOT drop. Do NOT winsorise upward - that would invent stay length.",
    ),
    Handling(
        "billed_amount == 500 floor (1.0% of claims)",
        "keep + flag", "billed_at_floor",
        "500 is a billing-system minimum charge. billed_amount is the strongest "
        "single predictor of claim outcome, so the rows stay; add `billed_at_floor`.",
        "Keep in the mid/low `billed_band`; the flag lets Phase 4 test sensitivity.",
    ),
    Handling(
        "approved_amount missing (~5.3% of all claims, ~MCAR across statuses)",
        "impute for reporting / exclude as a feature", "paid_missing_approved",
        "Missingness is ~5% and independent of claim_status (5.5% Paid, 4.8% "
        "Pending, 5.3% Rejected) - missing-completely-at-random capture loss. For "
        "revenue reporting, impute deterministically from status: Paid -> "
        "billed_amount, Rejected -> 0, Pending -> leave null (genuinely unknown). "
        "approved_amount is a post-adjudication outcome and never enters a model.",
        "Imputation is for the revenue waterfall only, not for modelling.",
    ),
    Handling(
        "payment_days missing (~3.2% of claims) and present on non-Paid claims",
        "impute for reporting / exclude as a feature", "paid_missing_payment_days",
        "payment_days is present for ~97% of claims regardless of status (incl. "
        "Rejected), so it is not a clean 'Paid-only' field. It is a post-outcome "
        "field - excluded from both models. For payment-speed reporting, restrict "
        "to Paid claims with a non-null value.",
        "No model use. Report on Paid & non-null only.",
    ),
    Handling(
        "claim_status <-> approved_amount decoupling",
        "keep, treat as separate fields", "pending_with_approved",
        "95% of Pending claims already carry an approved_amount (a provisional "
        "figure) and the Paid/Rejected approved ratio is deterministic (100% / "
        "0%). The fields describe different moments; never derive one from the "
        "other for adjudicated logic. Revenue math uses the view's "
        "collected_amount / leakage_amount, which already encode the rule.",
        "Analysis only - both fields are post-outcome for modelling.",
    ),
    Handling(
        "Temporal inconsistency (~50% billing-before-visit, ~49% visit-before-registration)",
        "keep, use visit_date only", "billing_before_visit",
        "billing_date, registration_date and visit_date are independently ~uniform "
        "over the same 12-month window - their relative order is noise. `visit_date` "
        "is the ONLY usable temporal key: it anchors all time-based splits and "
        "every as-of feature. billing_lag_days, days-since-registration and any "
        "feature derived from date differences are forbidden.",
        "Phase 3 time-split keys on visit_date. No date-difference features.",
    ),
    Handling(
        "billed_amount / length_of_stay_hours right-tail outliers (IQR: ~1.5% / ~1.0%)",
        "keep (winsorise only inside the model pipeline if needed)",
        "billed_at_floor",
        "The upper tails are smooth and plausible (max billed 88.5k, max LOS "
        "78h) - real variation, not errors. Keep all rows; if a linear model is "
        "sensitive, cap at the 99th percentile inside the Phase 3 ColumnTransformer, "
        "not in the source data.",
        "No source-level clipping.",
    ),
    Handling(
        "33 registered patients with zero visits",
        "keep in patient master / absent from the spine", "n/a",
        "Expected - not every registered patient is seen in the window. They do "
        "not appear in v_visit_billing so they never reach a model; no action.",
    ),
]


# ---------------------------------------------------------------------------
# Reporting / transforms
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Run every applicable rule; return one row per rule with flagged counts."""
    rows = []
    n = len(df)
    for r in RULES:
        if not r.applicable(df):
            rows.append({
                "rule": r.name, "severity": r.severity, "scope": r.scope,
                "handling": r.handling, "records_flagged": pd.NA,
                "records_total": n, "pct_flagged": pd.NA,
                "applicable": False, "description": r.description,
            })
            continue
        mask = r.flag(df)
        cnt = int(mask.sum())
        rows.append({
            "rule": r.name, "severity": r.severity, "scope": r.scope,
            "handling": r.handling, "records_flagged": cnt,
            "records_total": n, "pct_flagged": round(100 * cnt / n, 3) if n else 0.0,
            "applicable": True, "description": r.description,
        })
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    return (pd.DataFrame(rows)
            .sort_values(["severity", "records_flagged"],
                         key=lambda s: s.map(order) if s.name == "severity" else s,
                         ascending=[True, False])
            .reset_index(drop=True))


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a boolean column per applicable rule and a
    combined ``dq_error`` / ``dq_warn`` summary."""
    out = df.copy()
    err_cols, warn_cols = [], []
    for r in RULES:
        if not r.applicable(df):
            continue
        col = f"flag_{r.name}"
        out[col] = r.flag(df)
        if r.severity == "ERROR":
            err_cols.append(col)
        elif r.severity == "WARN":
            warn_cols.append(col)
    out["dq_error"] = out[err_cols].any(axis=1) if err_cols else False
    out["dq_warn"] = out[warn_cols].any(axis=1) if warn_cols else False
    return out


def apply_training_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (kept, excluded) using the ``exclude`` rules only.

    ERROR-severity structural violations (bad enums, impossible numbers) are the
    only rows dropped before modelling. Floors, temporal noise and the
    status/amount decoupling are *kept* per :data:`HANDLING`.
    """
    excl = pd.Series(False, index=df.index)
    for r in RULES:
        if r.handling == "exclude" and r.applicable(df):
            excl |= r.flag(df)
    return df.loc[~excl].copy(), df.loc[excl].copy()


def impute_approved_amount(df: pd.DataFrame) -> pd.Series:
    """Deterministic reporting-only imputation of approved_amount from status."""
    approved = pd.to_numeric(df["approved_amount"], errors="coerce")
    billed = pd.to_numeric(df["billed_amount"], errors="coerce")
    status = df["claim_status"]
    out = approved.copy()
    out = out.mask(out.isna() & status.eq("Paid"), billed)
    out = out.mask(out.isna() & status.eq("Rejected"), 0.0)
    return out


if __name__ == "__main__":  # smoke test against the live spine
    from capstone.db import engine

    eng = engine()
    spine = pd.read_sql("SELECT * FROM v_visit_billing", eng)
    eng.dispose()
    print(validate(spine).to_string(index=False))
