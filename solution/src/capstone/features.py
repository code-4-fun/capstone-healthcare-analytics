"""Phase 2 :: leakage-safe feature engineering catalogue.

Builds one feature row per visit from the ``v_visit_billing`` spine. Every
derived field is computed **as of ``visit_date``** - the only temporal key we
trust (Phase 1 finding: billing_date / registration_date ordering is noise).

The catalogue is declared in :data:`FEATURE_SPEC` (also written to
``feature_spec.yaml``) with, per feature: definition, source, as-of rule, dtype
and a **leakage verdict for Model A (visit risk) and Model B (pre-submission
claim outcome)**. :func:`build_feature_frame` returns the computed matrix;
:func:`model_features` filters it to the columns a given model may use.

Hard rule: no post-outcome field (``approved_amount``, ``payment_days``,
``claim_status``, anything derived from ``billing_date``) feeds either model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPINE_SQL = "SELECT * FROM v_visit_billing"

# fields that describe the visit outcome / post-submission adjudication and must
# never be used as inputs for either model
FORBIDDEN_ALWAYS = [
    "approved_amount", "payment_days", "billing_date", "billing_month",
    "billing_lag_days", "collected_amount", "leakage_amount", "pending_amount",
    "is_paid", "is_pending", "is_rejected", "bill_id",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _prior_counts(df: pd.DataFrame, key: str, date: str,
                  flags: dict[str, pd.Series]) -> pd.DataFrame:
    """For every row, aggregate the same-``key`` rows whose ``date`` is strictly
    earlier (as-of the morning of ``date``). Returns prior-count columns and the
    most recent prior date. ``df`` must already be sorted by ``date``."""
    out = pd.DataFrame(index=df.index)
    out["_prior_n"] = 0
    out["_prior_last"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for name in flags:
        out[f"_prior_{name}"] = 0
    flag_arrays = {n: s.to_numpy().astype(int) for n, s in flags.items()}
    dates_all = df[date].to_numpy()

    for _, pos_idx in df.groupby(key, sort=False).indices.items():
        d = dates_all[pos_idx]
        rank = np.searchsorted(d, d, side="left")            # strictly-earlier count
        idx = df.index[pos_idx]
        out.loc[idx, "_prior_n"] = rank
        has_prior = rank > 0
        prev = np.full(len(d), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        prev[has_prior] = d[rank[has_prior] - 1]
        out.loc[idx, "_prior_last"] = prev
        for name, arr in flag_arrays.items():
            cum = np.concatenate([[0], np.cumsum(arr[pos_idx])])
            out.loc[idx, f"_prior_{name}"] = cum[rank]
    return out


def _asof_provider_rate(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Provider historical rejection rate as of ``date`` (strictly earlier visits
    only). ``df`` must already be sorted by ``date``."""
    out = pd.DataFrame(index=df.index, dtype=float)
    out["provider_prior_claim_count"] = 0.0
    out["provider_prior_rejection_rate"] = np.nan
    rej = df["claim_status"].eq("Rejected").to_numpy().astype(int)
    dates_all = df[date].to_numpy()
    for _, pos_idx in df.groupby("insurance_provider", sort=False).indices.items():
        d = dates_all[pos_idx]
        rank = np.searchsorted(d, d, side="left")
        cum_r = np.concatenate([[0], np.cumsum(rej[pos_idx])])
        idx = df.index[pos_idx]
        out.loc[idx, "provider_prior_claim_count"] = rank.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.where(rank > 0, cum_r[rank] / np.where(rank == 0, 1, rank), np.nan)
        out.loc[idx, "provider_prior_rejection_rate"] = rr
    return out


def _doctor_load_30d(df: pd.DataFrame, date: str) -> pd.Series:
    """Count of the visit's doctor's other visits in the trailing 30 days
    (strictly before ``date``). ``df`` must already be sorted by ``date``."""
    load = pd.Series(0, index=df.index, dtype=int)
    dates_all = df[date].to_numpy()
    window = np.timedelta64(30, "D")
    for _, pos_idx in df.groupby("doctor_id", sort=False).indices.items():
        d = dates_all[pos_idx]
        lo = np.searchsorted(d, d - window, side="left")
        hi = np.searchsorted(d, d, side="left")
        load.iloc[pos_idx] = hi - lo
    return load


def _billed_band(amount: pd.Series) -> pd.Categorical:
    return pd.cut(amount, [-np.inf, 5000, 15000, 30000, np.inf],
                 labels=["<5k", "5k-15k", "15k-30k", "30k+"])


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------
def build_feature_frame(spine: pd.DataFrame | None = None) -> pd.DataFrame:
    if spine is None:
        from capstone.db import engine
        eng = engine()
        spine = pd.read_sql(SPINE_SQL, eng)
        eng.dispose()

    df = spine.copy()
    for c in ("visit_date", "registration_date", "billing_date"):
        df[c] = pd.to_datetime(df[c])
    df = df.sort_values(["visit_date", "visit_id"]).reset_index(drop=True)

    f = pd.DataFrame(index=df.index)
    # -- keys & targets (carried, not features) --------------------------
    f["visit_id"] = df["visit_id"]
    f["patient_id"] = df["patient_id"]
    f["visit_date"] = df["visit_date"]
    f["target_risk_score"] = df["risk_score"]
    f["target_claim_status"] = df["claim_status"]
    f["target_is_high_risk"] = df["risk_score"].eq("High")
    f["target_is_rejected"] = df["claim_status"].eq("Rejected")

    # -- patient history as-of visit_date -------------------------------
    prior = _prior_counts(df, "patient_id", "visit_date", {
        "high_risk": df["risk_score"].eq("High"),
        "rejection": df["claim_status"].eq("Rejected"),
    })
    f["prior_visit_count"] = prior["_prior_n"].astype(int)
    f["prior_high_risk_count"] = prior["_prior_high_risk"].astype(int)
    f["prior_rejection_count"] = prior["_prior_rejection"].astype(int)
    f["is_first_visit"] = f["prior_visit_count"].eq(0)
    dsl = (df["visit_date"] - prior["_prior_last"]).dt.days
    f["days_since_last_visit"] = dsl  # NaN for first visit
    f["has_prior_visit"] = ~f["is_first_visit"]
    f["prior_rejection_rate"] = np.where(
        f["prior_visit_count"] > 0,
        f["prior_rejection_count"] / f["prior_visit_count"].replace(0, np.nan),
        np.nan,
    )

    # -- visit context ------------------------------------------------
    f["department"] = df["department"]
    f["visit_type"] = df["visit_type"]
    f["age"] = df["age"]
    f["age_band"] = df["age_band"]
    f["gender"] = df["gender"]
    f["city"] = df["city"]
    f["chronic_flag"] = df["chronic_flag"].astype(int)
    f["doctor_id"] = df["doctor_id"]
    f["doctor_load_30d"] = _doctor_load_30d(df, "visit_date")
    f["length_of_stay_hours"] = df["length_of_stay_hours"]
    f["los_at_floor"] = np.isclose(df["length_of_stay_hours"], 0.5)
    # risk_score is the Model A label but a legitimate pre-submission input for
    # Model B (it is assigned at the visit, before the claim is filed).
    f["risk_score"] = df["risk_score"]

    # -- billing context available pre-submission --------------------
    f["billed_amount"] = df["billed_amount"]
    f["log_billed_amount"] = np.log1p(df["billed_amount"])
    f["billed_band"] = _billed_band(df["billed_amount"]).astype(str)
    f["billed_at_floor"] = np.isclose(df["billed_amount"], 500.0)
    f["insurance_provider"] = df["insurance_provider"]
    prate = _asof_provider_rate(df, "visit_date")
    f["provider_prior_claim_count"] = prate["provider_prior_claim_count"]
    f["provider_prior_rejection_rate"] = prate["provider_prior_rejection_rate"]

    # -- seasonality (all derived from visit_date) -------------------
    vd = df["visit_date"].dt
    f["visit_month"] = df["visit_date"].dt.to_period("M").astype(str)
    f["month"] = vd.month
    f["day_of_week"] = vd.dayofweek
    f["day_name"] = vd.day_name()
    f["is_weekend"] = vd.dayofweek.ge(5)
    f["week_of_year"] = vd.isocalendar().week.astype(int)
    f["quarter"] = vd.quarter
    f["day_of_year"] = vd.dayofyear

    return f


# ---------------------------------------------------------------------------
# feature spec / leakage register
# ---------------------------------------------------------------------------
# verdict values: "allow" | "exclude" | "target"
FEATURE_SPEC: list[dict] = [
    # name, dtype, source, definition, asof, model_a, model_b, notes
    dict(name="prior_visit_count", dtype="int", source="visits (self-join on patient_id)",
         definition="Number of the patient's earlier visits.",
         asof="visits with visit_date strictly < this visit_date",
         model_a="allow", model_b="allow",
         notes="Uses only the existence of prior visits, not their outcomes."),
    dict(name="prior_high_risk_count", dtype="int", source="visits.risk_score of prior visits",
         definition="Count of the patient's prior visits scored High risk.",
         asof="prior visits only (visit_date <)",
         model_a="allow", model_b="allow",
         notes="risk_score of *past* visits is history, not the current label."),
    dict(name="prior_rejection_count", dtype="int", source="billing.claim_status of prior visits",
         definition="Count of the patient's prior claims that were Rejected.",
         asof="prior visits only (visit_date <)",
         model_a="allow", model_b="allow",
         notes="Assumes prior claims are adjudicated by now; claim_status of the "
               "CURRENT visit is never used."),
    dict(name="prior_rejection_rate", dtype="float", source="derived",
         definition="prior_rejection_count / prior_visit_count (NaN if no prior visit).",
         asof="prior visits only", model_a="allow", model_b="allow",
         notes="Patient-level denial propensity."),
    dict(name="days_since_last_visit", dtype="float", source="visits",
         definition="Days between this visit_date and the patient's previous visit_date.",
         asof="prior visits only", model_a="allow", model_b="allow",
         notes="NaN for first visit - impute with a large sentinel or a missing flag."),
    dict(name="is_first_visit", dtype="bool", source="derived",
         definition="True when the patient has no earlier visit.",
         asof="prior visits only", model_a="allow", model_b="allow", notes=""),
    dict(name="has_prior_visit", dtype="bool", source="derived",
         definition="Negation of is_first_visit.",
         asof="prior visits only", model_a="allow", model_b="allow", notes=""),
    dict(name="department", dtype="category", source="visits.department",
         definition="Clinical department of the visit.",
         asof="known at visit time", model_a="allow", model_b="allow", notes=""),
    dict(name="visit_type", dtype="category", source="visits.visit_type",
         definition="ER / OPD / ICU.",
         asof="known at visit time", model_a="allow", model_b="allow", notes=""),
    dict(name="age", dtype="int", source="patients.age",
         definition="Patient age in years.",
         asof="static patient attribute", model_a="allow", model_b="allow", notes=""),
    dict(name="age_band", dtype="category", source="derived from patients.age",
         definition="0-17 / 18-34 / 35-49 / 50-64 / 65+.",
         asof="static", model_a="allow", model_b="allow", notes=""),
    dict(name="gender", dtype="category", source="patients.gender",
         definition="Patient gender (M/F).",
         asof="static", model_a="allow", model_b="allow",
         notes="Monitor for fairness in Phase 4; not a driver in EDA."),
    dict(name="city", dtype="category", source="patients.city",
         definition="Patient home city (network has 6).",
         asof="static", model_a="allow", model_b="allow", notes=""),
    dict(name="chronic_flag", dtype="int", source="patients.chronic_flag",
         definition="1 if the patient is flagged chronic.",
         asof="static (assumed known at visit)", model_a="allow", model_b="allow", notes=""),
    dict(name="doctor_id", dtype="category", source="visits.doctor_id",
         definition="Attending doctor identifier (101 doctors).",
         asof="known at visit time", model_a="allow", model_b="allow",
         notes="High cardinality - target/impact encode or drop in Phase 3."),
    dict(name="doctor_load_30d", dtype="int", source="derived from visits",
         definition="Count of the attending doctor's other visits in the prior 30 days.",
         asof="visits with visit_date in [D-30, D-1]", model_a="allow", model_b="allow",
         notes="Operational load proxy."),
    dict(name="length_of_stay_hours", dtype="float", source="visits.length_of_stay_hours",
         definition="Recorded length of stay in hours (floored at 0.5h).",
         asof="known at/after discharge - BEFORE claim submission",
         model_a="exclude", model_b="allow",
         notes="Model A predicts risk at/around admission, so LOS is not yet known "
               "and is excluded. It is known before the claim is filed, so Model B "
               "may use it."),
    dict(name="los_at_floor", dtype="bool", source="derived",
         definition="True when length_of_stay_hours == 0.5 (capture floor).",
         asof="same as length_of_stay_hours",
         model_a="exclude", model_b="allow", notes="Pairs with length_of_stay_hours."),
    dict(name="billed_amount", dtype="float", source="billing.billed_amount",
         definition="Amount billed on the claim.",
         asof="set when the claim is prepared - BEFORE submission/adjudication",
         model_a="exclude", model_b="allow",
         notes="Not an operational/clinical field, so excluded from Model A. It is "
               "the single strongest predictor of claim outcome for Model B."),
    dict(name="log_billed_amount", dtype="float", source="derived",
         definition="log1p(billed_amount).",
         asof="same as billed_amount", model_a="exclude", model_b="allow", notes=""),
    dict(name="billed_band", dtype="category", source="derived from billing.billed_amount",
         definition="<5k / 5k-15k / 15k-30k / 30k+.",
         asof="same as billed_amount", model_a="exclude", model_b="allow",
         notes="Captures the non-monotonic rejection pattern."),
    dict(name="billed_at_floor", dtype="bool", source="derived",
         definition="True when billed_amount == 500 (capture floor).",
         asof="same as billed_amount", model_a="exclude", model_b="allow", notes=""),
    dict(name="insurance_provider", dtype="category", source="patients.insurance_provider",
         definition="Claim's insurer (4 providers).",
         asof="static patient attribute", model_a="allow", model_b="allow",
         notes="Known pre-submission; behaviour is near-identical across the 4 in EDA."),
    dict(name="provider_prior_claim_count", dtype="float", source="derived from billing",
         definition="Number of the insurer's claims from earlier visits.",
         asof="visits with visit_date strictly <", model_a="allow", model_b="allow", notes=""),
    dict(name="provider_prior_rejection_rate", dtype="float", source="derived from billing",
         definition="Rejected / total for the insurer over earlier visits (NaN before any).",
         asof="visits with visit_date strictly <", model_a="allow", model_b="allow",
         notes="As-of denial rate of the counterparty; uses only resolved past claims."),
    dict(name="visit_month", dtype="category", source="derived from visits.visit_date",
         definition="Calendar month of the visit (YYYY-MM).",
         asof="the visit_date itself", model_a="allow", model_b="allow", notes=""),
    dict(name="month", dtype="int", source="derived from visits.visit_date",
         definition="Month number 1-12.",
         asof="the visit_date itself", model_a="allow", model_b="allow", notes=""),
    dict(name="day_of_week", dtype="int", source="derived from visits.visit_date",
         definition="0=Mon .. 6=Sun.",
         asof="the visit_date itself", model_a="allow", model_b="allow", notes=""),
    dict(name="day_name", dtype="category", source="derived from visits.visit_date",
         definition="Weekday name.",
         asof="the visit_date itself", model_a="allow", model_b="allow", notes=""),
    dict(name="is_weekend", dtype="bool", source="derived from visits.visit_date",
         definition="True for Sat/Sun.",
         asof="the visit_date itself", model_a="allow", model_b="allow", notes=""),
    dict(name="week_of_year", dtype="int", source="derived from visits.visit_date",
         definition="ISO week number 1-53.",
         asof="the visit_date itself", model_a="allow", model_b="allow", notes=""),
    dict(name="quarter", dtype="int", source="derived from visits.visit_date",
         definition="Calendar quarter 1-4.",
         asof="the visit_date itself", model_a="allow", model_b="allow", notes=""),
    dict(name="day_of_year", dtype="int", source="derived from visits.visit_date",
         definition="Ordinal day 1-366.",
         asof="the visit_date itself", model_a="allow", model_b="allow", notes=""),
    # -- explicitly forbidden ------------------------------------------
    dict(name="risk_score", dtype="category", source="visits.risk_score",
         definition="Clinical risk band assigned to the visit.",
         asof="POST visit assessment", model_a="target", model_b="allow",
         notes="Target of Model A. Known before the claim is filed, so Model B may "
               "use it as an input."),
    dict(name="claim_status", dtype="category", source="billing.claim_status",
         definition="Adjudication outcome Paid/Pending/Rejected.",
         asof="POST adjudication", model_a="exclude", model_b="target",
         notes="Target of Model B. Post-outcome for Model A."),
    dict(name="approved_amount", dtype="float", source="billing.approved_amount",
         definition="Amount the insurer approved.",
         asof="POST adjudication", model_a="exclude", model_b="exclude",
         notes="Post-outcome. Forbidden for BOTH models. Reporting-only, "
               "deterministically imputable from claim_status."),
    dict(name="payment_days", dtype="float", source="billing.payment_days",
         definition="Days from billing to payment.",
         asof="POST payment", model_a="exclude", model_b="exclude",
         notes="Post-outcome. Forbidden for BOTH models. Present on ~97% of claims "
               "of every status, so not even a clean Paid indicator."),
    dict(name="billing_date", dtype="date", source="billing.billing_date",
         definition="Recorded claim date.",
         asof="unreliable - ~50% precede visit_date",
         model_a="exclude", model_b="exclude",
         notes="Ordering is noise. No feature may be derived from it, including "
               "billing_lag_days."),
    dict(name="registration_date", dtype="date", source="patients.registration_date",
         definition="Recorded patient registration date.",
         asof="unreliable - ~49% follow the visit",
         model_a="exclude", model_b="exclude",
         notes="Not a temporal anchor; days-since-registration is forbidden."),
    dict(name="billing_lag_days", dtype="int", source="derived (billing_date - visit_date)",
         definition="Day difference billing_date - visit_date.",
         asof="unreliable", model_a="exclude", model_b="exclude",
         notes="Derived from billing_date; pure noise (mean ~1d, sd ~150d)."),
    dict(name="collected_amount", dtype="float", source="v_visit_billing derived",
         definition="Cash collected (approved value on Paid claims).",
         asof="POST adjudication", model_a="exclude", model_b="exclude",
         notes="Post-outcome. Reporting only."),
    dict(name="leakage_amount", dtype="float", source="v_visit_billing derived",
         definition="Billed minus approved on adjudicated claims.",
         asof="POST adjudication", model_a="exclude", model_b="exclude", notes="Reporting only."),
    dict(name="pending_amount", dtype="float", source="v_visit_billing derived",
         definition="Billed value of Pending claims.",
         asof="POST adjudication", model_a="exclude", model_b="exclude", notes="Reporting only."),
]

_A_ALLOW = [s["name"] for s in FEATURE_SPEC if s["model_a"] == "allow"]
_B_ALLOW = [s["name"] for s in FEATURE_SPEC if s["model_b"] == "allow"]


def model_features(f: pd.DataFrame, model: str) -> list[str]:
    allow = {"A": _A_ALLOW, "B": _B_ALLOW}[model.upper()]
    return [c for c in f.columns if c in allow]


# post-outcome / not-yet-known fields that must never reach a model
_FORBIDDEN = {
    "approved_amount", "payment_days", "claim_status", "billing_date", "billing_month",
    "billing_lag_days", "collected_amount", "leakage_amount", "pending_amount",
    "is_paid", "is_pending", "is_rejected", "registration_date",
}


def leakage_violations(f: pd.DataFrame) -> list[str]:
    """Return a list of leakage-rule violations for the eligible feature sets of
    both models. Empty list == clean. Phase 3 calls this before training."""
    a, b = set(model_features(f, "A")), set(model_features(f, "B"))
    problems = []
    for name, cols in (("Model A", a), ("Model B", b)):
        bad = sorted(cols & _FORBIDDEN)
        if bad:
            problems.append(f"{name} exposes forbidden field(s): {bad}")
    if "risk_score" in a:
        problems.append("Model A exposes risk_score (its own target)")
    for field in ("length_of_stay_hours", "billed_amount", "billed_band", "log_billed_amount"):
        if field in a:
            problems.append(f"Model A exposes {field} (not known at admission / not operational)")
    return problems


def write_feature_spec_yaml(path: str | Path) -> Path:
    import yaml

    path = Path(path)
    doc = {
        "meta": {
            "phase": 2,
            "spine": "capstone_solution.v_visit_billing",
            "temporal_key": "visit_date",
            "grain": "one row per visit",
            "models": {
                "A": "visit risk (target: risk_score Low/Medium/High)",
                "B": "pre-submission claim outcome (target: claim_status Paid/Pending/Rejected)",
            },
            "verdicts": {
                "allow": "feature may enter this model",
                "exclude": "must not enter this model (leakage or not-yet-known)",
                "target": "this field is the model's label",
            },
            "hard_rule": ("no post-outcome field (approved_amount, payment_days, "
                          "claim_status, billing_date derivatives) feeds Model A or "
                          "the pre-submission Model B"),
        },
        "features": [
            {
                "name": s["name"],
                "dtype": s["dtype"],
                "definition": s["definition"],
                "source": s["source"],
                "as_of_rule": s["asof"],
                "leakage_verdict": {"model_a": s["model_a"], "model_b": s["model_b"]},
                "notes": s["notes"],
            }
            for s in FEATURE_SPEC
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, width=100))
    return path


if __name__ == "__main__":
    frame = build_feature_frame()
    print(frame.shape)
    print("\nModel A features:", model_features(frame, "A"))
    print("\nModel B features:", model_features(frame, "B"))
    print("\nwrote", write_feature_spec_yaml("feature_spec.yaml"))
