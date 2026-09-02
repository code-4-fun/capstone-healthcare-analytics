"""Phase 2 :: profiling + business analyses (reusable).

Every function returns a tidy DataFrame. :func:`build_all` runs them all and
returns an ordered ``{name: DataFrame}`` dict (no file writes - the caller
decides where CSVs land). Numbers here are the single source of truth for
``PHASE2_FINDINGS.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from capstone.db import engine
from capstone import data_quality as dq
from capstone import features as feat

BILLED_BANDS = [-np.inf, 5000, 15000, 30000, np.inf]
BILLED_LABELS = ["<5k", "5k-15k", "15k-30k", "30k+"]


def load_spine() -> pd.DataFrame:
    eng = engine()
    try:
        df = pd.read_sql("SELECT * FROM v_visit_billing", eng)
    finally:
        eng.dispose()
    for c in ("visit_date", "registration_date", "billing_date"):
        df[c] = pd.to_datetime(df[c])
    df["billed_band"] = pd.cut(df["billed_amount"], BILLED_BANDS, labels=BILLED_LABELS)
    df["approved_ratio"] = df["approved_amount"] / df["billed_amount"]
    return df


# ---------------------------------------------------------------------------
# 1. profiling
# ---------------------------------------------------------------------------
def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        r = {
            "column": c, "dtype": str(s.dtype), "n": n,
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 3),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            d = s.describe()
            r.update(min=round(float(d["min"]), 2), p25=round(float(d["25%"]), 2),
                     median=round(float(d["50%"]), 2), mean=round(float(d["mean"]), 2),
                     p75=round(float(d["75%"]), 2), max=round(float(d["max"]), 2),
                     std=round(float(d["std"]), 2), skew=round(float(s.skew()), 3))
        else:
            vc = s.value_counts(dropna=True)
            if len(vc):
                r["top_value"] = str(vc.index[0])
                r["top_share_pct"] = round(100 * vc.iloc[0] / n, 2)
        rows.append(r)
    return pd.DataFrame(rows)


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["approved_ratio", "billed_band"], errors="ignore")
    m = (df.isna().mean() * 100).round(3).rename("pct_missing").reset_index()
    m.columns = ["column", "pct_missing"]
    m["n_missing"] = df.isna().sum().values
    return m[m["n_missing"] > 0].sort_values("pct_missing", ascending=False).reset_index(drop=True)


def target_balance(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts()
    return pd.DataFrame({
        "class": vc.index, "n": vc.values,
        "share_pct": (100 * vc.values / len(df)).round(2),
    })


# ---------------------------------------------------------------------------
# 2. data-quality findings
# ---------------------------------------------------------------------------
def dq_report(df: pd.DataFrame) -> pd.DataFrame:
    return dq.validate(df)


def floor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    los = df["length_of_stay_hours"]
    billed = df["billed_amount"]
    return pd.DataFrame([
        {"field": "length_of_stay_hours", "floor_value": 0.5,
         "n_at_floor": int(np.isclose(los, 0.5).sum()),
         "pct_at_floor": round(100 * np.isclose(los, 0.5).mean(), 3),
         "next_value": round(float(los[los > 0.5].min()), 3),
         "share_below_1pct_of_range": round(100 * (los <= los.quantile(0.01)).mean(), 2),
         "median": round(float(los.median()), 2)},
        {"field": "billed_amount", "floor_value": 500.0,
         "n_at_floor": int(np.isclose(billed, 500).sum()),
         "pct_at_floor": round(100 * np.isclose(billed, 500).mean(), 3),
         "next_value": round(float(billed[billed > 500].min()), 2),
         "share_below_1pct_of_range": round(100 * (billed <= billed.quantile(0.01)).mean(), 2),
         "median": round(float(billed.median()), 2)},
    ])


def status_vs_approved(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("claim_status", observed=True)
    out = g.agg(
        claims=("bill_id", "size"),
        approved_present=("approved_amount", lambda s: int(s.notna().sum())),
        approved_missing=("approved_amount", lambda s: int(s.isna().sum())),
        payment_days_present=("payment_days", lambda s: int(s.notna().sum())),
    ).reset_index()
    out["approved_missing_pct"] = (100 * out.approved_missing / out.claims).round(2)
    out["payment_days_present_pct"] = (100 * out.payment_days_present / out.claims).round(2)
    ratio = g["approved_ratio"].describe()[["mean", "std", "min", "max"]].round(3).reset_index()
    ratio.columns = ["claim_status", "approved_ratio_mean", "approved_ratio_std",
                     "approved_ratio_min", "approved_ratio_max"]
    return out.merge(ratio, on="claim_status")


def temporal_consistency(df: pd.DataFrame) -> pd.DataFrame:
    lag = (df["billing_date"] - df["visit_date"]).dt.days
    reg_lag = (df["visit_date"] - df["registration_date"]).dt.days
    return pd.DataFrame([
        {"relation": "billing_date < visit_date", "share_pct": round(100 * (lag < 0).mean(), 2),
         "n": int((lag < 0).sum())},
        {"relation": "billing_date == visit_date", "share_pct": round(100 * (lag == 0).mean(), 2),
         "n": int((lag == 0).sum())},
        {"relation": "billing_date > visit_date", "share_pct": round(100 * (lag > 0).mean(), 2),
         "n": int((lag > 0).sum())},
        {"relation": "visit_date < registration_date", "share_pct": round(100 * (reg_lag < 0).mean(), 2),
         "n": int((reg_lag < 0).sum())},
        {"relation": "billing_lag_days mean", "share_pct": round(float(lag.mean()), 2), "n": np.nan},
        {"relation": "billing_lag_days std", "share_pct": round(float(lag.std()), 2), "n": np.nan},
        {"relation": "|billing_lag_days| > 30", "share_pct": round(100 * (lag.abs() > 30).mean(), 2),
         "n": int((lag.abs() > 30).sum())},
    ])


def date_field_monthly(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"month": pd.period_range("2025-01", "2026-01", freq="M").astype(str)})
    for label, col in [("visits_by_visit_date", "visit_date"),
                       ("claims_by_billing_date", "billing_date"),
                       ("registrations_by_registration_date", "registration_date")]:
        c = df[col].dt.to_period("M").astype(str).value_counts()
        out[label] = out["month"].map(c).fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------------
# 3. business analyses
# ---------------------------------------------------------------------------
def flow_monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(m=df["visit_date"].dt.to_period("M").astype(str)).groupby("m")
    out = g.agg(visits=("visit_id", "size"),
               high_risk=("risk_score", lambda s: int((s == "High").sum())),
               unique_patients=("patient_id", "nunique"),
               avg_los=("length_of_stay_hours", "mean")).reset_index()
    out.columns = ["month", "visits", "high_risk_visits", "unique_patients", "avg_los_hours"]
    out["high_risk_pct"] = (100 * out.high_risk_visits / out.visits).round(2)
    out["avg_los_hours"] = out.avg_los_hours.round(2)
    return out


def flow_dow(df: pd.DataFrame) -> pd.DataFrame:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    g = df.assign(d=df["visit_date"].dt.dayofweek).groupby("d")
    out = g.agg(visits=("visit_id", "size"),
               high_risk_pct=("risk_score", lambda s: round(100 * (s == "High").mean(), 2)),
               avg_los_hours=("length_of_stay_hours", lambda s: round(s.mean(), 2))).reset_index()
    out["day"] = out["d"].map(dict(enumerate(names)))
    return out[["day", "visits", "high_risk_pct", "avg_los_hours"]]


def los_drivers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in ["department", "visit_type", "risk_score", "age_band", "chronic_flag", "gender"]:
        for val, sub in df.groupby(dim, observed=True):
            rows.append({"dimension": dim, "value": str(val), "n": len(sub),
                         "avg_los_hours": round(sub.length_of_stay_hours.mean(), 2),
                         "median_los_hours": round(sub.length_of_stay_hours.median(), 2)})
    out = pd.DataFrame(rows)
    out["spread_vs_overall"] = (out.avg_los_hours - df.length_of_stay_hours.mean()).round(2)
    return out


def department_acuity(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("department", observed=True)
    out = g.agg(visits=("visit_id", "size"),
               er_pct=("visit_type", lambda s: round(100 * (s == "ER").mean(), 1)),
               opd_pct=("visit_type", lambda s: round(100 * (s == "OPD").mean(), 1)),
               icu_pct=("visit_type", lambda s: round(100 * (s == "ICU").mean(), 1)),
               high_risk_pct=("risk_score", lambda s: round(100 * (s == "High").mean(), 1)),
               avg_los_hours=("length_of_stay_hours", lambda s: round(s.mean(), 2)),
               total_billed=("billed_amount", "sum")).reset_index()
    return out.sort_values("visits", ascending=False)


def provider_behavior() -> pd.DataFrame:
    eng = engine()
    try:
        return pd.read_sql("SELECT * FROM v_insurance_provider_behavior", eng)
    finally:
        eng.dispose()


def denial_cohort(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("billed_band", observed=True)
    out = g.agg(claims=("bill_id", "size"),
               rejected=("is_rejected", "sum"),
               pending=("is_pending", "sum"),
               paid=("is_paid", "sum"),
               total_billed=("billed_amount", "sum"),
               denied_billed=("billed_amount", lambda s: df.loc[s.index].query("claim_status=='Rejected'").billed_amount.sum())).reset_index()
    out["rejection_rate_pct"] = (100 * out.rejected / out.claims).round(2)
    out["pending_rate_pct"] = (100 * out.pending / out.claims).round(2)
    out["paid_rate_pct"] = (100 * out.paid / out.claims).round(2)
    out["denied_share_of_leakage_pct"] = (100 * out.denied_billed / out.denied_billed.sum()).round(2)
    return out


def rejection_by_dimension(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dims = {"department": "department", "insurance_provider": "insurance_provider",
            "visit_type": "visit_type", "risk_score": "risk_score",
            "age_band": "age_band", "gender": "gender", "chronic_flag": "chronic_flag",
            "billed_band": "billed_band"}
    for label, col in dims.items():
        for val, sub in df.groupby(col, observed=True):
            rows.append({"dimension": label, "value": str(val), "claims": len(sub),
                         "rejection_rate_pct": round(100 * sub.is_rejected.mean(), 2)})
    return pd.DataFrame(rows)


def revenue_waterfall(df: pd.DataFrame) -> pd.DataFrame:
    billed = float(df.billed_amount.sum())
    collected = float(df.collected_amount.sum())
    pending = float(df.pending_amount.sum())
    leakage = float(df.leakage_amount.sum())
    return pd.DataFrame([
        {"component": "Billed", "amount": billed, "pct_of_billed": 100.0},
        {"component": "Collected", "amount": collected, "pct_of_billed": round(100 * collected / billed, 2)},
        {"component": "Pending (at risk)", "amount": pending, "pct_of_billed": round(100 * pending / billed, 2)},
        {"component": "Denied (leakage)", "amount": leakage, "pct_of_billed": round(100 * leakage / billed, 2)},
    ])


def visits_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    vc = df.groupby("patient_id").size()
    dist = vc.value_counts().sort_index().reset_index()
    dist.columns = ["visits_in_window", "n_patients"]
    return dist


# ---------------------------------------------------------------------------
# 4. feature signal strength (mutual information vs each target)
# ---------------------------------------------------------------------------
def feature_signal(fframe: pd.DataFrame) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    rng = 0
    candidate = [c for c in fframe.columns if not c.startswith("target_")
                 and c not in ("visit_id", "patient_id", "visit_date", "visit_month", "day_name")]
    X = pd.DataFrame(index=fframe.index)
    discrete = []
    for c in candidate:
        s = fframe[c]
        is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if not is_numeric:
            X[c] = LabelEncoder().fit_transform(s.astype(str))
            discrete.append(True)
        else:
            num = pd.to_numeric(s, errors="coerce")
            X[c] = num.fillna(num.median() if num.notna().any() else 0)
            discrete.append(int(s.nunique()) < 25)
    allow = {"target_risk_score": set(feat._A_ALLOW), "target_claim_status": set(feat._B_ALLOW)}
    rows = []
    for target, name in [("target_risk_score", "Model A (risk_score)"),
                         ("target_claim_status", "Model B (claim_status)")]:
        yv = fframe[target].astype(str)
        y = LabelEncoder().fit_transform(yv)
        p = yv.value_counts(normalize=True).to_numpy()
        h_target = float(-(p * np.log(p)).sum())  # entropy of the target, nats
        mi = mutual_info_classif(X.values, y, discrete_features=discrete, random_state=rng)
        y_perm = np.random.RandomState(rng).permutation(y)
        mi_perm = mutual_info_classif(X.values, y_perm, discrete_features=discrete, random_state=rng)
        for c, m, mp in zip(candidate, mi, mi_perm):
            rows.append({"model": name, "feature": c,
                         "model_allowed": c in allow[target],
                         "mutual_info": round(float(m), 5),
                         "noise_floor": round(float(mp), 5),
                         "signal_above_noise": round(float(m - mp), 5),
                         "pct_of_target_entropy": round(100 * float(m) / h_target, 3),
                         "noise_pct_of_target_entropy": round(100 * float(mp) / h_target, 3)})
    return pd.DataFrame(rows).sort_values(["model", "mutual_info"], ascending=[True, False])


def feature_catalogue() -> pd.DataFrame:
    return pd.DataFrame([
        {"feature": s["name"], "dtype": s["dtype"], "source": s["source"],
         "definition": s["definition"], "as_of_rule": s["asof"],
         "model_a_verdict": s["model_a"], "model_b_verdict": s["model_b"],
         "notes": s["notes"]}
        for s in feat.FEATURE_SPEC
    ])


# ---------------------------------------------------------------------------
def build_all(spine: pd.DataFrame | None = None,
              fframe: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Run every analysis and return ``{name: DataFrame}`` (no file writes)."""
    spine = load_spine() if spine is None else spine
    fframe = feat.build_feature_frame(spine) if fframe is None else fframe

    tables = {
        "column_profile": column_profile(spine),
        "feature_frame_profile": column_profile(fframe),
        "missingness": missingness(spine),
        "target_balance_risk_score": target_balance(spine, "risk_score"),
        "target_balance_claim_status": target_balance(spine, "claim_status"),
        "data_quality_report": dq_report(spine),
        "floor_analysis": floor_analysis(spine),
        "status_vs_approved": status_vs_approved(spine),
        "temporal_consistency": temporal_consistency(spine),
        "date_field_monthly": date_field_monthly(spine),
        "flow_monthly": flow_monthly(spine),
        "flow_day_of_week": flow_dow(spine),
        "los_drivers": los_drivers(spine),
        "department_acuity": department_acuity(spine),
        "provider_behavior": provider_behavior(),
        "denial_cohort": denial_cohort(spine),
        "rejection_by_dimension": rejection_by_dimension(spine),
        "revenue_waterfall": revenue_waterfall(spine),
        "visits_per_patient": visits_per_patient(spine),
        "feature_signal": feature_signal(fframe),
        "feature_catalogue": feature_catalogue(),
    }
    return tables


def export_tables(tables: dict[str, pd.DataFrame], out_dir) -> None:
    """Write each table to ``<out_dir>/<name>.csv``."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)


if __name__ == "__main__":
    for k, v in build_all().items():
        print(f"\n=== {k} ===")
        print(v.head(12).to_string(index=False))
