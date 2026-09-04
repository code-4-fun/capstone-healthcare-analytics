"""Phase 6 :: reusable monitoring, drift-detection and governance logic.

Phase 5's FastAPI service appends every prediction (request payload, response,
model / feature versions, latency) to ``capstone_solution.prediction_log``. This
module turns that stream into a monitored signal:

* :func:`validation_gate` - a batch data-quality gate over served request
  payloads, reusing the Phase 2 rule registry (:mod:`capstone.data_quality`).
* :func:`psi` / :func:`ks` / :func:`feature_drift` - distribution drift of the
  served model inputs against the Phase 3 **test**-window reference
  (:func:`reference_frame`; see :data:`REFERENCE_SPLIT` for why the test split,
  not the training split).
* :func:`prediction_drift` - drift of the predicted-class mix.
* :func:`performance_drift` - once real outcomes are joined back, recall on the
  costly class and net recoverable leakage vs the Phase 4 operating baseline.
* :func:`run_drift` - the single call the scheduled job
  (``phase6_monitoring/drift_job.py``) and the notebook share: pull a window
  from ``prediction_log``, rebuild the served feature rows with
  :func:`capstone.serving.build_model_row`, compute every metric, and return a
  :class:`DriftResult` whose rows :func:`write_drift_report` persists to
  ``capstone_solution.drift_report``.

Leakage discipline is inherited: the monitored feature set for each model is
exactly its Phase 3 manifest feature list, so Model A is never compared on a
billing / LOS / ``risk_score`` distribution.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from capstone import data_quality as dq
from capstone import modeling as M
from capstone import serving as S

log = logging.getLogger("capstone.monitoring")

# --------------------------------------------------------------------------
# versioning + thresholds
# --------------------------------------------------------------------------

MONITORING_VERSION = "1.0.0"

#: PSI interpretation bands (industry-standard cut points).
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25

#: What turns a metric into an alert. Kept in one place so the notebook, the
#: job and ``governance.md`` cite the same numbers.
ALERT_RULES: dict[str, Any] = {
    "feature_psi": PSI_SIGNIFICANT,          # any monitored feature PSI above this
    "prediction_psi": 0.20,                  # predicted-class-mix PSI above this
    "recall_costly_drop_pts": 0.10,          # recall on the costly class falls this many points
    "gate_fail_rate": 0.01,                  # >1% of served requests fail the validation gate
}

#: Phase 4 operating baseline for Model B (phase4_eval/model_card_B.md,
#: PHASE4_FINDINGS.md - test window, threshold P(Rejected) >= 0.19).
PHASE4_BASELINE: dict[str, dict[str, float]] = {
    "B": {
        "operating_threshold": 0.19,
        "recall_costly": 0.62,               # recall on Rejected (377/608)
        "precision_costly": 0.22,
        "net_recovered_per_month": 250_000.0,  # ~Rs 5.0L net over the 2-month test window
        "alerts_per_month": 848.0,
    },
    "A": {
        # Model A ships as a base-rate monitor: it predicts "Low" for every
        # visit, so its only meaningful signal is the observed risk-mix prior.
        "recall_costly": 0.0,
    },
}

COSTLY_CLASS = M.COSTLY_CLASS            # {"A": "High", "B": "Rejected"}
CLASS_ORDER = M.CLASS_ORDER

_SEED_TAGS = ("phase6-seed:baseline", "phase6-seed:drift")

_HERE = Path(__file__).resolve().parents[2]
PHASE2_PARQUET = _HERE / "phase2_eda" / "output" / "feature_frame.parquet"
DRIFT_DDL = _HERE / "phase6_monitoring" / "sql" / "drift_report.sql"
PREDICTION_LOG_DDL = _HERE / "phase5_api" / "sql" / "prediction_log.sql"


# --------------------------------------------------------------------------
# reference window
# --------------------------------------------------------------------------

#: Which Phase 3 split is the drift reference. The **test** window (the last
#: two months of the pilot year) is used, not the training window: it is the
#: most recent distribution the models were validated against (Phase 4), and it
#: immediately precedes "production", whereas the training window's calendar
#: features (``month``, ``week_of_year``, ``day_of_year``) are disjoint from any
#: later window by construction and would swamp every real signal.
REFERENCE_SPLIT = "test"


def reference_frame(parquet: str | Path | None = None, *, split: str = REFERENCE_SPLIT) -> pd.DataFrame:
    """The Phase 3 reference split of the Phase 2 feature frame (see
    :data:`REFERENCE_SPLIT`). Same calendar-anchored ``visit_date`` split as
    Phase 3."""
    parquet = Path(parquet) if parquet else PHASE2_PARQUET
    df = pd.read_parquet(parquet)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    return getattr(M.time_split(df), split).reset_index(drop=True)


def reference_label(ref: pd.DataFrame) -> str:
    lo, hi = ref["visit_date"].min().date(), ref["visit_date"].max().date()
    return f"phase3-{REFERENCE_SPLIT} {lo.isoformat()}..{hi.isoformat()}"


def monitored_features(bundle: S.ServingBundle, model: str) -> tuple[list[str], list[str]]:
    """``(numeric, categorical)`` monitored for ``model`` - exactly its Phase 3
    manifest feature list, so the monitored set == the trained set."""
    fl = bundle.feature_lists(model)
    return list(fl["numeric"]), list(fl["categorical"])


def feature_row_to_payload(row: pd.Series, model: str) -> dict[str, Any]:
    """A request payload from one feature-frame row - the business fields plus
    the real as-of history and the ``visit_id`` correlation key. Shared by the
    traffic seeder and the reference assembly so both go through the identical
    :func:`capstone.serving.build_model_row` path."""
    p: dict[str, Any] = {
        "visit_id": int(row["visit_id"]),
        "visit_date": pd.Timestamp(row["visit_date"]).date().isoformat(),
        "department": str(row["department"]),
        "visit_type": str(row["visit_type"]),
        "age": int(row["age"]),
        "gender": str(row["gender"]),
        "city": str(row["city"]),
        "insurance_provider": str(row["insurance_provider"]),
        "chronic_flag": bool(row["chronic_flag"]),
    }
    for f in S.HISTORY_FIELDS:
        v = row.get(f)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            p[f] = float(v)
    if model == "B":
        p["billed_amount"] = float(row["billed_amount"])
        p["length_of_stay_hours"] = float(row["length_of_stay_hours"])
        p["risk_score"] = str(row["risk_score"])
    return p


def assemble_frame(payloads, bundle: S.ServingBundle, model: str) -> pd.DataFrame:
    """Stack :func:`capstone.serving.build_model_row` over an iterable of request
    payloads - the exact model-input representation, used for both the current
    window and the reference so drift is measured like-for-like."""
    fl = bundle.feature_lists(model)
    frames = [S.build_model_row(p, model, fl)[0] for p in payloads]
    if not frames:
        return pd.DataFrame(columns=fl["numeric"] + fl["categorical"])
    return pd.concat(frames, ignore_index=True)


_REFERENCE_SERVED: dict[tuple[str, int], pd.DataFrame] = {}


def reference_served(bundle: S.ServingBundle, model: str,
                     reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The reference window assembled through the serving feature path.

    Cached per ``(model, frame)`` for the process - assembling ~4k rows through
    :func:`capstone.serving.build_model_row` is the slow step, and the reference
    is stable within a run.
    """
    key = (model, id(reference_df) if reference_df is not None else 0)
    if key in _REFERENCE_SERVED:
        return _REFERENCE_SERVED[key]
    ref = reference_frame() if reference_df is None else reference_df
    served = assemble_frame(
        (feature_row_to_payload(r, model) for _, r in ref.iterrows()), bundle, model)
    _REFERENCE_SERVED[key] = served
    return served


# --------------------------------------------------------------------------
# distribution drift primitives
# --------------------------------------------------------------------------

def _psi_from_counts(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI from two aligned bucket-count vectors. Laplace-smoothed (+0.5 per
    bucket) so an empty bucket contributes a bounded term instead of blowing the
    log ratio up."""
    e = np.asarray(expected, dtype=float) + 0.5
    a = np.asarray(actual, dtype=float) + 0.5
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi(reference: pd.Series, current: pd.Series, *, bins: int = 10) -> float:
    """Population Stability Index of ``current`` against ``reference``.

    Numeric series are bucketed on ``reference`` quantiles (deciles by default);
    categorical / low-cardinality series are compared on value shares.
    """
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return float("nan")

    numeric = pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref)
    if numeric and ref.nunique() > bins:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(ref, bins=edges)[0].astype(float)
        a = np.histogram(cur, bins=edges)[0].astype(float)
        return _psi_from_counts(e, a)

    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    e = ref.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    a = cur.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    return _psi_from_counts(e, a)


def ks(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value (numeric only)."""
    from scipy.stats import ks_2samp

    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return float("nan"), float("nan")
    res = ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def psi_band(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                  numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """One row per monitored feature: PSI, KS (numeric), interpretation band and
    a ``drifted`` flag (PSI in the significant band)."""
    rows = []
    for col in numeric + categorical:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        kind = "numeric" if col in numeric else "categorical"
        p = psi(reference_df[col], current_df[col])
        ks_stat, ks_p = ks(reference_df[col], current_df[col]) if kind == "numeric" else (np.nan, np.nan)
        rows.append({
            "feature": col, "kind": kind, "psi": p,
            "ks_stat": ks_stat, "ks_pvalue": ks_p,
            "band": psi_band(p), "drifted": psi_band(p) == "significant",
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def prediction_drift(reference_labels: pd.Series, current_labels: pd.Series,
                     classes: list[str]) -> pd.DataFrame:
    """Predicted-class share table (reference vs current) with the class-mix PSI
    repeated on every row for convenience."""
    ref_n = reference_labels.astype(str).value_counts().reindex(classes, fill_value=0)
    cur_n = current_labels.astype(str).value_counts().reindex(classes, fill_value=0)
    mix_psi = _psi_from_counts(ref_n.to_numpy(), cur_n.to_numpy())
    ref = ref_n / ref_n.sum()
    cur = cur_n / max(cur_n.sum(), 1)
    return pd.DataFrame({
        "predicted_class": classes,
        "reference_share": ref.to_numpy(),
        "current_share": cur.to_numpy(),
        "share_delta": (cur - ref).to_numpy(),
        "mix_psi": mix_psi,
    })


# --------------------------------------------------------------------------
# validation gate  (reuses the Phase 2 rule registry)
# --------------------------------------------------------------------------

#: Rules from :data:`capstone.data_quality.RULES` that are checkable on a served
#: request payload (enum domains + the numeric ranges known before submission).
GATE_RULES: tuple[str, ...] = (
    "gender_enum", "visit_type_enum", "risk_score_enum", "department_enum",
    "city_enum", "provider_enum", "age_range", "billed_negative", "los_negative",
)


@dataclass
class GateResult:
    n: int
    offences: pd.DataFrame            # rule, severity, description, n_offending
    offending_ids: list[str]
    fail_rate: float

    @property
    def passed(self) -> bool:
        return self.fail_rate <= ALERT_RULES["gate_fail_rate"]


def validation_gate(payloads: pd.DataFrame, *, id_col: str = "request_id") -> GateResult:
    """Run the request-time subset of the Phase 2 data-quality rules over a
    batch of served request payloads."""
    df = payloads.copy()
    ids = df[id_col].astype(str) if id_col in df.columns else pd.Series(range(len(df)), dtype=str)

    rows, offending = [], set()
    for name in GATE_RULES:
        rule = dq.RULES_BY_NAME[name]
        if not rule.applicable(df):
            continue
        mask = rule.flag(df)
        n_bad = int(mask.sum())
        if n_bad:
            offending.update(ids[mask.to_numpy()].tolist())
        rows.append({
            "rule": name, "severity": rule.severity,
            "description": rule.description, "n_offending": n_bad,
        })
    offences = pd.DataFrame(rows)
    fail_rate = len(offending) / len(df) if len(df) else 0.0
    return GateResult(n=len(df), offences=offences,
                      offending_ids=sorted(offending), fail_rate=fail_rate)


# --------------------------------------------------------------------------
# performance drift  (needs actual outcomes joined back)
# --------------------------------------------------------------------------

def performance_drift(pred_df: pd.DataFrame, *, model: str,
                      threshold: float | None = None) -> dict[str, Any]:
    """Recall on the costly class and net recoverable leakage on a window whose
    rows carry ``actual`` outcomes, compared to the Phase 4 baseline.

    ``pred_df`` columns: ``actual``, ``prob_<class>`` (calibrated),
    ``leakage_amount``, ``visit_date``. For Model A (a base-rate monitor) only
    the risk-mix recall is reported.
    """
    from capstone import evaluation as E

    costly = COSTLY_CLASS[model]
    base = PHASE4_BASELINE[model]
    out: dict[str, Any] = {"model": model, "n": int(len(pred_df)), "costly_class": costly}

    if model == "B":
        thr = float(threshold if threshold is not None else base["operating_threshold"])
        bs = E.business_summary(pred_df, thr)
        recall = bs["recall"]
        out.update({
            "operating_threshold": thr,
            "recall_costly": recall,
            "recall_costly_baseline": base["recall_costly"],
            "recall_costly_delta": recall - base["recall_costly"],
            "precision_costly": bs["precision"],
            "net_recovered_per_month": bs["net_recovered"] / max(bs["months_in_window"], 1e-9),
            "net_recovered_baseline_per_month": base["net_recovered_per_month"],
            "alerts_per_month": bs["alerts_per_month"],
            "recall_breach": (base["recall_costly"] - recall) > ALERT_RULES["recall_costly_drop_pts"],
        })
    else:
        actual = pred_df["actual"].astype(str)
        out.update({
            "risk_high_share": float((actual == costly).mean()),
            "predicted_high_share": 0.0,     # Model A never predicts High
            "recall_costly": 0.0,
            "recall_costly_baseline": base["recall_costly"],
            "recall_costly_delta": 0.0,
            "recall_breach": False,
        })
    return out


# --------------------------------------------------------------------------
# prediction_log window -> served feature rows
# --------------------------------------------------------------------------

_WINDOW_TAGS = {"baseline": "phase6-seed:baseline", "drift": "phase6-seed:drift"}


def load_window(conn, *, model: str, window: str) -> pd.DataFrame:
    """Pull one window of ``prediction_log`` rows for ``model``.

    ``window`` is ``"baseline"`` / ``"drift"`` (the seeded demo tags),
    ``"last-week"``/``"last-day"``, or an ISO range ``"<start>..<end>"``.
    """
    sql = [
        "SELECT request_id::text, ts, model, model_version, predicted_class,",
        "       probabilities, request_payload, latency_ms, client_host",
        "FROM capstone_solution.prediction_log WHERE model = %(model)s",
    ]
    params: dict[str, Any] = {"model": model}
    if window in _WINDOW_TAGS:
        sql.append("AND client_host = %(tag)s")
        params["tag"] = _WINDOW_TAGS[window]
    elif window == "last-week":
        sql.append("AND ts >= now() - interval '7 days'")
    elif window == "last-day":
        sql.append("AND ts >= now() - interval '1 day'")
    elif ".." in window:
        start, end = window.split("..", 1)
        sql.append("AND ts >= %(start)s AND ts < %(end)s")
        params.update(start=start, end=end)
    else:
        raise ValueError(f"unrecognised window spec: {window!r}")
    sql.append("ORDER BY ts")

    with conn.cursor() as cur:
        cur.execute("\n".join(sql), params)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    for c in ("probabilities", "request_payload"):
        if c in df.columns:
            df[c] = df[c].apply(lambda v: v if isinstance(v, dict) or v is None else json.loads(v))
    return df


def assemble_served_rows(window_df: pd.DataFrame, bundle: S.ServingBundle,
                         model: str) -> pd.DataFrame:
    """Rebuild the exact model-input frame for every logged request via
    :func:`capstone.serving.build_model_row` - so drift is measured on what the
    model actually saw, using the same assembly the API used."""
    return assemble_frame(list(window_df["request_payload"]), bundle, model)


def _payload_visit_dates(window_df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(window_df["request_payload"].apply(lambda p: p.get("visit_date")))


def join_actuals(window_df: pd.DataFrame, spine: pd.DataFrame, model: str) -> pd.DataFrame:
    """Build the ``pred_df`` :func:`performance_drift` needs by joining each
    logged prediction to its real outcome on the caller-supplied ``visit_id``.

    Returns an empty frame when the window carries no ``visit_id`` correlation
    key (real production traffic before outcomes have landed)."""
    vids = window_df["request_payload"].apply(lambda p: p.get("visit_id"))
    if vids.isna().all():
        return pd.DataFrame()

    classes = CLASS_ORDER[model]
    target = "claim_status" if model == "B" else "risk_score"
    money_cols = [c for c in ("leakage_amount", "billed_amount") if c in spine.columns]
    sp = spine[["visit_id", target, *money_cols]].copy()

    df = pd.DataFrame({
        "visit_id": vids.to_numpy(),
        "visit_date": _payload_visit_dates(window_df).to_numpy(),
        "pred_pipeline": window_df["predicted_class"].to_numpy(),
    })
    for cls in classes:
        df[f"prob_{cls.lower()}"] = window_df["probabilities"].apply(
            lambda d, c=cls: float(d.get(c, 0.0)))
    df = df.merge(sp, on="visit_id", how="inner").rename(columns={target: "actual"})
    df["actual"] = df["actual"].astype(str)
    if "leakage_amount" not in df.columns:
        df["leakage_amount"] = 0.0
    return df


# --------------------------------------------------------------------------
# the drift run
# --------------------------------------------------------------------------

@dataclass
class DriftResult:
    run_id: str
    run_ts: datetime
    model: str
    window: str
    window_n: int
    window_start: datetime | None
    window_end: datetime | None
    model_version: str
    reference_window: str
    feature: pd.DataFrame            # feature_drift output
    prediction: pd.DataFrame        # prediction_drift output
    gate: GateResult
    performance: dict[str, Any]
    report_rows: pd.DataFrame       # the rows write_drift_report persists
    status: str                    # OK | WARN | ALERT

    @property
    def alerts(self) -> pd.DataFrame:
        return self.report_rows[self.report_rows["alert"]].reset_index(drop=True)


def _gate_payload_frame(window_df: pd.DataFrame) -> pd.DataFrame:
    flat = pd.json_normalize(window_df["request_payload"])
    flat["request_id"] = window_df["request_id"].to_numpy()
    return flat


def run_drift(conn, bundle: S.ServingBundle, *, model: str, window: str,
              reference: pd.DataFrame | None = None,
              spine: pd.DataFrame | None = None,
              run_id: str | None = None) -> DriftResult:
    """The single monitoring pass the scheduled job and the notebook share."""
    ref = reference_frame() if reference is None else reference
    ref_label = reference_label(ref)
    numeric, categorical = monitored_features(bundle, model)
    ref_served = reference_served(bundle, model, None if reference is None else ref)

    win = load_window(conn, model=model, window=window)
    run_ts = datetime.now(timezone.utc)
    rid = run_id or str(uuid.uuid4())
    w_start = pd.to_datetime(win["ts"]).min() if len(win) else None
    w_end = pd.to_datetime(win["ts"]).max() if len(win) else None
    mv = win["model_version"].iloc[0] if len(win) else bundle.manifest["model_version"]

    served = assemble_served_rows(win, bundle, model)
    feat = feature_drift(ref_served, served, numeric, categorical) if len(served) else pd.DataFrame()

    ref_pred = pd.Series(bundle.pipelines[model].predict(ref_served[numeric + categorical]))
    pred = (prediction_drift(ref_pred, win["predicted_class"], CLASS_ORDER[model])
            if len(win) else pd.DataFrame())

    gate = validation_gate(_gate_payload_frame(win)) if len(win) else GateResult(0, pd.DataFrame(), [], 0.0)

    perf: dict[str, Any] = {}
    if len(win):
        sp = spine
        if sp is None:
            try:
                from capstone import eda
                sp = eda.load_spine()
            except Exception as exc:  # noqa: BLE001
                log.warning("spine unavailable, skipping performance drift: %s", exc)
                sp = None
        if sp is not None:
            joined = join_actuals(win, sp, model)
            if len(joined):
                perf = performance_drift(joined, model=model)

    rows = _build_report_rows(rid, run_ts, model, win, w_start, w_end, mv, ref_label,
                              feat, pred, gate, perf)
    status = _status(rows)
    return DriftResult(
        run_id=rid, run_ts=run_ts, model=model, window=window, window_n=len(win),
        window_start=w_start, window_end=w_end, model_version=mv, reference_window=ref_label,
        feature=feat, prediction=pred, gate=gate, performance=perf,
        report_rows=rows, status=status,
    )


def _build_report_rows(run_id, run_ts, model, win, w_start, w_end, mv, ref_label,
                       feat, pred, gate: GateResult, perf: dict) -> pd.DataFrame:
    base = dict(run_id=run_id, run_ts=run_ts, model=model,
                window_start=w_start, window_end=w_end, window_n=int(len(win)),
                model_version=mv, reference_window=ref_label)
    out: list[dict] = []

    for _, r in feat.iterrows():
        out.append({**base, "metric_kind": "feature_psi", "feature": r["feature"],
                    "value": float(r["psi"]), "reference": 0.0, "band": r["band"],
                    "alert": bool(r["psi"] > ALERT_RULES["feature_psi"]),
                    "detail": {"kind": r["kind"], "ks_stat": _n(r["ks_stat"]),
                               "ks_pvalue": _n(r["ks_pvalue"])}})
        if r["kind"] == "numeric" and np.isfinite(r["ks_stat"]):
            out.append({**base, "metric_kind": "feature_ks", "feature": r["feature"],
                        "value": float(r["ks_stat"]), "reference": 0.0,
                        "band": "significant" if r["ks_pvalue"] < 0.05 else "stable",
                        "alert": False, "detail": {"ks_pvalue": _n(r["ks_pvalue"])}})

    if len(pred):
        mix_psi = float(pred["mix_psi"].iloc[0])
        out.append({**base, "metric_kind": "prediction_psi", "feature": None,
                    "value": mix_psi, "reference": 0.0, "band": psi_band(mix_psi),
                    "alert": bool(mix_psi > ALERT_RULES["prediction_psi"]),
                    "detail": {"classes": pred["predicted_class"].tolist()}})
        for _, r in pred.iterrows():
            out.append({**base, "metric_kind": "class_share", "feature": r["predicted_class"],
                        "value": float(r["current_share"]), "reference": float(r["reference_share"]),
                        "band": "info", "alert": False,
                        "detail": {"share_delta": float(r["share_delta"])}})

    if gate.n:
        out.append({**base, "metric_kind": "gate_fail_rate", "feature": None,
                    "value": float(gate.fail_rate), "reference": 0.0,
                    "band": "alert" if not gate.passed else "ok",
                    "alert": bool(not gate.passed),
                    "detail": {"offending": gate.offending_ids[:50],
                               "by_rule": gate.offences.to_dict("records")}})

    if perf and model == "B":
        rc = perf["recall_costly"]
        out.append({**base, "metric_kind": "perf_recall_costly", "feature": "Rejected",
                    "value": float(rc), "reference": float(perf["recall_costly_baseline"]),
                    "band": "alert" if perf["recall_breach"] else "ok",
                    "alert": bool(perf["recall_breach"]),
                    "detail": {"delta_pts": round(perf["recall_costly_delta"], 4),
                               "precision": round(perf["precision_costly"], 4),
                               "alerts_per_month": round(perf["alerts_per_month"], 1)}})
        out.append({**base, "metric_kind": "perf_net_recovered", "feature": None,
                    "value": float(perf["net_recovered_per_month"]),
                    "reference": float(perf["net_recovered_baseline_per_month"]),
                    "band": "info", "alert": False, "detail": {}})

    df = pd.DataFrame(out)
    if not df.empty:
        df["detail"] = df["detail"].apply(lambda d: json.dumps(d, default=str))
    return df


def _n(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def _status(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "OK"
    if rows["alert"].any():
        return "ALERT"
    if (rows["band"] == "moderate").any():
        return "WARN"
    return "OK"


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

_DRIFT_INSERT = """
INSERT INTO capstone_solution.drift_report
    (run_id, run_ts, model, window_start, window_end, window_n, metric_kind,
     feature, value, reference, band, alert, detail, model_version, reference_window)
VALUES
    (%(run_id)s, %(run_ts)s, %(model)s, %(window_start)s, %(window_end)s, %(window_n)s,
     %(metric_kind)s, %(feature)s, %(value)s, %(reference)s, %(band)s, %(alert)s,
     %(detail)s, %(model_version)s, %(reference_window)s)
"""


def ensure_tables(conn) -> None:
    """Apply the Phase 5 prediction-log DDL then the Phase 6 drift DDL. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(PREDICTION_LOG_DDL.read_text())
        cur.execute(DRIFT_DDL.read_text())
    conn.commit()


def write_drift_report(conn, result: DriftResult) -> int:
    """Persist a drift run's rows to ``capstone_solution.drift_report``.
    Returns the row count written."""
    if result.report_rows.empty:
        return 0
    records = result.report_rows.to_dict("records")
    with conn.cursor() as cur:
        cur.executemany(_DRIFT_INSERT, records)
    conn.commit()
    return len(records)


def latest_runs(conn, *, limit: int = 20) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_ts, model, metric_kind, feature, value, reference, band, alert "
            "FROM capstone_solution.drift_report ORDER BY run_ts DESC, id DESC LIMIT %s",
            (limit,))
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
