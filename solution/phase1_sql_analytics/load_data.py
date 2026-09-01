"""Phase 1 :: Direct typed load of the raw CSVs into the core tables.

Strategy (per the chosen "direct typed load"):
  * read each CSV with pandas, coerce to the target types,
  * validate every row against the business domain rules,
  * COPY the accepted rows straight into the typed, constrained tables,
  * park every rejected row in ``load_rejects`` (JSONB) with a reason.

Load order respects the foreign keys: patients -> visits -> billing.
Re-running is safe: the three tables are TRUNCATEd first.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from capstone.db import SETTINGS, connect

VALID_GENDER = {"M", "F"}
VALID_VISIT_TYPE = {"ER", "OPD", "ICU"}
VALID_RISK = {"Low", "Medium", "High"}
VALID_CLAIM_STATUS = {"Paid", "Pending", "Rejected"}


def _clean(value):
    """Normalise pandas NA / NaN / empty string to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if value is pd.NaT:
        return None
    return value


def _to_date(value):
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _to_int(value):
    value = _clean(value)
    return None if value is None else int(float(value))


def _to_float(value):
    value = _clean(value)
    return None if value is None else float(value)


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return str(obj)


# ---------------------------------------------------------------------------
# Row validators: return (typed_tuple, None) on success or (None, reason)
# ---------------------------------------------------------------------------
def validate_patient(row: dict):
    try:
        rec = (
            _to_int(row["patient_id"]),
            _to_int(row["age"]),
            _clean(row["gender"]),
            _clean(row["city"]),
            _clean(row["insurance_provider"]),
            _to_int(row["chronic_flag"]),
            _to_date(row["registration_date"]),
        )
    except (ValueError, TypeError, KeyError) as exc:
        return None, f"type coercion failed: {exc}"

    pid, age, gender, city, provider, chronic, reg = rec
    if pid is None:
        return None, "patient_id is null"
    if age is None or not (0 <= age <= 120):
        return None, f"age out of range: {age}"
    if gender not in VALID_GENDER:
        return None, f"invalid gender: {gender}"
    if not city or not provider:
        return None, "city or insurance_provider is null"
    if chronic not in (0, 1):
        return None, f"invalid chronic_flag: {chronic}"
    if reg is None:
        return None, "registration_date is null"
    return rec, None


def validate_visit(row: dict, known_patients: set[int]):
    try:
        rec = (
            _to_int(row["visit_id"]),
            _to_int(row["patient_id"]),
            _to_date(row["visit_date"]),
            _clean(row["department"]),
            _clean(row["visit_type"]),
            _to_float(row["length_of_stay_hours"]),
            _clean(row["risk_score"]),
            _to_int(row["doctor_id"]),
        )
    except (ValueError, TypeError, KeyError) as exc:
        return None, f"type coercion failed: {exc}"

    vid, pid, vdate, dept, vtype, los, risk, doc = rec
    if vid is None:
        return None, "visit_id is null"
    if pid not in known_patients:
        return None, f"patient_id {pid} not in patients"
    if vdate is None:
        return None, "visit_date is null"
    if not dept:
        return None, "department is null"
    if vtype not in VALID_VISIT_TYPE:
        return None, f"invalid visit_type: {vtype}"
    if los is None or los < 0:
        return None, f"invalid length_of_stay_hours: {los}"
    if risk not in VALID_RISK:
        return None, f"invalid risk_score: {risk}"
    if doc is None:
        return None, "doctor_id is null"
    return rec, None


def validate_bill(row: dict, known_visits: set[int], seen_visits: set[int]):
    try:
        rec = (
            _to_int(row["bill_id"]),
            _to_int(row["visit_id"]),
            _to_float(row["billed_amount"]),
            _to_float(row["approved_amount"]),
            _clean(row["claim_status"]),
            _to_float(row["payment_days"]),
            _to_date(row["billing_date"]),
        )
    except (ValueError, TypeError, KeyError) as exc:
        return None, f"type coercion failed: {exc}"

    bid, vid, billed, approved, status, pdays, bdate = rec
    if bid is None:
        return None, "bill_id is null"
    if vid not in known_visits:
        return None, f"visit_id {vid} not in visits"
    if vid in seen_visits:
        return None, f"duplicate visit_id {vid} in billing"
    if billed is None or billed < 0:
        return None, f"invalid billed_amount: {billed}"
    if approved is not None and (approved < 0 or approved > billed):
        return None, f"approved_amount {approved} outside [0, billed={billed}]"
    if status not in VALID_CLAIM_STATUS:
        return None, f"invalid claim_status: {status}"
    if pdays is not None and pdays < 0:
        return None, f"negative payment_days: {pdays}"
    if bdate is None:
        return None, "billing_date is null"
    return rec, None


# ---------------------------------------------------------------------------
def _copy_rows(cur, table: str, columns: list[str], rows: list[tuple]) -> None:
    cols = ", ".join(columns)
    with cur.copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(row)


def _record_rejects(cur, source_table: str, rejects: list[tuple[dict, str]]) -> None:
    for raw, reason in rejects:
        cur.execute(
            "INSERT INTO load_rejects (source_table, source_row, reason) VALUES (%s, %s, %s)",
            (source_table, json.dumps(raw, default=_json_default), reason),
        )


def load() -> dict:
    data_dir: Path = SETTINGS.data_dir
    patients_df = pd.read_csv(data_dir / "patients.csv")
    visits_df = pd.read_csv(data_dir / "visits.csv")
    billing_df = pd.read_csv(data_dir / "billing.csv")

    summary: dict[str, dict] = {}

    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE billing, visits, patients RESTART IDENTITY")
        cur.execute("TRUNCATE load_rejects RESTART IDENTITY")

        # --- patients ---------------------------------------------------------
        good, bad = [], []
        for raw in patients_df.to_dict("records"):
            rec, reason = validate_patient(raw)
            (good.append(rec) if reason is None else bad.append((raw, reason)))
        _copy_rows(cur, "patients",
                   ["patient_id", "age", "gender", "city", "insurance_provider",
                    "chronic_flag", "registration_date"], good)
        _record_rejects(cur, "patients", bad)
        known_patients = {r[0] for r in good}
        summary["patients"] = {"read": len(patients_df), "loaded": len(good), "rejected": len(bad)}

        # --- visits ----------------------------------------------------------
        good, bad = [], []
        for raw in visits_df.to_dict("records"):
            rec, reason = validate_visit(raw, known_patients)
            (good.append(rec) if reason is None else bad.append((raw, reason)))
        _copy_rows(cur, "visits",
                   ["visit_id", "patient_id", "visit_date", "department", "visit_type",
                    "length_of_stay_hours", "risk_score", "doctor_id"], good)
        _record_rejects(cur, "visits", bad)
        known_visits = {r[0] for r in good}
        summary["visits"] = {"read": len(visits_df), "loaded": len(good), "rejected": len(bad)}

        # --- billing -------------------------------------------------------
        good, bad, seen = [], [], set()
        for raw in billing_df.to_dict("records"):
            rec, reason = validate_bill(raw, known_visits, seen)
            if reason is None:
                good.append(rec)
                seen.add(rec[1])
            else:
                bad.append((raw, reason))
        _copy_rows(cur, "billing",
                   ["bill_id", "visit_id", "billed_amount", "approved_amount",
                    "claim_status", "payment_days", "billing_date"], good)
        _record_rejects(cur, "billing", bad)
        summary["billing"] = {"read": len(billing_df), "loaded": len(good), "rejected": len(bad)}

        conn.commit()

    return summary


if __name__ == "__main__":
    result = load()
    print(f"Typed load into schema '{SETTINGS.schema}' complete:\n")
    for table, stats in result.items():
        print(f"  {table:<10} read={stats['read']:>6}  "
              f"loaded={stats['loaded']:>6}  rejected={stats['rejected']:>4}")
