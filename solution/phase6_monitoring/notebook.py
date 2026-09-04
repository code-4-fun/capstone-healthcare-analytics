"""Builds ``phase6.ipynb`` - the Phase 6 review artefact.

The notebook is the deliverable (per `CLAUDE.md`: notebooks are the phase
artefact from Phase 2 on). It stays thin - it imports the reusable logic from
``capstone.monitoring`` and the phase-local ``seed_traffic`` / ``make_charts`` /
``report`` helpers, then orchestrates, explains and displays inline.
``run_phase6.py`` regenerates and executes it.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent

CELLS: list[tuple[str, str]] = [
    ("md", """\
# Phase 6 - Monitoring, Drift Detection & Governance

*Hospital Operations & Revenue Risk Intelligence Platform*

**Goal:** long-term reliability and compliance for the deployed models - a
request validation gate, drift monitoring off the Phase 5 prediction log, a
scheduled drift job with threshold alerts, an append-only audit trail, and the
governance documents.

This notebook is thin: all reusable logic lives in `capstone.monitoring`; the
phase-local helpers are `seed_traffic` (demo traffic), `make_charts` and
`report`. It runs top-to-bottom from a clean kernel and is safe to re-run
(seeding is idempotent).

**Inputs:** `capstone_solution.prediction_log` (Phase 5), the persisted Phase 3
models + `phase5_api/serving_config.json`, `phase2_eda/output/feature_frame.parquet`,
`capstone_solution.v_visit_billing` (actual outcomes).
**Outputs:** `capstone_solution.drift_report`, `phase6_monitoring/output/`
(CSVs, `charts/`), `phase6_monitoring/PHASE6_FINDINGS.md`."""),

    ("code", """\
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from capstone import modeling as M, monitoring as mon, serving as S, viz
from capstone.db import connect, engine
import seed_traffic, make_charts, report

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 170)
viz.apply_house_style()

HERE = Path.cwd()
OUT = HERE / "output"; (OUT / "charts").mkdir(parents=True, exist_ok=True)
FINDINGS_MD = HERE / "PHASE6_FINDINGS.md"

bundle = S.load_serving_bundle()
config = bundle.config
print("serving_version", config["serving_version"], "| model_version", config["model_version"])"""),

    ("md", "## 1. Apply the DDL and seed two comparable traffic windows\n\n"
           "`prediction_log` only fills from live API calls, so we replay real Phase 3 "
           "**test-window** visits through the exact serving path the API uses. The **baseline** "
           "window is un-perturbed; the **drifted** window applies a deliberate shift - billed "
           "amounts +60%, department mix toward ER, one insurer over-represented, patients ~12 "
           "years older, and a volume spike. Seeding is idempotent (prior `phase6-seed:*` rows "
           "are removed first)."),
    ("code", """\
with connect() as conn:
    mon.ensure_tables(conn)
    seed = seed_traffic.seed_all(conn)
seed"""),

    ("md", "## 2. The drift reference and the monitored feature set\n\n"
           "The reference is the Phase 3 **test** split (the most recent window the models were "
           "validated on; the training window's calendar features are disjoint from any later "
           "window by construction). Each model monitors exactly its Phase 3 manifest feature "
           "list, so Model A is never compared on a billing distribution."),
    ("code", """\
ref = mon.reference_frame()
ref_label = mon.reference_label(ref)
monitored = {m: dict(zip(("numeric", "categorical"),
                         (len(a), len(b)))) for m in ("A", "B")
             for a, b in [mon.monitored_features(bundle, m)]}
print(ref_label)
pd.DataFrame(monitored).T"""),

    ("md", "## 3. Request validation gate\n\nThe Phase 2 data-quality rules (`capstone.data_quality`) "
           "reused as a batch gate over served request payloads: the enum domains and the "
           "pre-submission numeric ranges. A clean batch passes; a batch with bad enums / a "
           "negative amount / an out-of-range age is rejected, each offence named."),
    ("code", """\
def _payloads(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(S.PHASE2_PARQUET)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    test = M.time_split(df).test
    idx = rng.choice(len(test), size=n, replace=False)
    return [mon.feature_row_to_payload(r, "B") for _, r in test.iloc[idx].iterrows()]

clean = _payloads(120, 1)
bad = _payloads(60, 2)
for i, p in enumerate(bad):                       # corrupt ~half the batch
    if i % 2 == 0:
        p["department"] = "Radiology"             # not in the domain
    if i % 3 == 0:
        p["age"] = 210                            # out of range
    if i % 5 == 0:
        p["billed_amount"] = -500.0               # negative

gate_ok = mon.validation_gate(pd.json_normalize(clean).assign(request_id=range(len(clean))))
gate_bad = mon.validation_gate(pd.json_normalize(bad).assign(request_id=range(len(bad))))
print(f"clean  : fail_rate {gate_ok.fail_rate:.1%}  passed={gate_ok.passed}")
print(f"bad    : fail_rate {gate_bad.fail_rate:.1%}  passed={gate_bad.passed}")
display(gate_bad.offences)"""),

    ("md", "## 4. Run the drift job - baseline vs drifted window, both models\n\n"
           "`mon.run_drift` pulls a window from `prediction_log`, rebuilds the served feature rows "
           "with `capstone.serving.build_model_row`, computes feature / prediction / gate / "
           "performance drift, and returns the rows `write_drift_report` persists. Each window "
           "gets one `run_id` covering both models (the shape a scheduled run takes)."),
    ("code", """\
import uuid
res = {}
with connect() as conn:
    from capstone import eda
    spine = eda.load_spine()
    written = 0
    for window in ("baseline", "drift"):
        rid = str(uuid.uuid4())
        for model in ("A", "B"):
            r = mon.run_drift(conn, bundle, model=model, window=window,
                              reference=ref, spine=spine, run_id=rid)
            written += mon.write_drift_report(conn, r)
            res[(model, window)] = r
statuses = pd.DataFrame(
    [{"model": m, "window": w, "n": res[(m, w)].window_n,
      "status": res[(m, w)].status, "alerts": len(res[(m, w)].alerts)}
     for m in ("A", "B") for w in ("baseline", "drift")])
print(f"{written} rows written to drift_report")
statuses"""),

    ("md", "## 5. Feature drift\n\nModel B monitored features, PSI vs the test-window reference. "
           "The baseline window stays inside the stable band; the drifted window trips billed "
           "amount, department, insurer and age past the significant line."),
    ("code", """\
feat_baseline = res[("B", "baseline")].feature
feat_drift = res[("B", "drift")].feature
display(Markdown("**Baseline window** (top 8 by PSI)"))
display(feat_baseline.head(8))
display(Markdown("**Drifted window** (top 10 by PSI)"))
display(feat_drift.head(10))"""),

    ("md", "## 6. Prediction-distribution drift"),
    ("code", """\
pm = (res[("B", "baseline")].prediction[["predicted_class", "reference_share", "current_share"]]
      .rename(columns={"current_share": "baseline_share"}))
pm["drift_share"] = res[("B", "drift")].prediction["current_share"].to_numpy()
pred_psi_baseline = float(res[("B", "baseline")].prediction["mix_psi"].iloc[0])
pred_psi_drift = float(res[("B", "drift")].prediction["mix_psi"].iloc[0])
print(f"predicted-mix PSI  baseline {pred_psi_baseline:.3f}  drift {pred_psi_drift:.3f}  "
      f"(alert > {mon.ALERT_RULES['prediction_psi']})")
pm"""),

    ("md", "## 7. Performance drift\n\nActual `claim_status` joined back to the served predictions "
           "on the caller-supplied `visit_id`. Model B's monitored business metric is recall on "
           "Rejected claims at the operating threshold, against the Phase 4 baseline of 62%."),
    ("code", """\
perf = {
    "operating_threshold": mon.PHASE4_BASELINE["B"]["operating_threshold"],
    "recall_baseline": mon.PHASE4_BASELINE["B"]["recall_costly"],
    "recall_baseline_window": res[("B", "baseline")].performance.get("recall_costly", float("nan")),
    "recall_drift_window": res[("B", "drift")].performance.get("recall_costly", float("nan")),
    "precision_drift_window": res[("B", "drift")].performance.get("precision_costly", float("nan")),
    "recall_breach": res[("B", "drift")].performance.get("recall_breach", False),
}
pd.Series(perf).to_frame("value")"""),

    ("md", "## 8. Governance and audit\n\n`v_prediction_audit` joins every prediction to any manual "
           "override. We insert one example override, read it back through the view, and confirm "
           "the append-only trigger blocks an `UPDATE`."),
    ("code", """\
_REASON = "Provider flagged prior-auth mismatch not visible to the model"
_ACTOR = "claims.lead@hospital.example"
with connect() as conn, conn.cursor() as cur:
    cur.execute(\"\"\"SELECT request_id::text, predicted_class FROM capstone_solution.prediction_log
                   WHERE client_host = 'phase6-seed:drift' AND model = 'B'
                   ORDER BY ts DESC LIMIT 1\"\"\")
    rid_over, orig = cur.fetchone()
    cur.execute(\"\"\"INSERT INTO capstone_solution.prediction_override
        (request_id, model, original_class, override_class, actor, reason)
        VALUES (%s, 'B', %s, 'Rejected', %s, %s)\"\"\", (rid_over, orig, _ACTOR, _REASON))
    conn.commit()
    try:
        cur.execute("UPDATE capstone_solution.drift_report SET value = 0 "
                    "WHERE id = (SELECT min(id) FROM capstone_solution.drift_report)")
        append_only = "NOT enforced - UPDATE succeeded"
    except Exception as exc:
        conn.rollback()
        append_only = f"enforced: {str(exc).splitlines()[0]}"

override_example = {"actor": _ACTOR, "original_class": orig,
                    "override_class": "Rejected", "reason": _REASON}
eng = engine()
try:
    audit_sample = pd.read_sql(
        \"\"\"SELECT request_id, predicted_at, model, model_version, predicted_class,
                  override_class, override_actor, was_overridden
           FROM capstone_solution.v_prediction_audit
           WHERE client_host LIKE 'phase6-seed:%%'
           ORDER BY was_overridden DESC, predicted_at DESC LIMIT 5\"\"\", eng)
finally:
    eng.dispose()
print("append-only:", append_only)
audit_sample"""),

    ("md", "## 9. Charts, findings and the exit-criteria table\n\nEvery finding is backed by a "
           "house-style chart (`capstone.viz`); `make_charts.build_all` also writes the CSV "
           "exports. The Grafana dashboard (`phase6_monitoring/grafana/`) is the live operational "
           "view of the same `drift_report` / `prediction_log` data."),
    ("code", """\
all_rows = pd.concat([res[k].report_rows for k in res], ignore_index=True)
baseline_alerts = len(res[("A", "baseline")].alerts) + len(res[("B", "baseline")].alerts)
drift_alerts = len(res[("A", "drift")].alerts) + len(res[("B", "drift")].alerts)

ctx = report.Phase6Context(
    generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    config=config, reference_window=ref_label, monitored_counts=monitored,
    gate_ok_n=gate_ok.n, gate_bad_n=gate_bad.n,
    gate_ok_fail_rate=gate_ok.fail_rate, gate_bad_fail_rate=gate_bad.fail_rate,
    gate_offences=gate_bad.offences,
    feat_baseline=feat_baseline, feat_drift=feat_drift,
    prediction_mix=pm, pred_psi_baseline=pred_psi_baseline, pred_psi_drift=pred_psi_drift,
    performance_drift=perf,
    drift_rows=all_rows,
    baseline_status=res[("B", "baseline")].status, drift_status=res[("B", "drift")].status,
    baseline_alert_count=baseline_alerts, drift_alert_count=drift_alerts,
    runs_written=written,
    audit_sample=audit_sample, override_example=override_example,
    seed_summary=seed,
)
charts = make_charts.build_all(ctx)
_C = {k: p for k, p, _ in charts}
report.write_findings(ctx, charts, FINDINGS_MD)
list(_C)"""),
    ("code", 'Image(str(_C["gate_offences"]))'),
    ("code", 'Image(str(_C["feature_psi_baseline"]))'),
    ("code", 'Image(str(_C["feature_psi_drift"]))'),
    ("code", 'Image(str(_C["prediction_mix"]))'),
    ("code", 'Image(str(_C["performance_drift"]))'),
    ("code", 'Image(str(_C["alert_summary"]))'),

    ("md", """\
## 10. Exit criteria

- [x] Data validation gate on incoming requests (range / enum / schema),
      reusing the Phase 2 rule registry (section 3).
- [x] Drift monitoring off `prediction_log`: feature drift (PSI / KS vs the
      Phase 3 reference), prediction-distribution drift, and performance drift
      once outcomes are joined back (sections 5-7).
- [x] Scheduled drift job + threshold alerts (`phase6_monitoring/scheduler.py`,
      `capstone.monitoring.ALERT_RULES`) - fires on the injected drift.
- [x] `drift_report` table - one append-only row per (run, metric).
- [x] Audit log: `v_prediction_audit` (who / what / when for predictions) +
      `prediction_override` (manual overrides); immutable-by-convention,
      enforced by trigger (section 8).
- [x] Governance docs: `governance.md`, `retraining_policy.md`, `runbook.md`.

**Hand-off to the executive presentation:** the platform is complete
end-to-end. The final phase turns it into a leadership deck - the problem, the
architecture, the SQL + EDA insights, model performance in money and risk
terms, and this monitoring / retraining story."""),
]


def build(path: str | Path | None = None) -> Path:
    path = Path(path) if path else (HERE / "phase6.ipynb")
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbformat.write(nb, str(path))
    return path


if __name__ == "__main__":
    print("wrote", build())
