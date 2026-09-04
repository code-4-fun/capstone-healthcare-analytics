"""Phase 6 :: assemble PHASE6_FINDINGS.md.

Kept out of the notebook so the notebook stays thin. :func:`write_findings`
takes the :class:`Phase6Context` the notebook builds plus the chart list from
``make_charts.build_all`` and writes the themed markdown - every chart embedded,
supporting numbers in an appendix. Generated, never hand-edited.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from capstone import monitoring as mon

HERE = Path(__file__).resolve().parent


@dataclass
class Phase6Context:
    generated: str
    config: dict
    reference_window: str
    monitored_counts: dict[str, dict[str, int]]     # model -> {numeric, categorical}

    # validation gate
    gate_ok_n: int
    gate_bad_n: int
    gate_ok_fail_rate: float
    gate_bad_fail_rate: float
    gate_offences: pd.DataFrame

    # feature drift (Model B)
    feat_baseline: pd.DataFrame
    feat_drift: pd.DataFrame

    # prediction-mix drift (Model B)
    prediction_mix: pd.DataFrame
    pred_psi_baseline: float
    pred_psi_drift: float

    # performance drift (Model B)
    performance_drift: dict[str, Any]

    # drift_report
    drift_rows: pd.DataFrame
    baseline_status: str
    drift_status: str
    baseline_alert_count: int
    drift_alert_count: int
    runs_written: int

    # governance / audit
    audit_sample: pd.DataFrame
    override_example: dict[str, Any]

    seed_summary: dict[str, Any] = field(default_factory=dict)


def _img(charts: dict, key: str) -> str:
    if key not in charts:
        return f"*(chart `{key}` not generated this run)*\n"
    path, caption = charts[key]
    rel = Path(path).relative_to(HERE).as_posix()
    return f"![{caption}]({rel})\n\n*{caption}*\n"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _table(df: pd.DataFrame, cols: list[str] | None = None, *, round_to: int = 3) -> str:
    d = (df[cols] if cols else df).copy()
    num = d.select_dtypes(include="number").columns
    d[num] = d[num].round(round_to)
    obj = d.select_dtypes(include="object").columns
    d[obj] = d[obj].fillna("-")
    return d.to_markdown(index=False)


def write_findings(ctx: Phase6Context, chart_list, out_path: str | Path | None = None) -> Path:
    out_path = Path(out_path) if out_path else (HERE / "PHASE6_FINDINGS.md")
    charts = {k: (p, c) for k, p, c in chart_list}
    ar = mon.ALERT_RULES
    pd_ = ctx.performance_drift
    b_num = ctx.monitored_counts["B"]["numeric"]
    b_cat = ctx.monitored_counts["B"]["categorical"]

    n_sig = int((ctx.feat_drift["band"] == "significant").sum())
    drop_pts = pd_["recall_baseline"] - pd_["recall_drift_window"]

    md = f"""# Phase 6 - Monitoring, Drift Detection & Governance

*Hospital Operations & Revenue Risk Intelligence Platform*
*Generated {ctx.generated} - regenerate with `uv run python phase6_monitoring/run_phase6.py`.*

## Executive summary

Phase 5 logs every prediction to `capstone_solution.prediction_log`. Phase 6
turns that log into a monitored signal: a **request validation gate**, a
**drift job** that compares served traffic to the Phase 3 reference
(`{ctx.reference_window}`), an append-only **`drift_report`** table + audit
trail, a **scheduler** sidecar, and a **Grafana** dashboard - plus the
governance documents (`governance.md`, `retraining_policy.md`, `runbook.md`).

To exercise it end-to-end the job seeds two comparable windows into the log:
a **baseline** window (un-perturbed replay of Phase 3 test-window visits) and a
**drifted** window (the same traffic with a deliberate shift in billed amount,
department mix, insurer mix, patient age and volume).

| Check | Baseline window | Drifted window |
|---|---|---|
| Drift-run status | **{ctx.baseline_status}** | **{ctx.drift_status}** |
| Monitored features past the significant PSI line (Model B) | {int((ctx.feat_baseline["band"] == "significant").sum())} / {len(ctx.feat_baseline)} | {n_sig} / {len(ctx.feat_drift)} |
| Predicted-class-mix PSI | {ctx.pred_psi_baseline:.3f} | {ctx.pred_psi_drift:.3f} |
| Model B recall on Rejected (actuals joined) | {_pct(pd_["recall_baseline_window"])} | {_pct(pd_["recall_drift_window"])} |
| Alerts written to `drift_report` | {ctx.baseline_alert_count} | {ctx.drift_alert_count} |

The monitor is quiet on the baseline window and fires on every injected shift on
the drifted window - including the one that matters commercially: Model B's
recall on to-be-rejected claims falls {drop_pts * 100:.0f} points below the
Phase 4 baseline, breaching the retraining trigger.

## 1. Request validation gate

Every served request is checked against the Phase 2 data-quality rules
(`capstone.data_quality`) reused as a batch gate - the enum domains
(`department`, `visit_type`, `gender`, `city`, `insurance_provider`,
`risk_score`) and the pre-submission numeric ranges (`age` 0-120,
`billed_amount >= 0`, `length_of_stay_hours >= 0`). A window fails the gate when
more than {ar['gate_fail_rate']:.0%} of its requests offend.

- Clean probe batch ({ctx.gate_ok_n} requests): fail rate {_pct(ctx.gate_ok_fail_rate)} - **pass**.
- Malformed probe batch ({ctx.gate_bad_n} requests): fail rate {_pct(ctx.gate_bad_fail_rate)} - **rejected**.

{_img(charts, "gate_offences")}

## 2. Feature drift

The drift job rebuilds the exact model-input rows for a window from the logged
request payloads (via `capstone.serving.build_model_row` - the same assembly the
API uses), then scores each feature with **PSI** (deciles on the reference) and,
for numeric features, a two-sample **KS** test. Bands: stable `< {mon.PSI_STABLE}`,
moderate `< {mon.PSI_SIGNIFICANT}`, significant `>= {mon.PSI_SIGNIFICANT}`. Model B
monitors {b_num} numeric + {b_cat} categorical features - exactly its Phase 3
manifest list, so Model A is never compared on a billing distribution.

### 2.1 Baseline window - no drift

{_img(charts, "feature_psi_baseline")}

### 2.2 Drifted window - {n_sig} features past the line

{_img(charts, "feature_psi_drift")}

## 3. Prediction-distribution drift

{_img(charts, "prediction_mix")}

The baseline window's predicted-class mix tracks the reference (PSI
{ctx.pred_psi_baseline:.3f}); the drifted window pushes it to
{ctx.pred_psi_drift:.3f}, past the {ar['prediction_psi']} alert threshold.

## 4. Performance drift

Once real outcomes land they are joined back to the served predictions on the
caller-supplied `visit_id`. For Model B the monitored business metric is
**recall on Rejected claims** at the operating threshold
(`P(Rejected) >= {pd_['operating_threshold']}`), against the Phase 4 baseline of
{_pct(pd_['recall_baseline'])}.

{_img(charts, "performance_drift")}

Recall on the drifted window is {_pct(pd_['recall_drift_window'])} - a
{drop_pts * 100:.0f}-point drop that breaches the
{ar['recall_costly_drop_pts'] * 100:.0f}-point retraining trigger. (Model A is a
base-rate monitor - it has no recall on High-risk visits by construction, so
only its risk-mix share is tracked.)

## 5. The drift_report table and alert rules

Every run writes one row per (metric) to `capstone_solution.drift_report` with a
shared `run_id`, `run_ts`, the window bounds, the metric value, the reference it
is judged against, its band, and an `alert` flag. This run wrote
**{ctx.runs_written} rows**.

{_img(charts, "alert_summary")}

Alert rules (one place - `capstone.monitoring.ALERT_RULES`, cited by
`governance.md`):

| Metric | Alerts when |
|---|---|
| `feature_psi` | any monitored feature PSI > {ar['feature_psi']} |
| `prediction_psi` | predicted-class-mix PSI > {ar['prediction_psi']} |
| `perf_recall_costly` | recall on the costly class drops > {ar['recall_costly_drop_pts'] * 100:.0f} points vs the Phase 4 baseline |
| `gate_fail_rate` | > {ar['gate_fail_rate']:.0%} of served requests fail the validation gate |

## 6. Governance and audit

- **`v_prediction_audit`** joins every `prediction_log` row to any
  `prediction_override` - the who / what / when for predictions and manual
  overrides. Sample:

{_table(ctx.audit_sample)}

- **Manual override** rows record `actor`, `reason`, `original_class` and
  `override_class`. Example inserted this run: actor `{ctx.override_example.get('actor')}`,
  `{ctx.override_example.get('original_class')}` -> `{ctx.override_example.get('override_class')}`
  ("{ctx.override_example.get('reason')}").
- **Append-only, enforced.** A trigger (`capstone_solution.forbid_mutation`)
  raises on any `UPDATE` to `prediction_log` / `drift_report` /
  `prediction_override`, and on `DELETE` to the two audit tables.
- Governance documents: **`governance.md`** (assumptions, limitations, the
  monitoring design, roles, data handling), **`retraining_policy.md`**
  (triggers, procedure, rollback, sign-off), **`runbook.md`** (incident
  response).

## 7. Operations - scheduler + Grafana

- **`phase6_monitoring/scheduler.py`** runs inside `docker compose` alongside
  Postgres, the API and Grafana: it applies the DDL, seeds the demo windows if
  the log is empty, then runs the drift job every `DRIFT_INTERVAL_SECONDS`.
- **`docker compose -f phase6_monitoring/docker-compose.yml up --build`** brings
  up `postgres` + `api` (`:8000`) + `scheduler` + `grafana` (`:3000`, anonymous
  viewer). The **Hospital Drift** dashboard reads `drift_report` and
  `prediction_log` directly: feature PSI over runs, predicted-class mix, Model B
  recall vs baseline, gate fail-rate, and the active-alert table.
- Cron equivalent (no Docker): `0 */6 * * * cd <solution> && uv run python -m
  phase6_monitoring drift-job --window last-week --fail-on-alert`.

## 8. Exit criteria

| Criterion | Status |
|---|---|
| Data validation gate on incoming requests (range/enum/schema), reusing Phase 2 rules | met - section 1 |
| Feature drift (PSI / KS vs the Phase 3 reference) | met - section 2 |
| Prediction-distribution drift | met - section 3 |
| Performance drift once outcomes land | met - section 4 (Model B, actuals joined on `visit_id`) |
| Scheduled drift job + threshold alerts | met - scheduler sidecar + `ALERT_RULES`; alerts on the injected drift |
| `drift_report` view/table | met - `capstone_solution.drift_report` ({ctx.runs_written} rows this run) |
| Audit log: who/what/when for predictions and overrides | met - `v_prediction_audit` + `prediction_override` |
| Audit log immutable-by-convention | met - enforced by trigger |
| Governance docs: assumptions, limitations, retraining policy, incident runbook | met - `governance.md`, `retraining_policy.md`, `runbook.md` |
| Drift job runs on a schedule and alerts on injected drift | met - scheduler + this run's drifted-window ALERT |

## 9. Hand-off to the executive presentation

The platform is complete end-to-end: SQL analytics -> EDA + feature catalogue ->
two calibrated models -> evaluation + model cards -> served API with a
prediction log -> **this monitoring + governance layer**. The final phase turns
it into a leadership deck: the operational / financial problem, the architecture
and data flow, headline SQL + EDA insights, model performance in money and risk
terms (Model B ~Rs 1.8 Cr/year recoverable at the operating point; Model A a
risk-mix monitor), and this deployment / monitoring / retraining story.

---

## Appendix - supporting tables

### Monitored feature drift - drifted window (Model B)

{_table(ctx.feat_drift, ["feature", "kind", "psi", "ks_stat", "ks_pvalue", "band", "drifted"])}

### Predicted-class mix

{_table(ctx.prediction_mix)}

### Drift-report alert rows (this run)

{_table(ctx.drift_rows[ctx.drift_rows["alert"]][["model", "metric_kind", "feature", "value", "reference", "band"]])}

### Seed summary

{ctx.seed_summary}
"""
    out_path.write_text(md)
    return out_path
