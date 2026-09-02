"""Phase 2 orchestrator: EDA, data-quality formalisation, feature catalogue.

    uv run python phase2_eda/run_phase2.py

Idempotent - rebuilds every table, chart and document from the Phase 1 analytics
layer. If the layer is unreachable it rebuilds it first via ``run_phase1``.

Steps:
  1. check / rebuild the Phase 1 analytics layer
  2. profiling + business analyses      -> output/*.csv
  3. feature catalogue + as-of matrix   -> feature_spec.yaml, output/feature_frame.*
  4. data-quality validators            -> output/data_quality_report.csv
  5. leakage self-check                 -> asserts no post-outcome field is model-eligible
  6. charts (one per finding)           -> output/charts/*.png
  7. PHASE2_FINDINGS.md                 -> findings by theme, charts embedded, tables in appendix
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

from capstone.db import SETTINGS, engine  # noqa: E402

import analyses  # noqa: E402
import data_quality_rules as dq  # noqa: E402
import features as feat  # noqa: E402
import make_charts  # noqa: E402

OUT = HERE / "output"


def ensure_analytics_layer() -> None:
    try:
        eng = engine()
        pd.read_sql("SELECT 1 FROM v_visit_billing LIMIT 1", eng)
        eng.dispose()
        print(f"[1] analytics layer reachable (db={SETTINGS.database}, schema={SETTINGS.schema})")
    except Exception as exc:  # noqa: BLE001
        print(f"[1] analytics layer unreachable ({exc}); rebuilding via run_phase1")
        import subprocess
        subprocess.run(
            [sys.executable, str(HERE.parents[0] / "phase1_sql_analytics" / "run_phase1.py")],
            check=True,
        )


def leakage_self_check(fframe: pd.DataFrame) -> list[str]:
    """No post-outcome field may be eligible for either model."""
    a = set(feat.model_features(fframe, "A"))
    b = set(feat.model_features(fframe, "B"))
    forbidden = {"approved_amount", "payment_days", "claim_status", "billing_date",
                 "billing_lag_days", "collected_amount", "leakage_amount", "pending_amount",
                 "is_paid", "is_pending", "is_rejected", "registration_date"}
    problems = []
    for name, cols in (("Model A", a), ("Model B", b)):
        bad = cols & forbidden
        if bad:
            problems.append(f"{name} exposes forbidden field(s): {sorted(bad)}")
    # risk_score must be absent from Model A (it is the target) and present for B
    if "risk_score" in a:
        problems.append("Model A exposes risk_score (its own target)")
    if "length_of_stay_hours" in a:
        problems.append("Model A exposes length_of_stay_hours (unknown at admission)")
    if "billed_amount" in a:
        problems.append("Model A exposes billed_amount (billing artefact)")
    return problems


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    ensure_analytics_layer()

    print("[2] profiling + business analyses -> phase2_eda/output/")
    tables = analyses.build_all()
    make_charts._T.update(tables)  # reuse; don't recompute in the chart layer
    for name, t in tables.items():
        print(f"      {name:<32} {len(t):>4} rows")

    print("[3] feature catalogue + as-of matrix")
    spec_path = feat.write_feature_spec_yaml()
    fframe = feat.build_feature_frame(analyses.load_spine())
    print(f"      {spec_path.name}  ({len(feat.FEATURE_SPEC)} features)")
    print(f"      feature_frame: {fframe.shape[0]} rows x {fframe.shape[1]} cols  "
          f"(Model A: {len(feat.model_features(fframe, 'A'))}, "
          f"Model B: {len(feat.model_features(fframe, 'B'))} eligible)")

    print("[4] data-quality validators")
    dq_rep = dq.validate(analyses.load_spine())
    dq_rep.to_csv(OUT / "data_quality_report.csv", index=False)
    n_err = int(((dq_rep["severity"] == "ERROR") & (dq_rep["records_flagged"].fillna(0) > 0)).sum())
    print(f"      {len(dq_rep)} rules, {n_err} ERROR-level with flagged rows")

    print("[5] leakage self-check")
    problems = leakage_self_check(fframe)
    if problems:
        raise SystemExit("LEAKAGE CHECK FAILED:\n  - " + "\n  - ".join(problems))
    print("      OK - no post-outcome field is eligible for Model A or Model B")

    print("[6] charts -> phase2_eda/output/charts/")
    charts = make_charts.build_all()

    print("[7] PHASE2_FINDINGS.md")
    _write_report(started, tables, dq_rep, fframe, charts)
    print(f"\nDone. Findings: {HERE / 'PHASE2_FINDINGS.md'}")


# ---------------------------------------------------------------------------
def _img(charts: dict, key: str) -> str:
    path, caption = charts[key]
    rel = path.relative_to(HERE).as_posix()
    return f"![{caption}]({rel})\n\n*{caption}*\n"


def _write_report(started, tables, dq_rep, fframe, chart_list) -> None:
    charts = {k: (p, c) for k, p, c in chart_list}
    rw = tables["revenue_waterfall"].set_index("component")
    rb = tables["target_balance_risk_score"].set_index("class")
    cb = tables["target_balance_claim_status"].set_index("class")
    dc = tables["denial_cohort"].set_index("billed_band")
    sig = tables["feature_signal"]
    fa = len(feat.model_features(fframe, "A"))
    fb = len(feat.model_features(fframe, "B"))

    L: list[str] = []
    L.append("# Phase 2 - Exploratory Data Analysis & Data Quality :: Findings\n")
    L.append("*Hospital Operations & Revenue Risk Intelligence Platform - modelling readiness*\n")
    L.append(f"- Generated: {started.isoformat(timespec='seconds')}")
    L.append(f"- Source: `{SETTINGS.database}` / `{SETTINGS.schema}.v_visit_billing` - "
             "25,000 visits, 25,000 claims, 5,000 patients (2025-01-20 to 2026-01-20)")
    L.append("- Every finding is backed by a chart in `output/charts/`; supporting numbers are in "
             "the appendix and as CSVs in `output/`.")
    L.append("- Artefacts for Phase 3: `feature_spec.yaml` (catalogue + leakage register), "
             "`data_quality_rules.py` (validators), `output/feature_frame.*` (as-of matrix).\n")

    L.append("## Executive summary\n")
    L.append(f"1. **Model A (visit risk) has no learnable signal.** Every eligible feature sits at "
             f"the permuted-target noise floor (best < 0.5% of target entropy). `risk_score` is "
             f"effectively randomly assigned - Phase 3 should expect near-base-rate performance and "
             f"treat Model A as a monitoring baseline, not a predictor.")
    L.append(f"2. **Model B (claim outcome) is a billed-amount model.** `billed_amount` / "
             f"`billed_band` are the only features above the noise floor; rejection rate is "
             f"**non-monotonic** - {dc.loc['15k-30k','rejection_rate_pct']:.1f}% at 15k-30k vs "
             f"{dc.loc['<5k','rejection_rate_pct']:.1f}% below 5k and "
             f"{dc.loc['30k+','rejection_rate_pct']:.1f}% above 30k - and that mid band is "
             f"{dc.loc['15k-30k','denied_share_of_leakage_pct']:.0f}% of denial leakage.")
    L.append(f"3. **Targets are imbalanced toward the cheap class.** risk_score "
             f"{rb.loc['Low','share_pct']:.0f}/{rb.loc['Medium','share_pct']:.0f}/"
             f"{rb.loc['High','share_pct']:.0f} (Low/Med/High); claim_status "
             f"{cb.loc['Paid','share_pct']:.0f}/{cb.loc['Pending','share_pct']:.0f}/"
             f"{cb.loc['Rejected','share_pct']:.0f} (Paid/Pending/Rejected). Class weighting and "
             f"threshold tuning are mandatory; recall on High / Rejected is the headline metric.")
    L.append("4. **Four Phase 1 data-quality findings are now quantified and have a written "
             "handling policy** (below): the 0.5h / 500 capture floors (keep + flag), the "
             "status<->approved_amount decoupling (analysis-only, both post-outcome), ~5% MCAR "
             "missingness on the two billing outcome fields (impute for reporting only), and the "
             "temporal inconsistency (~50% billing-before-visit) -> **`visit_date` is the only "
             "temporal key**.")
    L.append("5. **The business is structurally uniform.** No seasonality, no weekday pattern, "
             "near-identical departments and insurers, no length-of-stay drivers. Planning levers "
             "are network-level (risk mix, billed-amount triage), not local.\n")

    L.append("---\n")
    L.append("## 1. Model targets & feature signal\n")
    L.append(_img(charts, "target_balance"))
    L.append(f"\nMutual-information screen of the candidate features against each target "
             f"({fa} fields eligible for Model A, {fb} for Model B), with a permuted-target run as "
             f"the noise floor:\n")
    L.append(_img(charts, "signal_model_a"))
    L.append(_img(charts, "signal_model_b"))
    L.append("> **Implication for Phase 3.** Model A cannot beat a stratified baseline on this "
             "data; ship it as a calibrated base-rate model and document the ceiling. Model B has a "
             "real but narrow signal - a regularised linear model on `billed_amount` + a spline / "
             "band term is the honest baseline; gradient boosting will mostly re-learn the band "
             "shape.\n")

    L.append("---\n")
    L.append("## 2. Data quality - findings formalised\n")
    L.append("### 2.1 Capture floors\n")
    L.append(_img(charts, "distribution_floors"))
    L.append("### 2.2 claim_status <-> approved_amount decoupling\n")
    L.append(_img(charts, "approved_ratio_by_status"))
    L.append(_img(charts, "post_outcome_missingness"))
    L.append(_img(charts, "missingness_overview"))
    L.append("### 2.3 Temporal inconsistency\n")
    L.append(_img(charts, "temporal_inconsistency"))
    L.append(_img(charts, "date_fields_monthly"))
    L.append("### 2.4 Handling policy\n")
    L.append("| Finding | Decision | Rationale |")
    L.append("|---|---|---|")
    for h in dq.HANDLING:
        L.append(f"| {h.finding} | **{h.decision}** | {h.rationale} |")
    L.append("\nValidators for all of the above live in `data_quality_rules.py` "
             "(`validate(df)`, `add_quality_flags(df)`, `apply_training_exclusions(df)`), "
             "importable by Phase 3 and the Phase 5/6 request gate.\n")

    L.append("---\n")
    L.append("## 3. Business analyses\n")
    L.append(_img(charts, "revenue_waterfall"))
    L.append(_img(charts, "rejection_by_billed_band"))
    L.append(_img(charts, "rejection_flat_across_dims"))
    L.append(_img(charts, "denial_leakage_concentration"))
    L.append(_img(charts, "patient_flow_seasonality"))
    L.append(_img(charts, "flow_day_of_week"))
    L.append(_img(charts, "department_acuity_mix"))
    L.append(_img(charts, "los_no_drivers"))
    L.append(_img(charts, "provider_behavior"))

    L.append("---\n")
    L.append("## 4. Feature engineering & leakage register\n")
    L.append(f"`feature_spec.yaml` catalogues **{len(feat.FEATURE_SPEC)} fields** - "
             f"{len(feat.model_features(fframe,'A'))} eligible for Model A, "
             f"{len(feat.model_features(fframe,'B'))} for Model B - each with a definition, a "
             f"source, an as-of rule keyed on `visit_date`, a dtype and a per-model leakage "
             f"verdict. The as-of matrix is materialised at `output/feature_frame.*` "
             f"({fframe.shape[0]} rows x {fframe.shape[1]} cols).\n")
    L.append(_img(charts, "visits_per_patient"))
    L.append("**Leakage rules (enforced by `run_phase2.py` step 5):**\n")
    L.append("- `visit_date` is the sole temporal key. No feature derives from `billing_date` or "
             "`registration_date` (including `billing_lag_days`, days-since-registration).")
    L.append("- No post-outcome field (`approved_amount`, `payment_days`, `claim_status`, "
             "`collected/leakage/pending_amount`) feeds Model A or the pre-submission Model B.")
    L.append("- `length_of_stay_hours` and `billed_amount` are **excluded from Model A** (not "
             "known at admission / not an operational field) but **allowed for Model B** "
             "(known before the claim is filed).")
    L.append("- `risk_score` is the Model A target and a legitimate Model B input.")
    L.append("- Patient- and provider-history features count only visits with "
             "`visit_date` strictly earlier than the current visit.\n")
    L.append("### Feature catalogue\n")
    L.append(tables["feature_catalogue"].to_markdown(index=False))

    L.append("\n---\n")
    L.append("## Appendix - supporting tables\n")
    L.append("### Column profile (spine)\n")
    L.append(tables["column_profile"].to_markdown(index=False))
    L.append("\n### Target balance\n")
    L.append("**risk_score**\n")
    L.append(tables["target_balance_risk_score"].to_markdown(index=False))
    L.append("\n**claim_status**\n")
    L.append(tables["target_balance_claim_status"].to_markdown(index=False))
    L.append("\n### Data-quality report\n")
    L.append(dq_rep.to_markdown(index=False))
    L.append("\n### Denial cohort (by billed band)\n")
    L.append(tables["denial_cohort"].to_markdown(index=False))
    L.append("\n### Feature signal (mutual information)\n")
    L.append(sig.to_markdown(index=False))
    L.append("\n### Status vs approved_amount\n")
    L.append(tables["status_vs_approved"].to_markdown(index=False))
    L.append("\n### Temporal consistency\n")
    L.append(tables["temporal_consistency"].to_markdown(index=False))
    L.append("\n### Length-of-stay by dimension\n")
    L.append(tables["los_drivers"].to_markdown(index=False))
    L.append("\n### Department acuity\n")
    L.append(tables["department_acuity"].to_markdown(index=False))
    L.append("\n### Insurance provider behaviour\n")
    L.append(tables["provider_behavior"].to_markdown(index=False))
    L.append("\n### Monthly patient flow\n")
    L.append(tables["flow_monthly"].to_markdown(index=False))

    (HERE / "PHASE2_FINDINGS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
