"""Phase 2 :: assemble PHASE2_FINDINGS.md from the analysis tables + charts.

Kept out of the notebook so the notebook stays thin. ``write_findings`` takes the
already-computed tables, the chart list from ``make_charts.build_all`` and the
data-quality report, and writes the themed markdown document with every chart
embedded and the supporting numbers in an appendix.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from capstone import data_quality as dq
from capstone import features as feat
from capstone.db import SETTINGS

HERE = Path(__file__).resolve().parent


def _img(charts: dict, key: str) -> str:
    path, caption = charts[key]
    rel = Path(path).relative_to(HERE).as_posix()
    return f"![{caption}]({rel})\n\n*{caption}*\n"


def write_findings(tables: dict[str, pd.DataFrame],
                   chart_list: list[tuple[str, Path, str]],
                   dq_rep: pd.DataFrame,
                   fframe: pd.DataFrame,
                   out_path: str | Path | None = None) -> Path:
    out_path = Path(out_path) if out_path else (HERE / "PHASE2_FINDINGS.md")
    charts = {k: (p, c) for k, p, c in chart_list}
    rb = tables["target_balance_risk_score"].set_index("class")
    cb = tables["target_balance_claim_status"].set_index("class")
    dc = tables["denial_cohort"].set_index("billed_band")
    sig = tables["feature_signal"]
    fa = len(feat.model_features(fframe, "A"))
    fb = len(feat.model_features(fframe, "B"))
    started = datetime.now(timezone.utc)

    L: list[str] = []
    L.append("# Phase 2 - Exploratory Data Analysis & Data Quality :: Findings\n")
    L.append("*Hospital Operations & Revenue Risk Intelligence Platform - modelling readiness*\n")
    L.append(f"- Generated: {started.isoformat(timespec='seconds')}")
    L.append(f"- Source: `{SETTINGS.database}` / `{SETTINGS.schema}.v_visit_billing` - "
             "25,000 visits, 25,000 claims, 5,000 patients (2025-01-20 to 2026-01-20)")
    L.append("- Driven by `phase2_eda/phase2.ipynb`; reusable logic in "
             "`capstone.eda` / `capstone.features` / `capstone.data_quality`.")
    L.append("- Every finding is backed by a chart in `output/charts/`; supporting numbers are in "
             "the appendix and as CSVs in `output/`.")
    L.append("- Artefacts for Phase 3: `feature_spec.yaml` (catalogue + leakage register), "
             "`capstone.data_quality` (validators), `output/feature_frame.parquet` (as-of matrix).\n")

    L.append("## Executive summary\n")
    L.append("1. **Model A (visit risk) has no learnable signal.** Every eligible feature sits at "
             "the permuted-target noise floor (best < 0.5% of target entropy). `risk_score` is "
             "effectively randomly assigned - Phase 3 should expect near-base-rate performance and "
             "treat Model A as a calibrated monitoring baseline, not a predictor.")
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
             "real but narrow signal - a regularised linear model on `billed_amount` + a band term "
             "is the honest baseline; gradient boosting will mostly re-learn the band shape.\n")

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
    L.append("\nValidators for all of the above live in `capstone/data_quality.py` "
             "(`validate(df)`, `add_quality_flags(df)`, `apply_training_exclusions(df)`), "
             "importable by Phase 3 and the Phase 5/6 request gate.\n")

    L.append("---\n")
    L.append("## 3. Business analyses\n")
    for key in ("revenue_waterfall", "rejection_by_billed_band", "rejection_flat_across_dims",
                "denial_leakage_concentration", "patient_flow_seasonality", "flow_day_of_week",
                "department_acuity_mix", "los_no_drivers", "provider_behavior"):
        L.append(_img(charts, key))

    L.append("---\n")
    L.append("## 4. Feature engineering & leakage register\n")
    L.append(f"`feature_spec.yaml` catalogues **{len(feat.FEATURE_SPEC)} fields** - "
             f"{fa} eligible for Model A, {fb} for Model B - each with a definition, a source, an "
             f"as-of rule keyed on `visit_date`, a dtype and a per-model leakage verdict. The as-of "
             f"matrix is materialised at `output/feature_frame.parquet` "
             f"({fframe.shape[0]} rows x {fframe.shape[1]} cols).\n")
    L.append(_img(charts, "visits_per_patient"))
    L.append("**Leakage rules (enforced by `capstone.features.leakage_violations`, run in the "
             "notebook):**\n")
    L.append("- `visit_date` is the sole temporal key. No feature derives from `billing_date` or "
             "`registration_date` (including `billing_lag_days`, days-since-registration).")
    L.append("- No post-outcome field (`approved_amount`, `payment_days`, `claim_status`, "
             "`collected/leakage/pending_amount`) feeds Model A or the pre-submission Model B.")
    L.append("- `length_of_stay_hours` and `billed_amount` are **excluded from Model A** (not "
             "known at admission / not an operational field) but **allowed for Model B** "
             "(known before the claim is filed).")
    L.append("- `risk_score` is the Model A target and a legitimate Model B input.")
    L.append("- Patient- and provider-history features count only visits with `visit_date` "
             "strictly earlier than the current visit.\n")
    L.append("### Feature catalogue\n")
    L.append(tables["feature_catalogue"].to_markdown(index=False))

    L.append("\n---\n")
    L.append("## Appendix - supporting tables\n")
    L.append("### Column profile (spine)\n")
    L.append(tables["column_profile"].to_markdown(index=False))
    L.append("\n### Target balance\n\n**risk_score**\n")
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

    out_path.write_text("\n".join(L) + "\n")
    return out_path
