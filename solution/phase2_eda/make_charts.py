"""Phase 2 :: charts backing every EDA / data-quality finding.

One function per finding -> ``(key, path, caption)``; ``build_all`` returns the
ordered list for the report. House style only (``capstone.viz``): form from the
data's job, fixed palette, status colours only for claim state, takeaway titles,
one y-axis, direct labels, table shipped alongside (CSV in ``output/``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capstone import viz

from capstone import eda

CHART_DIR = Path(__file__).resolve().parent / "output" / "charts"
SRC = "Source: capstone_solution.v_visit_billing (Phase 2 EDA) - 25,000 visits / claims, 5,000 patients, 2025-01-20 to 2026-01-20."

_T: dict[str, pd.DataFrame] = {}


def _tables() -> dict[str, pd.DataFrame]:
    if not _T:
        _T.update(eda.build_all())
    return _T


# ---------------------------------------------------------------------------
# targets & class balance
# ---------------------------------------------------------------------------
def chart_target_balance() -> tuple[str, Path, str]:
    t = _tables()
    risk = t["target_balance_risk_score"].set_index("class").reindex(["Low", "Medium", "High"])
    claim = t["target_balance_claim_status"].set_index("class").reindex(["Paid", "Pending", "Rejected"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, d, title, costly in (
        (axes[0], risk, "Model A target: risk_score", "High"),
        (axes[1], claim, "Model B target: claim_status", "Rejected"),
    ):
        colors = [viz.CATEGORICAL[0] if k != costly else viz.STATUS["critical"] for k in d.index]
        bars = ax.bar(d.index, d["share_pct"], color=colors, width=0.62)
        viz.label_bars(ax, bars, d["share_pct"], fmt="{:.1f}%")
        ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=viz.INK_SECONDARY)
        ax.set_ylim(0, 70)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    axes[0].set_ylabel("Share of visits")
    return "target_balance", viz.finalize(
        fig, axes[0],
        title="Both targets are imbalanced and the costly class is the rarest",
        subtitle="High-risk visits are 20.1% of the total; Rejected claims 15.2%. Phase 3 needs class "
                 "weighting / threshold tuning, and recall on these classes is the headline metric.",
        source=SRC, out_path=CHART_DIR / "target_balance.png",
    ), "Class balance for both model targets - the expensive class (High / Rejected) is the smallest."


def _signal_chart(model_key: str, drop: list[str], title: str, subtitle: str,
                  fname: str, caption: str) -> tuple[str, Path, str]:
    d = _tables()["feature_signal"]
    d = d[d["model"].str.contains(model_key) & d["model_allowed"] & ~d["feature"].isin(drop)].copy()
    d = d.sort_values("pct_of_target_entropy", ascending=True).tail(14)
    fig, ax = viz.new_figure(9, 5.0)
    bars = ax.barh(d["feature"], d["pct_of_target_entropy"], color=viz.CATEGORICAL[0], height=0.66)
    floor = float(d["noise_pct_of_target_entropy"].median())
    ax.axvline(floor, color=viz.INK_MUTED, linewidth=1)
    ax.text(floor, len(d) - 0.3, "  permuted-target noise floor", fontsize=8, color=viz.INK_MUTED)
    for bar, v in zip(bars, d["pct_of_target_entropy"]):
        ax.text(bar.get_width() + d["pct_of_target_entropy"].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=8, color=viz.INK_SECONDARY)
    ax.set_xlabel("Share of the target's uncertainty explained (mutual information / target entropy)")
    ax.grid(False)
    ax.set_xlim(0, max(d["pct_of_target_entropy"].max() * 1.25, 0.5))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.1f}%")
    return fname, viz.finalize(
        fig, ax, title=title, subtitle=subtitle, source=SRC,
        out_path=CHART_DIR / f"{fname}.png",
    ), caption


def chart_signal_model_a() -> tuple[str, Path, str]:
    return _signal_chart(
        "Model A", drop=["risk_score"],
        title="No Model-A feature explains more than 1% of visit-risk uncertainty",
        subtitle="Mutual information with risk_score, as a share of the target's entropy. Every "
                 "Model-A-eligible feature sits at or below the permuted-target noise floor - "
                 "risk_score is effectively randomly assigned here, so Model A will not beat the base "
                 "rate and Phase 3 should set expectations accordingly.",
        fname="signal_model_a",
        caption="Model-A feature signal vs risk_score - all at the noise floor.")


def chart_signal_model_b() -> tuple[str, Path, str]:
    return _signal_chart(
        "Model B", drop=["claim_status"],
        title="Only billed amount predicts claim outcome",
        subtitle="Mutual information with claim_status, as a share of the target's entropy. "
                 "billed_amount / log / band clear the noise floor (the non-monotonic mid-band denial "
                 "pattern); every other candidate - provider, department, risk, patient history - is "
                 "at the floor. Model B is essentially a billed-amount model.",
        fname="signal_model_b",
        caption="Model-B feature signal vs claim_status - billed amount is the only signal.")


# ---------------------------------------------------------------------------
# denial cohort analysis
# ---------------------------------------------------------------------------
def chart_rejection_by_billed_band() -> tuple[str, Path, str]:
    d = _tables()["denial_cohort"]
    fig, ax = viz.new_figure(8.5, 4.4)
    bars = ax.bar(d["billed_band"], d["rejection_rate_pct"], color=viz.CATEGORICAL[0], width=0.6)
    viz.label_bars(ax, bars, d["rejection_rate_pct"], fmt="{:.1f}%")
    ax.axhline(15.19, color=viz.INK_MUTED, linewidth=1)
    ax.text(3.35, 15.9, "network avg 15.2%", fontsize=8, color=viz.INK_MUTED, ha="right")
    ax.set_ylabel("Claim rejection rate")
    ax.set_xlabel("Billed-amount band")
    ax.set_ylim(0, 27)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    return "rejection_by_billed_band", viz.finalize(
        fig, ax,
        title="Rejections peak in the mid-value band, not the largest claims",
        subtitle="Rejection rate by billed-amount band - non-monotonic: 22.7% at 15k-30k vs 4.5% below "
                 "5k and 6.5% above 30k. A 'scrutinise the big claims' rule would miss it.",
        source=SRC, out_path=CHART_DIR / "rejection_by_billed_band.png",
    ), "Claim rejection rate by billed-amount band (non-monotonic, peaks mid-band)."


def chart_rejection_flat_across_dims() -> tuple[str, Path, str]:
    d = _tables()["rejection_by_dimension"]
    labels = {"department": "Department", "insurance_provider": "Insurer", "visit_type": "Visit type",
              "risk_score": "Risk band", "age_band": "Age band", "gender": "Gender",
              "chronic_flag": "Chronic flag", "billed_band": "Billed band"}
    order = list(labels)
    fig, ax = viz.new_figure(8.5, 4.6)
    for i, dim in enumerate(order):
        sub = d[d["dimension"] == dim]
        hi = dim == "billed_band"
        ax.scatter(sub["rejection_rate_pct"], [i] * len(sub), s=70,
                   color=viz.STATUS["critical"] if hi else viz.CATEGORICAL[0],
                   alpha=0.9, zorder=3)
    ax.axvline(15.19, color=viz.INK_MUTED, linewidth=1, zorder=1)
    ax.text(15.19, len(order) - 0.3, " network avg 15.2%", fontsize=8, color=viz.INK_MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[o] for o in order])
    ax.set_xlabel("Claim rejection rate by category value")
    ax.grid(False)
    ax.set_xlim(0, 27)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    return "rejection_flat_across_dims", viz.finalize(
        fig, ax,
        title="Rejection rate is flat on every dimension except billed amount",
        subtitle="Each dot is one category value. Only the billed-amount band (red) spreads away from "
                 "the 15.2% network average - the denial signal is purely financial.",
        source=SRC, out_path=CHART_DIR / "rejection_flat_across_dims.png",
    ), "Rejection-rate spread across operational dimensions - only billed band moves."


def chart_denial_leakage_concentration() -> tuple[str, Path, str]:
    d = _tables()["denial_cohort"]
    fig, ax = viz.new_figure(8.5, 4.2)
    bars = ax.bar(d["billed_band"], d["denied_billed"] / 1e7, color=viz.SEQUENTIAL_BLUE[4], width=0.6)
    for bar, amt, sh in zip(bars, d["denied_billed"], d["denied_share_of_leakage_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{viz.money(amt)}\n{sh:.0f}%", ha="center", va="bottom", fontsize=8.5,
                color=viz.INK_SECONDARY, fontweight="bold")
    ax.set_ylabel("Denied billed value (Cr)")
    ax.set_xlabel("Billed-amount band")
    ax.set_ylim(0, d["denied_billed"].max() / 1e7 * 1.25)
    ax.grid(True, axis="y")
    return "denial_leakage_concentration", viz.finalize(
        fig, ax,
        title="70% of denial leakage is one cohort: mid-value claims",
        subtitle="Denied billed value by band. The 15k-30k band loses 5.3Cr (52.7M) - 70% of the "
                 "9.2Cr (91.8M) adjudicated-denial leakage - and is where a pre-submission triage "
                 "model pays off.",
        source=SRC, out_path=CHART_DIR / "denial_leakage_concentration.png",
    ), "Denied billed value by band - the mid band is 70% of denial leakage."


def chart_revenue_waterfall() -> tuple[str, Path, str]:
    d = _tables()["revenue_waterfall"].set_index("component")
    billed = float(d.loc["Billed", "amount"])
    parts = [("Collected", viz.STATUS["good"]), ("Pending (at risk)", viz.STATUS["warning"]),
             ("Denied (leakage)", viz.STATUS["critical"])]
    fig, ax = viz.new_figure(9, 2.9)
    left = 0.0
    for name, color in parts:
        val = float(d.loc[name, "amount"])
        ax.barh([0], [val], left=left, color=color, height=0.55, edgecolor=viz.SURFACE, linewidth=2)
        ax.text(left + val / 2, 0, f"{name}\n{viz.money(val)} ({val / billed * 100:.0f}%)",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        left += val
    ax.set_xlim(0, billed)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Billed value")
    ax.xaxis.set_major_formatter(lambda x, _: viz.money(x))
    return "revenue_waterfall", viz.finalize(
        fig, ax,
        title="57.9% of billed revenue is collected; 42.1% leaks or waits",
        subtitle=f"Of {viz.money(billed)} billed: {viz.money(float(d.loc['Denied (leakage)','amount']))} "
                 f"lost to denials, {viz.money(float(d.loc['Pending (at risk)','amount']))} unadjudicated.",
        source=SRC, out_path=CHART_DIR / "revenue_waterfall.png",
    ), "Revenue realization waterfall: billed value splits into collected, pending and denied."


# ---------------------------------------------------------------------------
# data quality: status <-> amount decoupling
# ---------------------------------------------------------------------------
def chart_approved_ratio_by_status() -> tuple[str, Path, str]:
    d = _tables()["status_vs_approved"].set_index("claim_status").reindex(["Paid", "Pending", "Rejected"])
    fig, ax = viz.new_figure(8.5, 4.0)
    mean = d["approved_ratio_mean"] * 100
    lo = d["approved_ratio_min"] * 100
    hi = d["approved_ratio_max"] * 100
    colors = [viz.CLAIM_STATUS_COLORS[s] for s in d.index]
    bars = ax.bar(d.index, mean, color=colors, width=0.55)
    ax.errorbar(range(len(d)), mean, yerr=[mean - lo, hi - mean], fmt="none",
                ecolor=viz.INK_SECONDARY, capsize=6, linewidth=1.2)
    for i, (m, l, h) in enumerate(zip(mean, lo, hi)):
        ax.text(i, min(m + 12, 116), f"mean {m:.0f}%  (range {l:.0f}-{h:.0f}%)", ha="center",
                fontsize=8.5, color=viz.INK_SECONDARY, fontweight="bold")
    ax.set_ylabel("approved / billed ratio")
    ax.set_ylim(0, 125)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    return "approved_ratio_by_status", viz.finalize(
        fig, ax,
        title="When approved_amount is present it is mechanical, given the status",
        subtitle="Paid -> exactly 100% of billed, Rejected -> exactly 0%, Pending -> a uniform 50-90% "
                 "provisional band. claim_status and approved_amount encode the same adjudication - "
                 "never derive one from the other.",
        source=SRC, out_path=CHART_DIR / "approved_ratio_by_status.png",
    ), "approved/billed ratio by claim status - deterministic for Paid & Rejected."


def chart_post_outcome_missingness() -> tuple[str, Path, str]:
    d = _tables()["status_vs_approved"].set_index("claim_status").reindex(["Paid", "Pending", "Rejected"])
    fig, ax = viz.new_figure(8.5, 4.2)
    x = np.arange(len(d))
    w = 0.38
    b1 = ax.bar(x - w / 2, d["approved_missing_pct"], w, color=viz.CATEGORICAL[0],
               label="approved_amount missing")
    b2 = ax.bar(x + w / 2, 100 - d["payment_days_present_pct"], w, color=viz.CATEGORICAL[1],
               label="payment_days missing")
    viz.label_bars(ax, b1, d["approved_missing_pct"], fmt="{:.1f}%")
    viz.label_bars(ax, b2, 100 - d["payment_days_present_pct"], fmt="{:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(d.index)
    ax.set_ylabel("Share of claims with the field missing")
    ax.set_ylim(0, 9)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    return "post_outcome_missingness", viz.finalize(
        fig, ax,
        title="approved_amount and payment_days go missing at random (~5%), regardless of status",
        subtitle="Missingness is ~MCAR: it does not track claim_status. payment_days is present on "
                 "~97% of Rejected claims too - it is not a clean 'Paid' indicator. Both are "
                 "post-outcome and never feed a model.",
        source=SRC, out_path=CHART_DIR / "post_outcome_missingness.png",
    ), "Missing rate of approved_amount / payment_days by status - flat ~5%, i.e. MCAR."


def chart_missingness_overview() -> tuple[str, Path, str]:
    d = _tables()["missingness"]
    fig, ax = viz.new_figure(8.5, 3.2 + 0.3 * len(d))
    bars = ax.barh(d["column"], d["pct_missing"], color=viz.CATEGORICAL[0], height=0.5)
    for bar, p, n in zip(bars, d["pct_missing"], d["n_missing"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{p:.2f}%  ({int(n):,})", va="center", fontsize=8, color=viz.INK_SECONDARY)
    ax.grid(False)
    ax.set_xlabel("% of rows missing")
    ax.set_xlim(0, max(d["pct_missing"]) * 1.4)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    return "missingness_overview", viz.finalize(
        fig, ax,
        title="Only two source columns have any missing values",
        subtitle="Both are post-adjudication billing fields. Every operational and clinical field on "
                 "the spine is complete - no imputation is needed for modelling inputs.",
        source=SRC, out_path=CHART_DIR / "missingness_overview.png",
    ), "Missing-value share by spine column - only approved_amount and payment_days."


# ---------------------------------------------------------------------------
# data quality: temporal
# ---------------------------------------------------------------------------
def chart_temporal_inconsistency() -> tuple[str, Path, str]:
    spine = eda.load_spine()
    lag = (spine["billing_date"] - spine["visit_date"]).dt.days
    fig, ax = viz.new_figure(9, 4.4)
    ax.hist(lag, bins=60, color=viz.CATEGORICAL[0])
    ax.axvline(0, color=viz.INK, linewidth=1)
    neg = (lag < 0).mean() * 100
    ax.text(-330, ax.get_ylim()[1] * 0.85, f"{neg:.0f}% of claims are\nbilled BEFORE the visit",
            fontsize=9, color=viz.STATUS["critical"], fontweight="bold")
    ax.set_xlabel("billing_date - visit_date (days)")
    ax.set_ylabel("Claims")
    return "temporal_inconsistency", viz.finalize(
        fig, ax,
        title="billing_date carries no real relation to visit_date",
        subtitle="The lag is centred near zero but spreads +/- a full year (sd ~150d); 49.6% of claims "
                 "are 'billed' before the visit and 48.6% of visits precede registration. visit_date "
                 "is the only usable temporal key.",
        source=SRC, out_path=CHART_DIR / "temporal_inconsistency.png",
    ), "Distribution of billing_date - visit_date: noise, ~50% negative."


def chart_date_fields_monthly() -> tuple[str, Path, str]:
    d = _tables()["date_field_monthly"].copy()
    d["month"] = pd.to_datetime(d["month"])
    d = d[(d["month"] >= "2025-02") & (d["month"] <= "2025-12")]  # full months only
    fig, ax = viz.new_figure(9, 4.4)
    series = [("visits_by_visit_date", "By visit_date", viz.CATEGORICAL[0]),
              ("claims_by_billing_date", "By billing_date", viz.CATEGORICAL[1]),
              ("registrations_by_registration_date", "By registration_date", viz.CATEGORICAL[2])]
    for col, label, color in series:
        ax.plot(d["month"], d[col], color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.set_ylim(0, d[[s[0] for s in series]].to_numpy().max() * 1.2)
    ax.set_ylabel("Records per month (full months)")
    ax.legend(loc="lower center", ncol=3)
    return "date_fields_monthly", viz.finalize(
        fig, ax,
        title="All three timestamps are independently uniform across the year",
        subtitle="Monthly record counts keyed on each date field are flat and interchangeable - there "
                 "is no ingestion cadence or seasonality encoded in any of them.",
        source=SRC, out_path=CHART_DIR / "date_fields_monthly.png",
    ), "Monthly counts by visit / billing / registration date - all flat and independent."


# ---------------------------------------------------------------------------
# data quality: floors
# ---------------------------------------------------------------------------
def chart_distribution_floors() -> tuple[str, Path, str]:
    spine = eda.load_spine()
    f = _tables()["floor_analysis"].set_index("field")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].hist(spine["length_of_stay_hours"], bins=60, color=viz.CATEGORICAL[0])
    axes[0].set_xlabel("Length of stay (hours)")
    axes[0].set_ylabel("Visits")
    axes[0].annotate(f"{int(f.loc['length_of_stay_hours','n_at_floor'])} visits\npinned at 0.5h",
                     xy=(0.5, f.loc['length_of_stay_hours','n_at_floor']), xytext=(14, 240),
                     fontsize=8, color=viz.INK_SECONDARY,
                     arrowprops=dict(arrowstyle="->", color=viz.INK_MUTED))
    axes[1].hist(spine["billed_amount"], bins=60, color=viz.CATEGORICAL[0])
    axes[1].set_xlabel("Billed amount")
    axes[1].set_ylabel("Claims")
    axes[1].annotate(f"{int(f.loc['billed_amount','n_at_floor'])} claims\npinned at 500",
                     xy=(500, f.loc['billed_amount','n_at_floor']), xytext=(22000, 200),
                     fontsize=8, color=viz.INK_SECONDARY,
                     arrowprops=dict(arrowstyle="->", color=viz.INK_MUTED))
    for ax in axes:
        ax.grid(True, axis="y")
    return "distribution_floors", viz.finalize(
        fig, axes[0],
        title="Length of stay is left-censored at 0.5h; billed amount at 500",
        subtitle="1.2% of visits sit exactly at 0.5h and 1.0% of claims exactly at 500 - capture "
                 "floors, not real minimums. Decision: keep the rows, add boolean at-floor flags, "
                 "do not impute upward.",
        source=SRC, out_path=CHART_DIR / "distribution_floors.png",
    ), "LOS and billed-amount distributions with their capture floors."


def chart_los_no_drivers() -> tuple[str, Path, str]:
    d = _tables()["los_drivers"]
    labels = {"department": "Department", "visit_type": "Visit type", "risk_score": "Risk band",
              "age_band": "Age band", "chronic_flag": "Chronic flag", "gender": "Gender"}
    order = list(labels)
    overall = eda.load_spine()["length_of_stay_hours"].mean()
    fig, ax = viz.new_figure(8.5, 4.4)
    for i, dim in enumerate(order):
        sub = d[d["dimension"] == dim]
        ax.scatter(sub["avg_los_hours"], [i] * len(sub), s=70, color=viz.CATEGORICAL[0], alpha=0.9, zorder=3)
    ax.axvline(overall, color=viz.INK_MUTED, linewidth=1)
    ax.text(overall, len(order) - 0.3, f" overall {overall:.1f}h", fontsize=8, color=viz.INK_MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[o] for o in order])
    ax.set_xlabel("Average length of stay (hours) by category value")
    ax.grid(False)
    ax.set_xlim(17, 22)
    return "los_no_drivers", viz.finalize(
        fig, ax,
        title="Length of stay has no drivers - it is flat across every dimension",
        subtitle="Average LOS by category value clusters within ~0.5h of the 19.6h overall mean, "
                 "including across ICU / ER / OPD and risk band. LOS is a near-useless Model B feature "
                 "and is excluded from Model A (unknown at admission).",
        source=SRC, out_path=CHART_DIR / "los_no_drivers.png",
    ), "Average length of stay by dimension - no meaningful variation."


# ---------------------------------------------------------------------------
# business: flow, acuity, provider
# ---------------------------------------------------------------------------
def chart_patient_flow_seasonality() -> tuple[str, Path, str]:
    d = _tables()["flow_monthly"].copy()
    d["month"] = pd.to_datetime(d["month"])
    full = d[(d["month"] >= "2025-02") & (d["month"] <= "2025-12")]
    fig, ax = viz.new_figure(9, 4.4)
    ax.plot(full["month"], full["visits"], color=viz.CATEGORICAL[0], linewidth=2, marker="o",
            markersize=4, label="Visits / month")
    ax.plot(full["month"], full["high_risk_visits"], color=viz.CATEGORICAL[1], linewidth=2, marker="o",
            markersize=4, label="High-risk visits / month")
    ax.set_ylim(0, full["visits"].max() * 1.2)
    ax.set_ylabel("Visits (full months)")
    for col, color in (("visits", viz.CATEGORICAL[0]), ("high_risk_visits", viz.CATEGORICAL[1])):
        ax.annotate(f"{full[col].iloc[-1]:,}", (full["month"].iloc[-1], full[col].iloc[-1]),
                    textcoords="offset points", xytext=(8, 0), va="center", fontsize=9,
                    fontweight="bold", color=color)
    ax.legend(loc="center left")
    return "patient_flow_seasonality", viz.finalize(
        fig, ax,
        title="No patient-flow seasonality - volume and acuity are flat all year",
        subtitle="Monthly visits hold near 2,050 and the High-risk share near 20% every month "
                 "(Jan 2025 / Jan 2026 partial and excluded). Staffing plans cannot lean on a "
                 "seasonal forecast; the lever is the risk mix, not the calendar.",
        source=SRC, out_path=CHART_DIR / "patient_flow_seasonality.png",
    ), "Monthly visit volume and High-risk subset - flat, no seasonality."


def chart_flow_day_of_week() -> tuple[str, Path, str]:
    d = _tables()["flow_day_of_week"]
    fig, ax = viz.new_figure(8.5, 4.0)
    bars = ax.bar(d["day"], d["visits"], color=viz.CATEGORICAL[0], width=0.62)
    viz.label_bars(ax, bars, d["visits"], fmt="{:,.0f}")
    ax.set_ylabel("Visits")
    ax.set_ylim(0, d["visits"].max() * 1.18)
    return "flow_day_of_week", viz.finalize(
        fig, ax,
        title="Visit volume is even across the week - no weekday or weekend pattern",
        subtitle="Total visits by day of week span just 3,528-3,637. day_of_week / is_weekend are kept "
                 "as features for completeness but carry no signal in EDA.",
        source=SRC, out_path=CHART_DIR / "flow_day_of_week.png",
    ), "Visits by day of week - flat."


def chart_department_acuity_mix() -> tuple[str, Path, str]:
    d = _tables()["department_acuity"].sort_values("high_risk_pct")
    fig, ax = viz.new_figure(9, 4.6)
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for col, name, color in (("er_pct", "ER", viz.CATEGORICAL[0]), ("opd_pct", "OPD", viz.CATEGORICAL[2]),
                             ("icu_pct", "ICU", viz.CATEGORICAL[1])):
        bars = ax.barh(y, d[col], left=left, color=color, height=0.5, label=name,
                       edgecolor=viz.SURFACE, linewidth=2)
        for bar, v in zip(bars, d[col]):
            ax.text(bar.get_x() + v / 2, bar.get_y() + bar.get_height() / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left = left + d[col].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels([f"{dep}\n({hr:.0f}% High-risk)" for dep, hr in zip(d["department"], d["high_risk_pct"])])
    ax.set_xlim(0, 100)
    ax.grid(False)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    return "department_acuity_mix", viz.finalize(
        fig, ax,
        title="Departments are near-interchangeable on acuity mix",
        subtitle="Visit-type split is ~33/33/33 in every department and the High-risk share sits in a "
                 "19-21% band. Department is a weak feature; acuity planning is a network-level problem.",
        source=SRC, out_path=CHART_DIR / "department_acuity_mix.png",
    ), "Visit-type mix and High-risk share by department - uniform."


def chart_provider_behavior() -> tuple[str, Path, str]:
    d = _tables()["provider_behavior"].sort_values("insurance_provider")
    fig, ax = viz.new_figure(9, 4.2)
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for col, name in (("paid_rate_pct", "Paid"), ("pending_rate_pct", "Pending"),
                      ("rejection_rate_pct", "Rejected")):
        bars = ax.barh(y, d[col], left=left, height=0.5, color=viz.CLAIM_STATUS_COLORS[name],
                       label=name, edgecolor=viz.SURFACE, linewidth=2)
        for bar, v in zip(bars, d[col]):
            ax.text(bar.get_x() + v / 2, bar.get_y() + bar.get_height() / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        left = left + d[col].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}\n(pays in {dd:.0f}d, p90 {p90:.0f}d)"
                        for p, dd, p90 in zip(d["insurance_provider"], d["avg_payment_days"], d["p90_payment_days"])])
    ax.set_xlim(0, 100)
    ax.grid(False)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    return "provider_behavior", viz.finalize(
        fig, ax,
        title="All four insurers adjudicate and pay alike",
        subtitle="~60% Paid / ~25% Pending / ~15% Rejected and a 12.5-day mean settlement for every "
                 "provider. provider_prior_rejection_rate is kept as an as-of feature, but there is no "
                 "counterparty to single out.",
        source=SRC, out_path=CHART_DIR / "provider_behavior.png",
    ), "Claim-outcome mix and payment speed by insurer - near-identical."


def chart_visits_per_patient() -> tuple[str, Path, str]:
    d = _tables()["visits_per_patient"]
    fig, ax = viz.new_figure(8.5, 4.0)
    bars = ax.bar(d["visits_in_window"], d["n_patients"], color=viz.CATEGORICAL[0], width=0.7)
    viz.label_bars(ax, bars, d["n_patients"], fmt="{:,.0f}")
    ax.set_xlabel("Visits per patient in the 12-month window")
    ax.set_ylabel("Patients")
    ax.set_xticks(d["visits_in_window"])
    ax.set_ylim(0, d["n_patients"].max() * 1.15)
    return "visits_per_patient", viz.finalize(
        fig, ax,
        title="Patients average 5 visits in the year - enough history for as-of features",
        subtitle="97% of patients have >=2 visits, so prior-visit / prior-rejection / days-since-last "
                 "features are populated for most rows; only 149 patients (3%) are first-and-only visits.",
        source=SRC, out_path=CHART_DIR / "visits_per_patient.png",
    ), "Distribution of visits per patient - supports the patient-history feature block."


# ---------------------------------------------------------------------------
BUILDERS = [
    chart_target_balance,
    chart_signal_model_a,
    chart_signal_model_b,
    chart_rejection_by_billed_band,
    chart_rejection_flat_across_dims,
    chart_denial_leakage_concentration,
    chart_revenue_waterfall,
    chart_approved_ratio_by_status,
    chart_post_outcome_missingness,
    chart_missingness_overview,
    chart_temporal_inconsistency,
    chart_date_fields_monthly,
    chart_distribution_floors,
    chart_los_no_drivers,
    chart_patient_flow_seasonality,
    chart_flow_day_of_week,
    chart_department_acuity_mix,
    chart_provider_behavior,
    chart_visits_per_patient,
]


def build_all() -> list[tuple[str, Path, str]]:
    viz.apply_house_style()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for b in BUILDERS:
        key, path, caption = b()
        print(f"      {path.name}")
        out.append((key, path, caption))
    return out


if __name__ == "__main__":
    build_all()
