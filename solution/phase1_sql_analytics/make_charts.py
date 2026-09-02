"""Phase 1 :: charts backing every C-suite finding.

Reads the Phase 1 views and writes PNGs to ``output/charts/``. Each function
owns one finding; ``build_all`` returns an ordered list of
``(key, path, caption)`` for the report to embed.

Charting rules follow ``capstone.viz`` (fixed CVD-validated palette, status
colours reserved for claim/severity state, direct labels on every bar).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from capstone import viz
from capstone.db import engine

CHART_DIR = Path(__file__).resolve().parent / "output" / "charts"
SRC = "Source: capstone_solution analytics layer (Phase 1), 25,000 visits / claims, 2025."


def _q(sql: str) -> pd.DataFrame:
    eng = engine()
    try:
        return pd.read_sql(sql, eng)
    finally:
        eng.dispose()


# ---------------------------------------------------------------------------
def chart_revenue_waterfall() -> tuple[str, Path, str]:
    d = _q("""
        SELECT SUM(collected_amount) collected,
               SUM(pending_amount)   pending,
               SUM(leakage_amount)   leakage,
               SUM(billed_amount)    billed
        FROM v_visit_billing
    """).iloc[0]
    billed = float(d.billed)
    parts = [
        ("Collected", float(d.collected), viz.STATUS["good"]),
        ("Pending (at risk)", float(d.pending), viz.STATUS["warning"]),
        ("Denied (leakage)", float(d.leakage), viz.STATUS["critical"]),
    ]
    fig, ax = viz.new_figure(9, 2.9)
    left = 0.0
    for name, val, color in parts:
        ax.barh([0], [val], left=left, color=color, height=0.55,
                edgecolor=viz.SURFACE, linewidth=2)
        ax.text(left + val / 2, 0, f"{name}\n{viz.money(val)}  ({val / billed * 100:.0f}%)",
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
        title=f"Only {d.collected / billed * 100:.0f}% of billed revenue is collected",
        subtitle=f"Of {viz.money(billed)} billed, {viz.money(float(d.leakage))} is lost to denials and "
                 f"{viz.money(float(d.pending))} sits unadjudicated.",
        source=SRC, out_path=CHART_DIR / "revenue_waterfall.png",
    ), "Revenue realization: billed value splits into collected, pending and denied."


def chart_realization_trend() -> tuple[str, Path, str]:
    d = _q("SELECT billing_month, realization_rate_pct, leakage_rate_pct "
           "FROM v_revenue_realization_monthly ORDER BY billing_month")
    d = d[d["billing_month"].notna()].copy()
    d["billing_month"] = pd.to_datetime(d["billing_month"])
    fig, ax = viz.new_figure(9, 4.4)
    ax.plot(d.billing_month, d.realization_rate_pct, color=viz.CATEGORICAL[0],
            linewidth=2, marker="o", markersize=4, label="Realization rate")
    ax.plot(d.billing_month, d.leakage_rate_pct, color=viz.STATUS["critical"],
            linewidth=2, marker="o", markersize=4, label="Leakage rate")
    for col in ("realization_rate_pct", "leakage_rate_pct"):
        ax.annotate(f"{d[col].iloc[-1]:.0f}%",
                    (d.billing_month.iloc[-1], d[col].iloc[-1]),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=9, fontweight="bold",
                    color=viz.CATEGORICAL[0] if col.startswith("real") else viz.STATUS["critical"])
    ax.set_ylim(0, max(d.realization_rate_pct.max(), 65) + 8)
    ax.set_ylabel("% of billed value")
    ax.legend(loc="center left")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    return "realization_trend", viz.finalize(
        fig, ax,
        title="Revenue realization is flat month to month, near 58%",
        subtitle="No seasonal recovery - the leakage is structural, not a timing effect.",
        source=SRC, out_path=CHART_DIR / "realization_trend.png",
    ), "Monthly realization vs leakage rate."


def chart_department_billed_collected() -> tuple[str, Path, str]:
    d = _q("SELECT department, total_billed, total_collected, realization_rate_pct "
           "FROM v_department_performance ORDER BY total_billed")
    fig, ax = viz.new_figure(9, 4.6)
    y = range(len(d))
    h = 0.38
    b1 = ax.barh([i + h / 2 for i in y], d.total_billed, height=h,
                 color=viz.CATEGORICAL[0], label="Billed")
    b2 = ax.barh([i - h / 2 for i in y], d.total_collected, height=h,
                 color=viz.CATEGORICAL[1], label="Collected")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d.department)
    ax.grid(False)
    ax.set_xlim(0, float(d.total_billed.max()) * 1.18)
    ax.xaxis.set_major_formatter(lambda x, _: viz.money(x))
    for bar, val in zip(b1, d.total_billed):
        ax.text(val, bar.get_y() + bar.get_height() / 2, "  " + viz.money(val),
                va="center", fontsize=8, color=viz.INK_SECONDARY)
    for bar, val, rr in zip(b2, d.total_collected, d.realization_rate_pct):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f"  {viz.money(val)} ({rr:.0f}%)",
                va="center", fontsize=8, color=viz.INK_SECONDARY)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    return "department_billed_collected", viz.finalize(
        fig, ax,
        title="Every department realizes ~57-58% of billings - the gap is network-wide",
        subtitle="Billed vs collected by department; realization rate in parentheses.",
        source=SRC, out_path=CHART_DIR / "department_billed_collected.png",
    ), "Billed vs collected revenue by department."


def chart_provider_claim_mix() -> tuple[str, Path, str]:
    d = _q("SELECT insurance_provider, paid_rate_pct, pending_rate_pct, rejection_rate_pct, "
           "avg_payment_days FROM v_insurance_provider_behavior ORDER BY insurance_provider")
    fig, ax = viz.new_figure(9, 4.2)
    y = range(len(d))
    left = [0.0] * len(d)
    for col, name in (("paid_rate_pct", "Paid"), ("pending_rate_pct", "Pending"),
                      ("rejection_rate_pct", "Rejected")):
        bars = ax.barh(list(y), d[col], left=left, height=0.5,
                       color=viz.CLAIM_STATUS_COLORS[name], label=name,
                       edgecolor=viz.SURFACE, linewidth=2)
        for bar, val in zip(bars, d[col]):
            ax.text(bar.get_x() + val / 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
        left = [a + b for a, b in zip(left, d[col])]
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{p}\n(pays in {dd:.0f}d)" for p, dd in zip(d.insurance_provider, d.avg_payment_days)])
    ax.set_xlim(0, 100)
    ax.grid(False)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    return "provider_claim_mix", viz.finalize(
        fig, ax,
        title="All four insurers behave alike: ~60% paid, ~25% pending, ~15% rejected",
        subtitle="Claim outcome mix and average payment speed by provider - no outlier to renegotiate first.",
        source=SRC, out_path=CHART_DIR / "provider_claim_mix.png",
    ), "Claim outcome mix by insurance provider."


def chart_rejection_by_billed_band() -> tuple[str, Path, str]:
    d = _q("SELECT value, claims, rejection_rate_pct FROM v_claim_rejection_analysis "
           "WHERE dimension = 'billed_band' ORDER BY value")
    d["label"] = d["value"].str.replace(r"^\d+\.\s*", "", regex=True)
    fig, ax = viz.new_figure(8.5, 4.4)
    bars = ax.bar(d.label, d.rejection_rate_pct, color=viz.CATEGORICAL[0], width=0.6)
    viz.label_bars(ax, bars, d.rejection_rate_pct, fmt="{:.1f}%")
    ax.set_ylabel("Claim rejection rate")
    ax.set_xlabel("Billed amount band")
    ax.set_ylim(0, d.rejection_rate_pct.max() * 1.25)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    return "rejection_by_billed_band", viz.finalize(
        fig, ax,
        title="Rejections peak in the mid-value band, not the largest claims",
        subtitle="Rejection rate by billed-amount band - the pattern is non-monotonic, so a simple "
                 "'scrutinise big claims' rule would miss it.",
        source=SRC, out_path=CHART_DIR / "rejection_by_billed_band.png",
    ), "Claim rejection rate by billed-amount band."


def chart_rejection_flat_across_dims() -> tuple[str, Path, str]:
    d = _q("""SELECT dimension, value, rejection_rate_pct
              FROM v_claim_rejection_analysis
              WHERE dimension IN ('department','insurance_provider','visit_type','risk_score')""")
    labels = {"department": "Department", "insurance_provider": "Insurer",
              "visit_type": "Visit type", "risk_score": "Risk band"}
    order = list(labels)
    fig, ax = viz.new_figure(8.5, 4.2)
    for i, dim in enumerate(order):
        sub = d[d.dimension == dim]
        ax.scatter(sub.rejection_rate_pct, [i] * len(sub), s=70,
                   color=viz.CATEGORICAL[0], alpha=0.85, zorder=3)
    ax.axvline(15.2, color=viz.INK_MUTED, linewidth=1, linestyle="-", zorder=1)
    ax.text(15.2, len(order) - 0.4, " network avg 15.2%", fontsize=8, color=viz.INK_MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[o] for o in order])
    ax.set_xlabel("Claim rejection rate by category value")
    ax.grid(False)
    ax.set_xlim(0, 25)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    return "rejection_flat_across_dims", viz.finalize(
        fig, ax,
        title="Rejection rate barely moves across departments, insurers, visit types or risk",
        subtitle="Each dot is one category value - all cluster near the 15% network average. "
                 "Denial drivers are not in these dimensions.",
        source=SRC, out_path=CHART_DIR / "rejection_flat_across_dims.png",
    ), "Rejection rate spread across operational dimensions."


def chart_status_vs_approved() -> tuple[str, Path, str]:
    d = _q("""SELECT claim_status,
                     COUNT(*) FILTER (WHERE approved_amount IS NOT NULL) AS with_approved,
                     COUNT(*) FILTER (WHERE approved_amount IS NULL)     AS without_approved
              FROM v_visit_billing GROUP BY claim_status ORDER BY claim_status""")
    d["total"] = d.with_approved + d.without_approved
    d["pct_missing"] = 100 * d.without_approved / d.total
    fig, ax = viz.new_figure(8.5, 4.0)
    y = range(len(d))
    xmax = float(d.total.max())
    b1 = ax.barh(list(y), d.with_approved, color=viz.CATEGORICAL[0], height=0.5,
                 label="approved_amount present", edgecolor=viz.SURFACE, linewidth=2)
    b2 = ax.barh(list(y), d.without_approved, left=d.with_approved, color=viz.STATUS["warning"],
                 height=0.5, label="approved_amount missing", edgecolor=viz.SURFACE, linewidth=2)
    for bar in b1:
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}", ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
    for bar in b2:
        val = bar.get_width()
        # small segments: label outside the bar end (relief rule)
        ax.text(bar.get_x() + val + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(val):,} missing", ha="left", va="center",
                fontsize=8.5, color=viz.INK_SECONDARY, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d.claim_status)
    ax.set_xlim(0, xmax * 1.16)
    ax.grid(False)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    return "status_vs_approved", viz.finalize(
        fig, ax,
        title="claim_status and approved_amount disagree",
        subtitle="95% of 'Pending' claims already carry an approved value and 817 'Paid' claims carry "
                 "none - the two fields cannot be trusted as a pair.",
        source=SRC, out_path=CHART_DIR / "status_vs_approved.png",
    ), "Presence of approved_amount within each claim status."


def chart_distribution_floors() -> tuple[str, Path, str]:
    d = _q("SELECT length_of_stay_hours, billed_amount FROM v_visit_billing")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].hist(d.length_of_stay_hours, bins=60, color=viz.CATEGORICAL[0])
    axes[0].set_xlabel("Length of stay (hours)")
    axes[0].set_ylabel("Visits")
    axes[0].annotate("300 visits pinned\nat exactly 0.5h",
                     xy=(0.5, 300), xytext=(12, 260), fontsize=8, color=viz.INK_SECONDARY,
                     arrowprops=dict(arrowstyle="->", color=viz.INK_MUTED))
    axes[1].hist(d.billed_amount, bins=60, color=viz.CATEGORICAL[0])
    axes[1].set_xlabel("Billed amount")
    axes[1].set_ylabel("Claims")
    axes[1].annotate("243 claims pinned\nat exactly 500",
                     xy=(500, 243), xytext=(20000, 210), fontsize=8, color=viz.INK_SECONDARY,
                     arrowprops=dict(arrowstyle="->", color=viz.INK_MUTED))
    for ax in axes:
        ax.grid(True, axis="y")
    return "distribution_floors", viz.finalize(
        fig, axes[0],
        title="Both length-of-stay and billed amount are clipped at a hard floor",
        subtitle="Left: LOS floors at 0.5h. Right: billed amount floors at 500. Likely capture "
                 "artefacts to clean in Phase 2, not real minimums.",
        source=SRC, out_path=CHART_DIR / "distribution_floors.png",
    ), "Distributions of length of stay and billed amount, showing floor artefacts."


def chart_data_quality() -> tuple[str, Path, str]:
    d = _q("SELECT check_name, severity, pct_flagged, records_flagged "
           "FROM v_data_quality_report WHERE records_flagged > 0 ORDER BY pct_flagged")
    fig, ax = viz.new_figure(9, 4.8)
    colors = [viz.SEVERITY_COLORS[s] for s in d.severity]
    bars = ax.barh(d.check_name, d.pct_flagged, color=colors, height=0.62)
    for bar, pct, n in zip(bars, d.pct_flagged, d.records_flagged):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%  ({int(n):,})", va="center", fontsize=8, color=viz.INK_SECONDARY)
    ax.grid(False)
    ax.set_xlabel("% of scoped rows flagged")
    ax.set_xlim(0, max(d.pct_flagged) * 1.28)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    handles = [plt.Rectangle((0, 0), 1, 1, color=viz.SEVERITY_COLORS[s]) for s in ("ERROR", "WARN", "INFO")]
    ax.legend(handles, ["ERROR", "WARN", "INFO"], loc="lower right", ncol=3)
    return "data_quality", viz.finalize(
        fig, ax,
        title="One hard error, plus timeline fields that cannot be trusted",
        subtitle="Automated data-quality checks with at least one flagged row. ~49% of records have "
                 "billing-before-visit or visit-before-registration.",
        source=SRC, out_path=CHART_DIR / "data_quality.png",
    ), "Phase 1 data-quality checks, by severity and share of rows flagged."


def chart_patient_flow_monthly() -> tuple[str, Path, str]:
    d = _q("""SELECT visit_month, SUM(visits) visits, SUM(high_risk_visits) high_risk
              FROM v_patient_flow_monthly GROUP BY visit_month ORDER BY visit_month""")
    d = d[d.visit_month.notna()].copy()
    d["visit_month"] = pd.to_datetime(d["visit_month"])
    fig, ax = viz.new_figure(9, 4.4)
    ax.plot(d.visit_month, d.visits, color=viz.CATEGORICAL[0], linewidth=2, marker="o",
            markersize=4, label="All visits")
    ax.plot(d.visit_month, d.high_risk, color=viz.CATEGORICAL[1], linewidth=2, marker="o",
            markersize=4, label="High-risk visits")
    ax.set_ylim(0, d.visits.max() * 1.15)
    ax.set_ylabel("Visits per month")
    ax.legend(loc="center left")
    return "patient_flow_monthly", viz.finalize(
        fig, ax,
        title="Visit volume is stable month to month; ~20% are High-risk",
        subtitle="Monthly visits and High-risk subset across the network (partial months at each end).",
        source=SRC, out_path=CHART_DIR / "patient_flow_monthly.png",
    ), "Monthly patient-flow volume and High-risk subset."


BUILDERS = [
    chart_revenue_waterfall,
    chart_realization_trend,
    chart_department_billed_collected,
    chart_provider_claim_mix,
    chart_rejection_by_billed_band,
    chart_rejection_flat_across_dims,
    chart_status_vs_approved,
    chart_distribution_floors,
    chart_patient_flow_monthly,
    chart_data_quality,
]


def build_all() -> list[tuple[str, Path, str]]:
    viz.apply_house_style()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for builder in BUILDERS:
        key, path, caption = builder()
        print(f"      {path.name}")
        results.append((key, path, caption))
    return results


if __name__ == "__main__":
    build_all()
