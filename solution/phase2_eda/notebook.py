"""Builds ``phase2.ipynb`` - the Phase 2 review artefact.

The notebook is the deliverable (per `CLAUDE.md`: notebooks are the phase artefact
from Phase 2 on). It stays thin - it imports the reusable logic from
``capstone.eda`` / ``capstone.features`` / ``capstone.data_quality`` and the
phase-local ``make_charts`` / ``report`` helpers, then orchestrates, explains and
displays inline. ``run_phase2.py`` regenerates and executes it.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent

# (kind, source) pairs. kind in {"md", "code"}.
CELLS: list[tuple[str, str]] = [
    ("md", """\
# Phase 2 - Exploratory Data Analysis & Data Quality

*Hospital Operations & Revenue Risk Intelligence Platform*

**Goal:** understand hospital operations and financial dynamics, and lock down
feature definitions and leakage rules before modelling.

This notebook is thin: all reusable logic lives in `capstone.eda`,
`capstone.features` and `capstone.data_quality`; the per-finding charts are in
`phase2_eda/make_charts.py` and the report assembler in `phase2_eda/report.py`.
It runs top-to-bottom from a clean kernel and is safe to re-run - it rebuilds
every CSV, chart, `feature_spec.yaml` and `PHASE2_FINDINGS.md`.

**Inputs:** `capstone_solution.v_visit_billing` (Phase 1).
**Outputs:** `phase2_eda/output/` (CSVs, `feature_frame.parquet`, `charts/`),
`phase2_eda/feature_spec.yaml`, `phase2_eda/PHASE2_FINDINGS.md`."""),

    ("code", """\
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

from capstone import eda, features as feat, data_quality as dq, viz
import make_charts, report

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)
viz.apply_house_style()

OUT = Path("output"); (OUT / "charts").mkdir(parents=True, exist_ok=True)
SPEC_YAML = Path("feature_spec.yaml")
FINDINGS_MD = Path("PHASE2_FINDINGS.md")"""),

    ("md", "## 1. Pull the joined dataset\n\nOne row per visit from the Phase 1 spine "
           "(`v_visit_billing`): visit + patient + billing, with derived bands and ratios."),
    ("code", """\
spine = eda.load_spine()
print(spine.shape)
spine.head()"""),

    ("md", "### 1.1 Profile - distributions, missingness, dtypes (per column)"),
    ("code", """\
tables = eda.build_all(spine)          # every analysis -> {name: DataFrame}
eda.export_tables(tables, OUT)         # ... written to output/*.csv
tables["column_profile"]"""),
    ("code", """\
print("Columns with missing values:")
display(tables["missingness"])
print("\\nOnly the two post-adjudication billing fields are ever null; every "
      "operational / clinical field is complete.")"""),

    ("md", """\
## 2. Model targets & feature signal

Two targets: **`risk_score`** (Model A, visit risk) and **`claim_status`**
(Model B, pre-submission claim outcome). Class balance first, then a
mutual-information screen of every candidate feature against each target with a
permuted-target run as the noise floor."""),
    ("code", """\
fframe = feat.build_feature_frame(spine)
fframe.to_parquet(OUT / "feature_frame.parquet", index=False)
make_charts._T.update(tables)          # reuse computed tables in the chart layer
charts = make_charts.build_all()       # -> list[(key, path, caption)]
_C = {k: p for k, p, _ in charts}
display(tables["target_balance_risk_score"], tables["target_balance_claim_status"])
Image(str(_C["target_balance"]))"""),
    ("code", 'Image(str(_C["signal_model_a"]))'),
    ("code", 'Image(str(_C["signal_model_b"]))'),
    ("code", """\
sig = tables["feature_signal"]
sig[sig.model.str.contains("B") & sig.model_allowed].head(8)"""),
    ("md", """\
**Read.** Model A: no eligible feature clears the noise floor (< 0.5% of target
entropy) - `risk_score` is effectively random, so Model A is a calibrated
base-rate baseline, not a predictor. Model B: only `billed_amount` /
`billed_band` carry signal, and the relationship is *non-monotonic* (next
section). Phase 3 should set expectations accordingly."""),

    ("md", """\
## 3. Data quality - Phase 1 findings formalised

Phase 1 flagged four issues in *values and timelines*. Here each is quantified
and given a written handling decision (`capstone.data_quality.HANDLING`), with
reusable validators in `capstone.data_quality.RULES`."""),
    ("code", """\
dq_rep = dq.validate(spine)
dq_rep.to_csv(OUT / "data_quality_report.csv", index=False)
dq_rep"""),
    ("md", "### 3.1 Capture floors - length_of_stay 0.5h, billed_amount 500"),
    ("code", "display(tables['floor_analysis']); Image(str(_C['distribution_floors']))"),
    ("md", "### 3.2 claim_status <-> approved_amount decoupling"),
    ("code", "display(tables['status_vs_approved']); Image(str(_C['approved_ratio_by_status']))"),
    ("code", "Image(str(_C['post_outcome_missingness']))"),
    ("code", "Image(str(_C['missingness_overview']))"),
    ("md", "### 3.3 Temporal inconsistency - `visit_date` is the only trusted key"),
    ("code", "display(tables['temporal_consistency']); Image(str(_C['temporal_inconsistency']))"),
    ("code", "Image(str(_C['date_fields_monthly']))"),
    ("md", "### 3.4 Handling policy"),
    ("code", """\
display(Markdown(
    "| Finding | Decision | Rationale |\\n|---|---|---|\\n" +
    "\\n".join(f"| {h.finding} | **{h.decision}** | {h.rationale} |" for h in dq.HANDLING)
))"""),
    ("code", """\
# the validators are importable and composable
flagged = dq.add_quality_flags(spine)
kept, excluded = dq.apply_training_exclusions(spine)
print(f"dq_error rows: {int(flagged.dq_error.sum())}   "
      f"dq_warn rows: {int(flagged.dq_warn.sum())}")
print(f"training exclusions (structural ERROR only): {len(excluded)} of {len(spine)}")"""),

    ("md", """\
## 4. Business analyses

Patient-flow seasonality, length-of-stay drivers, department acuity mix,
insurance provider payment behaviour, denial cohort analysis, revenue
waterfall."""),
    ("code", "display(tables['revenue_waterfall']); Image(str(_C['revenue_waterfall']))"),
    ("md", "### 4.1 Denial cohort - where rejections and leakage concentrate"),
    ("code", "display(tables['denial_cohort']); Image(str(_C['rejection_by_billed_band']))"),
    ("code", "Image(str(_C['rejection_flat_across_dims']))"),
    ("code", "Image(str(_C['denial_leakage_concentration']))"),
    ("md", "### 4.2 Patient flow - seasonality & day-of-week"),
    ("code", "display(tables['flow_monthly']); Image(str(_C['patient_flow_seasonality']))"),
    ("code", "Image(str(_C['flow_day_of_week']))"),
    ("md", "### 4.3 Department acuity mix & length-of-stay drivers"),
    ("code", "display(tables['department_acuity']); Image(str(_C['department_acuity_mix']))"),
    ("code", "display(tables['los_drivers']); Image(str(_C['los_no_drivers']))"),
    ("md", "### 4.4 Insurance provider payment behaviour"),
    ("code", "display(tables['provider_behavior']); Image(str(_C['provider_behavior']))"),

    ("md", """\
## 5. Feature engineering catalogue & leakage register

`capstone.features.FEATURE_SPEC` is the catalogue: each field has a definition, a
source, an **as-of rule keyed on `visit_date`**, a dtype and a **per-model
leakage verdict** (`allow` / `exclude` / `target`). It is written to
`feature_spec.yaml`. The as-of feature matrix (`feature_frame.parquet`) counts
only visits strictly earlier than the current `visit_date` for all patient- and
provider-history features."""),
    ("code", """\
spec_path = feat.write_feature_spec_yaml(SPEC_YAML)
print("wrote", spec_path, "-", len(feat.FEATURE_SPEC), "fields")
print("Model A eligible:", len(feat.model_features(fframe, "A")))
print("Model B eligible:", len(feat.model_features(fframe, "B")))
tables["feature_catalogue"]"""),
    ("code", "display(tables['visits_per_patient']); Image(str(_C['visits_per_patient']))"),
    ("md", "### 5.1 Leakage self-check\n\nNo post-outcome or not-yet-known field may be eligible "
           "for either model. Phase 3 re-runs this before training."),
    ("code", """\
violations = feat.leakage_violations(fframe)
assert not violations, violations
print("OK - leakage register holds:")
for m in ("A", "B"):
    print(f"  Model {m}: {len(feat.model_features(fframe, m))} eligible features, 0 post-outcome")"""),

    ("md", "## 6. Generate PHASE2_FINDINGS.md"),
    ("code", """\
path = report.write_findings(tables, charts, dq_rep, fframe, FINDINGS_MD)
print("wrote", path)"""),

    ("md", """\
## 7. Exit criteria

- [x] Every feature has a definition, a source, an as-of rule and a per-model
      leakage verdict (`feature_spec.yaml`).
- [x] Target distributions and class balance documented for **both** `risk_score`
      and `claim_status`, with charts.
- [x] Every quantitative claim in `PHASE2_FINDINGS.md` has a chart.
- [x] Idempotent: this notebook (and `run_phase2.py`) rebuilds everything from
      `v_visit_billing`.
- [x] Data-quality findings quantified with a handling decision; validators in
      `capstone.data_quality`.

**Handoff to Phase 3:** `feature_frame.parquet` + `feature_spec.yaml` +
`capstone.data_quality`. Time-split on `visit_date`. Expect Model A near
base-rate; Model B ~ a `billed_amount` model."""),
]


def build(path: str | Path | None = None) -> Path:
    path = Path(path) if path else (HERE / "phase2.ipynb")
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
