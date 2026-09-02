"""Builds ``phase3.ipynb`` - the Phase 3 review artefact.

The notebook is the deliverable (per `CLAUDE.md`: notebooks are the phase
artefact from Phase 2 on). It stays thin - it imports the reusable logic from
``capstone.modeling`` and the phase-local ``make_charts`` / ``report`` helpers,
then orchestrates, explains and displays inline. ``run_phase3.py`` regenerates
and executes it.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent

CELLS: list[tuple[str, str]] = [
    ("md", """\
# Phase 3 - Model Development (Classification)

*Hospital Operations & Revenue Risk Intelligence Platform*

**Goal:** two calibrated, time-validated classifiers.

* **Model A - visit risk** (`risk_score` Low/Medium/High): operational + clinical
  + patient-history features only, for staffing / bed planning / prioritisation.
* **Model B - pre-submission claim outcome** (`claim_status` Paid/Pending/
  Rejected): everything knowable before the claim is filed, for pre-submission
  denial triage.

This notebook is thin: all reusable logic lives in `capstone.modeling`; the
per-finding charts are in `phase3_models/make_charts.py` and the report
assembler in `phase3_models/report.py`. It runs top-to-bottom from a clean
kernel and is safe to re-run - it rebuilds every artefact, CSV, chart and
`PHASE3_FINDINGS.md`.

**Input:** `phase2_eda/output/feature_frame.parquet` (Phase 2 as-of feature
matrix) + `capstone.features` leakage register.
**Outputs:** `phase3_models/models/` (pipelines, calibrated wrappers,
`training_manifest.json`), `phase3_models/output/` (predictions, metrics,
`charts/`), `phase3_models/PHASE3_FINDINGS.md`."""),

    ("code", """\
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

from capstone import features as feat, modeling as M, viz
import make_charts, report

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)
viz.apply_house_style()

HERE = Path.cwd()
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)
OUT = HERE / "output"; (OUT / "charts").mkdir(parents=True, exist_ok=True)
PARQUET = HERE.parent / "phase2_eda" / "output" / "feature_frame.parquet"
FINDINGS_MD = HERE / "PHASE3_FINDINGS.md\""""),

    ("md", "## 1. Load the Phase 2 feature frame and split on `visit_date`\n\n"
           "The as-of matrix from Phase 2 - one row per visit, patient/provider history counted "
           "only from strictly earlier visits. Split is calendar-anchored: 9 months train, "
           "1 validate, 2 test. No shuffle."),
    ("code", """\
df = pd.read_parquet(PARQUET)
df["visit_date"] = pd.to_datetime(df["visit_date"])
splits = M.time_split(df)
splits.describe()"""),

    ("md", "### 1.1 Leakage self-check\n\nThe Phase 2 register is the contract. `train_model` "
           "re-runs this and refuses to fit if it fails."),
    ("code", """\
violations = feat.leakage_violations(splits.train)
assert not violations, violations
for m in ("A", "B"):
    n, c = M.feature_lists(splits.train, m)
    print(f"Model {m}: {len(n)} numeric + {len(c)} categorical features, 0 leakage violations")"""),

    ("md", """\
## 2. Train both models

Four candidates each - majority baseline, domain simple-rule baseline,
regularised logistic regression, gradient-boosted trees - all evaluated on the
held-out test window. A learned model is shipped only if it beats the majority
baseline's balanced accuracy by >= 0.02 and the simple rule's macro-F1;
otherwise the majority baseline ships as a base-rate monitor. The selected model
is then calibrated (sigmoid) on the validation month."""),
    ("code", """\
results = {k: M.train_model(splits, k) for k in ("A", "B")}
for k, r in results.items():
    print(f"Model {k}: selected '{r.chosen}'  |  "
          f"beats majority balanced-acc: {r.beats_majority_balanced_accuracy}  |  "
          f"beats simple-rule macro-F1: {r.beats_simple_rule_macro_f1}")"""),

    ("md", "### 2.1 Candidate metrics on the test window"),
    ("code", """\
for k, r in results.items():
    display(Markdown(f"**Model {k} - {r.target.replace('target_', '')}**"))
    display(r.metrics_frame().round(4))"""),

    ("md", "## 3. Charts and findings\n\nEvery finding below is backed by a house-style chart "
           "(`capstone.viz`); `make_charts.build_all` also writes the CSV exports."),
    ("code", """\
charts = make_charts.build_all(results)
_C = {k: p for k, p, _ in charts}
list(_C)"""),

    ("md", "### 3.1 Targets and the performance ceiling"),
    ("code", 'Image(str(_C["target_balance"]))'),
    ("code", 'Image(str(_C["feature_signal_recap"]))'),

    ("md", "### 3.2 Model A - no trained model separates the risk classes"),
    ("code", 'Image(str(_C["candidate_comparison_a"]))'),
    ("code", 'Image(str(_C["confusion_matrix_a"]))'),

    ("md", "### 3.3 Model B - gradient boosting for pre-submission triage"),
    ("code", 'Image(str(_C["candidate_comparison_b"]))'),
    ("code", 'Image(str(_C["confusion_matrix_b"]))'),
    ("code", 'Image(str(_C["costly_class_recall"]))'),

    ("md", "### 3.4 Probability calibration"),
    ("code", 'Image(str(_C["calibration_b"]))'),

    ("md", "## 4. Persist artefacts\n\nPipelines + calibrated wrappers + a portable "
           "`training_manifest.json` (relative paths, model version, per-model feature lists, "
           "every candidate's metrics)."),
    ("code", """\
window = {
    "start": splits.train["visit_date"].min().date().isoformat(),
    "end": splits.test["visit_date"].max().date().isoformat(),
    "rows": int(len(df)),
    "split": "9 months train / 1 validate / 2 test, calendar-anchored on visit_date",
}
provenance = {
    "source": "phase2_eda/output/feature_frame.parquet",
    "rows": int(len(df)),
    "modified": pd.Timestamp(PARQUET.stat().st_mtime, unit="s").isoformat(),
}
manifest_path = M.save_artifacts(results, MODELS, data_window=window,
                                 feature_frame_provenance=provenance)
print("wrote", manifest_path.relative_to(HERE))"""),

    ("md", "### 4.1 Reload parity - predict from a clean load"),
    ("code", """\
parity = M.reload_parity(MODELS, results)
display(parity)
assert parity["matches_in_memory_exactly"].all()"""),

    ("md", "## 5. Generate PHASE3_FINDINGS.md"),
    ("code", """\
import json
manifest = json.loads(manifest_path.read_text())
path = report.write_findings(results, charts, manifest, parity, FINDINGS_MD)
print("wrote", path.relative_to(HERE))"""),

    ("md", """\
## 6. Exit criteria

- [x] Time-based split on `visit_date` (9 / 1 / 2 months), no shuffle.
- [x] `ColumnTransformer` + model pipelines; logistic-regression and
      gradient-boosted-tree candidates, both with balanced class weights.
- [x] Probability calibration (sigmoid) on the validation month, both models.
- [x] Models + pipelines + `training_manifest.json` persisted with relative
      paths and a model version; artefacts reload and predict exactly.
- [x] Leakage register (`capstone.features.leakage_violations`) holds for both
      feature sets.
- [x] **Model B** beats the majority **and** simple-rule baselines on balanced
      accuracy, macro-F1 and rejected-claim recall.
- [ ] **Model A** cannot beat the baselines - no signal (Phase 2). Shipped as a
      calibrated base-rate monitor, documented in `PHASE3_FINDINGS.md`.
- [ ] Neither model beats the baseline **on raw accuracy** - the wrong bar for a
      skewed target; Model B trades accuracy for a two-thirds rejected-claim
      catch rate.

**Hand-off to Phase 4:** the persisted models + calibrated wrappers +
`training_manifest.json` + test predictions with calibrated probabilities.
Phase 4 does explainability, fairness, the leakage ablation, ROC/PR curves,
the operating-threshold choice and the model cards."""),
]


def build(path: str | Path | None = None) -> Path:
    path = Path(path) if path else (HERE / "phase3.ipynb")
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
