"""Builds ``phase4.ipynb`` - the Phase 4 review artefact.

The notebook is the deliverable (per `CLAUDE.md`: notebooks are the phase
artefact from Phase 2 on). It stays thin - it imports the reusable logic from
``capstone.evaluation`` and the phase-local ``make_charts`` / ``report`` helpers,
then orchestrates, explains and displays inline. ``run_phase4.py`` regenerates
and executes it.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

from capstone.evaluation import RECOVERY_RATE, REVIEW_COST

HERE = Path(__file__).resolve().parent
_HAIRCUT = int(RECOVERY_RATE * 100)
_REVIEW = f"{int(REVIEW_COST):,}"

CELLS: list[tuple[str, str]] = [
    ("md", """\
# Phase 4 - Model Evaluation & Explainability

*Hospital Operations & Revenue Risk Intelligence Platform*

**Goal:** prove the two Phase 3 classifiers are interpretable, reliable and safe
to deploy - and turn Model B into an operating decision.

This notebook evaluates the **persisted Phase 3 models as-is** (nothing is
retrained or retuned; the leakage ablation trains throwaway variants for
evidence only). It is thin: all reusable logic lives in `capstone.evaluation`;
the per-finding charts are in `phase4_eval/make_charts.py` and the report /
model-card assembler in `phase4_eval/report.py`. It runs top-to-bottom from a
clean kernel and is safe to re-run.

**Inputs:** `phase3_models/models/` (pipelines, calibrated wrappers,
`training_manifest.json`), `phase2_eda/output/feature_frame.parquet`,
`capstone_solution.v_visit_billing` (for reporting-only money fields).
**Outputs:** `phase4_eval/output/` (CSVs, `charts/`),
`phase4_eval/PHASE4_FINDINGS.md`, `phase4_eval/model_card_A.md`,
`phase4_eval/model_card_B.md`."""),

    ("code", """\
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from capstone import evaluation as E, modeling as M, features as feat, viz
import make_charts, report

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 170)
viz.apply_house_style()

HERE = Path.cwd()
OUT = HERE / "output"; (OUT / "charts").mkdir(parents=True, exist_ok=True)
PARQUET = HERE.parent / "phase2_eda" / "output" / "feature_frame.parquet"
FINDINGS_MD = HERE / "PHASE4_FINDINGS.md\""""),

    ("md", "## 1. Load the Phase 2 frame, the spine, and the persisted Phase 3 models\n\n"
           "Same calendar-anchored `visit_date` split as Phase 3 (9 months train / 1 validate / "
           "2 test). The spine is joined only for the reporting-only money fields "
           "(`billed_amount` / `approved_amount` / `leakage_amount` / `payment_days`) - never "
           "model inputs."),
    ("code", """\
from capstone import eda

df = pd.read_parquet(PARQUET)
df["visit_date"] = pd.to_datetime(df["visit_date"])
splits = M.time_split(df)
spine = eda.load_spine()
art = E.load_phase3()
manifest = art["manifest"]
print("model_version", manifest["model_version"], "| splits:")
splits.describe()"""),

    ("md", "### 1.1 Leakage self-check + reload parity\n\nThe Phase 2 register still holds, and the "
           "reloaded artefacts still reproduce their predictions - the Phase 3 exit criteria, "
           "re-verified here before we build anything on them."),
    ("code", """\
violations = feat.leakage_violations(splits.train)
assert not violations, violations
print("leakage register: clean")

# reload each artefact a second time from disk and confirm identical predictions
rows = []
for k in ("A", "B"):
    Xk = splits.test[sum(M.feature_lists(splits.train, k), [])]
    p2 = M.load_model(art["models_dir"], k, calibrated=False)
    c2 = M.load_model(art["models_dir"], k, calibrated=True)
    rows.append({"model": f"Model {k}", "n_test_rows": len(Xk),
                 "labels_match": bool((p2.predict(Xk) == art[k]["pipeline"].predict(Xk)).all()),
                 "proba_match": bool(np.allclose(c2.predict_proba(Xk),
                                                 art[k]["calibrated"].predict_proba(Xk)))})
parity = pd.DataFrame(rows)
display(parity)
assert parity[["labels_match", "proba_match"]].all().all()"""),

    ("md", "## 2. Prediction frames and per-class metrics\n\nOne row per test-window visit: the "
           "uncalibrated pipeline label (the Phase 3 default), the calibrated class probabilities, "
           "the protected-attribute columns, and the money fields."),
    ("code", """\
preds = {k: E.prediction_frame(splits, k, art, spine) for k in ("A", "B")}
per_class = {k: E.per_class_metrics(preds[k]["actual"], preds[k]["pred_pipeline"],
                                    M.CLASS_ORDER[k]) for k in ("A", "B")}
for k in ("A", "B"):
    display(Markdown(f"**Model {k} - per-class metrics (test window)**"))
    display(per_class[k])"""),

    ("code", """\
roc_pr = {}
for k in ("A", "B"):
    p = preds[k]
    classes = M.CLASS_ORDER[k]
    proba = p[[f"prob_{c.lower()}" for c in classes]].to_numpy()
    roc_pr[k] = E.roc_pr_frame(p["actual"], proba, classes, classes)
rp = (roc_pr["B"].groupby(["cls", "curve"]).agg(auc=("auc", "first"), ap=("ap", "first")).reset_index())
display(Markdown("**Model B one-vs-rest ROC AUC / average precision**"))
display(rp.round(4))"""),

    ("md", "## 3. Operating threshold for Model B - maximise net recovered leakage\n\n"
           "Sweep the calibrated `P(Rejected)` cut-off on the **validation** month; pick the "
           "threshold that maximises recoverable denial leakage on correctly-flagged rejections "
           f"(with a {_HAIRCUT}% haircut) minus review cost "
           f"(Rs. {_REVIEW}/claim). Apply it unchanged to the **test** window."),
    ("code", """\
val_split = M.Splits(splits.train, splits.val, splits.val)
pred_val = E.prediction_frame(val_split, "B", art, spine)
sweep_val = E.threshold_sweep(pred_val)
sweep_test = E.threshold_sweep(preds["B"])
chosen = E.choose_threshold(sweep_val)
threshold = float(chosen["threshold"])
business = E.business_summary(preds["B"], threshold)
print(f"operating threshold (chosen on validation): {threshold:.3f}")
display(sweep_val.iloc[::4].round(4))
display(Markdown("**Business impact at the operating threshold (test window)**"))
display(pd.Series(business).to_frame("value").round(2))"""),

    ("md", "## 4. Leakage verification by ablation\n\nRetrain Model B's Phase 3 learner (`gbm`) "
           "with features dropped, or with a forbidden post-outcome field injected, and score on "
           "the test window. `clean (shipped)` should reproduce Phase 3; every `LEAK +` row should "
           "spike. Model A is not ablated - it ships as a majority-class baseline that ignores "
           "every feature, so it cannot leak (its zero permutation importance in section 5 is the "
           "direct evidence)."),
    ("code", """\
ablation = {"B": E.leakage_ablation(splits, spine)}
display(Markdown("**Model B - leakage ablation**"))
display(ablation["B"])"""),

    ("md", "## 5. Explainability - permutation importance + SHAP (Model B)"),
    ("code", """\
feat_b = sum(M.feature_lists(splits.train, "B"), [])
perm_importance = {
    "B": E.permutation_importance_frame(art["B"]["pipeline"], splits.test[feat_b],
                                        preds["B"]["actual"]),
    "A": E.permutation_importance_frame(art["A"]["pipeline"],
                                        splits.test[sum(M.feature_lists(splits.train, "A"), [])],
                                        preds["A"]["actual"]),
}
display(Markdown("**Model B - permutation importance (top 12)**"))
display(perm_importance["B"].head(12))
display(Markdown("**Model A - permutation importance (every feature ~0: no signal)**"))
display(perm_importance["A"].head(6))
display(Markdown("**Model A - one-vs-rest ROC AUC (~0.50: constant scores)**"))
display(roc_pr["A"].groupby("cls")["auc"].first().dropna().round(4).to_frame("roc_auc"))"""),
    ("code", """\
X_b = splits.test[feat_b]
shap_res = E.shap_summary(art["B"]["pipeline"], X_b.sample(min(600, len(X_b)), random_state=0))
shap_local = E.shap_local_rows(preds["B"], shap_res) if shap_res is not None else {}
if shap_res is not None:
    display(Markdown("**Model B - SHAP mean |value| for P(Rejected) (top 10)**"))
    display(shap_res.mean_abs().head(10))
else:
    print("SHAP unavailable this run - permutation importance carries the explainability weight "
          "(documented in the report).")"""),

    ("md", "## 6. Calibration and fairness"),
    ("code", """\
reliability = {
    k: E.reliability_frame(art[k]["pipeline"], art[k]["calibrated"],
                           splits.test[sum(M.feature_lists(splits.train, k), [])],
                           preds[k]["actual"], M.CLASS_ORDER[k])
    for k in ("A", "B")
}
fairness = {"B": E.fairness_all(preds["B"], threshold=threshold)}
display(Markdown("**Model B - fairness parity summary (at the operating threshold)**"))
display(fairness["B"]["summary"].round(3))"""),

    ("md", "## 7. Charts, findings and model cards\n\nEvery finding is backed by a house-style "
           "chart (`capstone.viz`); `make_charts.build_all` also writes the CSV exports."),
    ("code", """\
ctx = E.EvalContext(
    manifest=manifest, review_cost=E.REVIEW_COST,
    preds=preds, per_class=per_class, roc_pr=roc_pr,
    sweep_val=sweep_val, sweep_test=sweep_test, threshold=threshold, business=business,
    ablation=ablation, perm_importance=perm_importance, reliability=reliability,
    fairness=fairness, shap=shap_res, shap_local=shap_local,
)
charts = make_charts.build_all(ctx)
_C = {k: p for k, p, _ in charts}
list(_C)"""),

    ("md", "### 7.1 Technical metrics"),
    ("code", 'Image(str(_C["per_class_b"]))'),
    ("code", 'Image(str(_C["confusion_operating_b"]))'),
    ("code", 'Image(str(_C["roc_b"]))'),
    ("code", 'Image(str(_C["pr_b"]))'),
    ("code", 'Image(str(_C["model_a_degeneracy"]))'),
    ("md", "### 7.2 Calibration"),
    ("code", 'Image(str(_C["calibration_ab"]))'),
    ("md", "### 7.3 Operating threshold & business impact"),
    ("code", 'Image(str(_C["threshold_sweep_b"]))'),
    ("code", 'Image(str(_C["net_recovery_b"]))'),
    ("md", "### 7.4 Explainability"),
    ("code", 'Image(str(_C["permutation_importance_b"]))'),
    ("code", 'display(Image(str(_C["shap_summary_b"]))) if "shap_summary_b" in _C else '
             'print("SHAP summary chart skipped (SHAP unavailable this run).")'),
    ("code", 'display(Image(str(_C["shap_local_b"]))) if "shap_local_b" in _C else None'),
    ("md", "### 7.5 Leakage ablation"),
    ("code", 'Image(str(_C["leakage_ablation"]))'),
    ("md", "### 7.6 Fairness"),
    ("code", 'Image(str(_C["fairness_b"]))'),

    ("md", "## 8. Generate PHASE4_FINDINGS.md and the model cards"),
    ("code", """\
p1 = report.write_findings(ctx, charts, FINDINGS_MD)
p2 = report.write_model_card(ctx, "A")
p3 = report.write_model_card(ctx, "B")
for p in (p1, p2, p3):
    print("wrote", p.relative_to(HERE))"""),

    ("md", """\
## 9. Exit criteria

- [x] Technical metrics: per-class precision / recall / F1, confusion matrices,
      ROC & PR curves, calibration plots (sections 1-2).
- [x] Business metrics: recall on Rejected claims, projected leakage recovered at
      the chosen operating threshold, alert volume (section 3).
- [x] **Model B** signed off at **>= 60% Rejected-claim recall** at a staffable
      review volume; achieved ~66% at the Phase 3 argmax and ~60% at the
      net-recovery operating threshold.
- [ ] **Model A** High-risk recall target waived - no signal on this data
      (Phase 2/3); retained as a monitor, documented in `model_card_A.md`.
- [x] Explainability: permutation importance (both models) + SHAP global & local
      for Model B (or a documented permutation-only fallback).
- [x] Leakage verified by ablation - test metrics move only when a post-outcome
      field is injected (section 4).
- [x] Fairness parity quantified across gender, age band, city and insurance
      provider; disparities and mitigations documented (section 6).
- [x] Model cards for A and B complete - seven sections each.

**Hand-off to Phase 5:** the persisted models + calibrated wrappers + the
**operating threshold** on `P(Rejected)` + the two model cards. Phase 5 serves
Model B, rate-limits the review queue, and logs predictions with model +
threshold version for Phase 6 monitoring."""),
]


def build(path: str | Path | None = None) -> Path:
    path = Path(path) if path else (HERE / "phase4.ipynb")
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
