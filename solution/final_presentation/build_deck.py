"""Build the executive PowerPoint deck for the Hospital Ops & Revenue Risk capstone.

Regenerate after `site/index.html` changes so the deck and the site stay in sync
(see final_presentation/README.md for the throwaway-venv build command).
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "site" / "assets" / "charts"
# Written inside site/ so it deploys with the page and is downloadable from it.
OUT = HERE / "site" / "Hospital_Ops_Revenue_Risk_Platform.pptx"

# ---- palette (matches the site: navy/blue ink, orange/green/amber/purple/red accents) ----
INK = RGBColor(0x0B, 0x0B, 0x0B)
SUB = RGBColor(0x5A, 0x5A, 0x57)
BG = RGBColor(0xFC, 0xFB, 0xF8)
CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
GREEN = RGBColor(0x1B, 0xAF, 0x7A)
AMBER = RGBColor(0xC0, 0x7F, 0x00)
PURPLE = RGBColor(0x4A, 0x3A, 0xA7)
RED = RGBColor(0xE3, 0x49, 0x48)
GRAY_LINE = RGBColor(0xD8, 0xD7, 0xD2)
BAD = RGBColor(0xC0, 0x30, 0x2E)
GOOD = RGBColor(0x1B, 0xAF, 0x7A)
NEUTRAL = RGBColor(0x2A, 0x78, 0xD6)

PHASE_COLORS = [BLUE, ORANGE, GREEN, AMBER, PURPLE, RED]

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

SW, SH = prs.slide_width, prs.slide_height


def add_slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    s.shapes._spTree.remove(bg._element)
    s.shapes._spTree.insert(2, bg._element)
    return s


def textbox(slide, l, t, w, h, text, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
            font="Calibri", italic=False, line_spacing=1.0, anchor=None):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    if anchor:
        tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
    return tb


def kicker(slide, text, color=ORANGE, top=Inches(0.45)):
    textbox(slide, Inches(0.6), top, Inches(8), Inches(0.35), text.upper(), size=13,
            color=color, bold=True)


def title(slide, text, top=Inches(0.78), size=30, width=Inches(12.1)):
    textbox(slide, Inches(0.6), top, width, Inches(1.0), text, size=size, color=INK, bold=True)


def rect(slide, l, t, w, h, fill=CARD_BG, line_color=None, line_w=Pt(1), radius=True, shadow=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape_type, l, t, w, h)
    if radius:
        try:
            sh.adjustments[0] = 0.045
        except Exception:
            pass
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line_color:
        sh.line.color.rgb = line_color
        sh.line.width = line_w
    else:
        sh.line.fill.background()
    sh.shadow.inherit = shadow
    return sh


def page_number(slide, n):
    textbox(slide, SW - Inches(0.9), SH - Inches(0.45), Inches(0.6), Inches(0.3), str(n),
            size=11, color=SUB, align=PP_ALIGN.RIGHT)


def footer_brand(slide):
    textbox(slide, Inches(0.6), SH - Inches(0.45), Inches(6), Inches(0.3),
            "Hospital Operations & Revenue Risk Intelligence Platform", size=10, color=SUB)


# =====================================================================
# SLIDE 1 — Title
# =====================================================================
s = add_slide()
band = rect(s, 0, 0, SW, Inches(0.14), fill=BLUE, radius=False)
textbox(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(0.4),
        "CAPSTONE · EXECUTIVE PRESENTATION", size=14, color=ORANGE, bold=True)
textbox(s, Inches(0.9), Inches(2.15), Inches(11.5), Inches(2.0),
        "A trusted data foundation, two decision\nmodels, and a governed platform for\none hospital network.",
        size=38, color=INK, bold=True, line_spacing=1.05)
textbox(s, Inches(0.9), Inches(4.55), Inches(10.6), Inches(1.3),
        "Hospital Operations & Revenue Risk Intelligence Platform — turning one calendar year of "
        "visit and claims data across a multi-specialty, multi-city network into a queryable analytics "
        "layer, a priced pre-submission claim-outcome model, a served API, and a drift-monitored, "
        "governed deployment.",
        size=15, color=SUB, line_spacing=1.25)
textbox(s, Inches(0.9), Inches(6.55), Inches(8), Inches(0.5),
        "Advanced Certificate Programme in Applied AI & Machine Learning  ·  Healthcare Business Capstone",
        size=12, color=SUB, italic=True)
textbox(s, Inches(0.9), Inches(6.95), Inches(8), Inches(0.4),
        "github.com/code-4-fun/capstone-healthcare-analytics", size=11, color=BLUE)

# =====================================================================
# SLIDE 2 — Agenda
# =====================================================================
s = add_slide()
kicker(s, "Where we're headed")
title(s, "Agenda")
agenda_items = [
    ("01", "Hospital business problem & operational risks", BLUE),
    ("02", "End-to-end system architecture & data flow", ORANGE),
    ("03", "Key insights from SQL analytics & EDA", GREEN),
    ("04", "Model performance, in business terms", AMBER),
    ("05", "Financial impact & revenue optimization potential", PURPLE),
    ("06", "Deployment, scaling & governance strategy", RED),
]
top = Inches(1.9)
for i, (num, txt, col) in enumerate(agenda_items):
    row_t = top + Inches(0.78) * i
    rect(s, Inches(0.6), row_t, Inches(0.62), Inches(0.58), fill=col, radius=True)
    textbox(s, Inches(0.6), row_t, Inches(0.62), Inches(0.58), num, size=18, color=RGBColor(255, 255, 255),
            bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, Inches(1.45), row_t + Inches(0.06), Inches(10.5), Inches(0.5), txt, size=18, color=INK,
            anchor=MSO_ANCHOR.MIDDLE)
page_number(s, 2)
footer_brand(s)

# =====================================================================
# SLIDE 3 — Business problem (three coupled failures)
# =====================================================================
s = add_slide()
kicker(s, "The business problem")
title(s, "Three coupled failures, one connected fix.")
textbox(s, Inches(0.6), Inches(1.55), Inches(11.8), Inches(0.5),
        "Measured directly from the network's own 2025 data — 5,000 patients, 25,000 visits, "
        "25,000 claims across departments, providers and insurers.", size=14, color=SUB)

cards = [
    ("01", "Patient-flow blindness",
     "No forward view of where demand, acuity and length-of-stay pressure will land, so "
     "staffing and bed planning stay reactive instead of planned.", BLUE),
    ("02", "Revenue leakage",
     "Insurance claims are rejected or under-approved after the fact — 17.6% of billed value "
     "lost to denials, another 24.5% sitting in pending limbo.", ORANGE),
    ("03", "No predictive layer",
     "Decisions are made on lagging reports, not on risk-scored, forward-looking signals a "
     "claims team can act on before submission.", GREEN),
]
card_w = Inches(3.85)
gap = Inches(0.2)
left0 = Inches(0.6)
card_t = Inches(2.35)
card_h = Inches(3.6)
for i, (num, h4, body, col) in enumerate(cards):
    l = left0 + (card_w + gap) * i
    rect(s, l, card_t, card_w, card_h, fill=CARD_BG, line_color=GRAY_LINE, line_w=Pt(1))
    top_bar = rect(s, l, card_t, card_w, Inches(0.08), fill=col, radius=False)
    textbox(s, l + Inches(0.3), card_t + Inches(0.3), card_w - Inches(0.6), Inches(0.6), num,
            size=26, color=col, bold=True)
    textbox(s, l + Inches(0.3), card_t + Inches(1.0), card_w - Inches(0.6), Inches(0.6), h4,
            size=18, color=INK, bold=True)
    textbox(s, l + Inches(0.3), card_t + Inches(1.6), card_w - Inches(0.6), Inches(1.8), body,
            size=13, color=SUB, line_spacing=1.2)
page_number(s, 3)
footer_brand(s)

# =====================================================================
# SLIDE 4 — Headline stats
# =====================================================================
s = add_slide()
kicker(s, "The stakes, in numbers")
title(s, "What's on the table for hospital leadership.")
stats = [
    ("17.6%", "of billed revenue lost to denials (₹91.8M)", BAD),
    ("57.9%", "revenue realization, billed → collected", NEUTRAL),
    ("62%", "recall on Rejected claims at the deployed operating threshold", GOOD),
    ("₹15L/mo", "recoverable denial leakage at the chosen threshold", NEUTRAL),
]
card_w = Inches(2.95)
gap = Inches(0.15)
left0 = Inches(0.6)
card_t = Inches(2.4)
card_h = Inches(2.6)
for i, (n, l_txt, col) in enumerate(stats):
    l = left0 + (card_w + gap) * i
    rect(s, l, card_t, card_w, card_h, fill=CARD_BG, line_color=GRAY_LINE, line_w=Pt(1))
    textbox(s, l + Inches(0.2), card_t + Inches(0.35), card_w - Inches(0.4), Inches(0.9), n,
            size=34, color=col, bold=True, align=PP_ALIGN.CENTER)
    textbox(s, l + Inches(0.25), card_t + Inches(1.35), card_w - Inches(0.5), Inches(1.1), l_txt,
            size=13, color=SUB, align=PP_ALIGN.CENTER, line_spacing=1.2)
textbox(s, Inches(0.6), Inches(5.4), Inches(11.8), Inches(1.4),
        "These are not abstract analytics metrics — they translate directly into a monthly claims-review "
        "workload, a cash-flow forecast, and a board-level revenue-leakage number.",
        size=15, color=SUB, italic=True, line_spacing=1.3)
page_number(s, 4)
footer_brand(s)

# =====================================================================
# SLIDE 5 — Architecture
# =====================================================================
s = add_slide()
kicker(s, "How the platform fits together")
title(s, "Each phase produces the artefact the next one consumes.", size=27)
textbox(s, Inches(0.6), Inches(1.5), Inches(11.8), Inches(0.5),
        "One pipeline, six phases, no hand-offs outside the repo — the same Postgres schema, feature "
        "catalogue and leakage register run from raw CSV to a served, monitored prediction.",
        size=13, color=SUB)

phases = [
    ("PHASE 1", "SQL Analytics", "Postgres · 10 BI views", BLUE),
    ("PHASE 2", "EDA & Data Quality", "Feature & leakage register", ORANGE),
    ("PHASE 3", "Modelling", "Model A + Model B", GREEN),
    ("PHASE 4", "Evaluation", "Threshold · fairness · cards", AMBER),
    ("PHASE 5", "Deployment", "FastAPI · prediction log", PURPLE),
    ("PHASE 6", "Monitoring", "Drift · governance · audit", RED),
]
box_w = Inches(1.83)
box_h = Inches(1.55)
box_gap = Inches(0.15)
box_t = Inches(2.35)
left0 = Inches(0.6)
centers = []
for i, (ph, name, sub, col) in enumerate(phases):
    l = left0 + (box_w + box_gap) * i
    rect(s, l, box_t, box_w, box_h, fill=CARD_BG, line_color=col, line_w=Pt(1.5))
    textbox(s, l, box_t + Inches(0.15), box_w, Inches(0.3), ph, size=11, color=col, bold=True,
            align=PP_ALIGN.CENTER)
    textbox(s, l + Inches(0.08), box_t + Inches(0.5), box_w - Inches(0.16), Inches(0.4), name,
            size=13, color=INK, bold=True, align=PP_ALIGN.CENTER)
    textbox(s, l + Inches(0.08), box_t + Inches(0.95), box_w - Inches(0.16), Inches(0.55), sub,
            size=9.5, color=SUB, align=PP_ALIGN.CENTER, line_spacing=1.05)
    centers.append((l, l + box_w))
    if i > 0:
        prev_r = centers[i - 1][1]
        arrow_l = prev_r
        conn = s.shapes.add_connector(2, arrow_l, box_t + box_h / 2, l, box_t + box_h / 2)
        conn.line.color.rgb = RGBColor(0x8A, 0x89, 0x85)
        conn.line.width = Pt(1.5)

textbox(s, Inches(0.6), Inches(4.25), Inches(11.8), Inches(0.6),
        "Retraining triggers (drift, quarterly cadence) feed back into Phase 2 → 3 → 4",
        size=11, color=SUB, italic=True, align=PP_ALIGN.CENTER)

textbox(s, Inches(0.6), Inches(5.1), Inches(11.8), Inches(1.6),
        "Raw CSVs → typed Postgres tables → curated feature frame → two calibrated classifiers → "
        "an evaluated, priced decision → a served, logged API → a monitored, governed production loop.",
        size=14, color=INK, italic=True, line_spacing=1.3)
page_number(s, 5)
footer_brand(s)

# =====================================================================
# Phase detail slides (6 slides), each with chart image
# =====================================================================
phase_details = [
    dict(num="1", name="SQL Analytics Layer", color=BLUE,
         tag="Postgres · typed schema · 10 BI views · data-quality report",
         body="Three raw CSVs become a trusted, indexed capstone_solution schema — PKs, FKs and "
              "CHECK constraints enforced in-database, ten business-intelligence views, and a "
              "12-check automated data-quality report. This is the spine every later phase reads from.",
         metric="521.8M billed → 302.3M collected — 57.9% realization; rejections peak at 22.7% "
                "in the 15k–30k billed band, non-monotonic in amount.",
         img=f"{CHARTS}/p1_revenue_waterfall.png",
         cap="Revenue waterfall: billed → approved → collected, with pending and denial leakage broken out."),
    dict(num="2", name="EDA & Data Quality", color=ORANGE,
         tag="Profiling · leakage register · feature catalogue",
         body="Turns the analytics layer into modelling readiness: every candidate feature is profiled "
              "for signal against the permuted-target noise floor, four data-quality findings get a "
              "written handling policy, and the leakage register that governs Phase 3 is written down "
              "field by field.",
         metric="Model A has no learnable signal — every feature sits below the noise floor. Model B "
                "is a billed-amount model, and the mid billed-band carries 70% of the 91.8M denial leakage.",
         img=f"{CHARTS}/p2_denial_leakage_concentration.png",
         cap="Denial leakage is concentrated in the mid billed-amount band, not spread evenly across claim size."),
    dict(num="3", name="Model Development", color=GREEN,
         tag="Two time-validated classifiers · calibrated",
         body="Model A (visit risk) and Model B (pre-submission claim outcome) are both split on "
              "visit_date — 9 months train, 1 validate, 2 test, no shuffle — and calibrated on the "
              "validation month. Only the learned candidate that clears both the majority baseline "
              "and the domain simple-rule ships; otherwise the baseline itself ships as a monitor.",
         metric="Model B recall on Rejected: 66% (vs 0% majority baseline, 62% simple rule) on the "
                "held-out test window.",
         img=f"{CHARTS}/p3_costly_class_recall.png",
         cap="Recall on the costly class (Rejected) — the model versus the majority and simple-rule baselines."),
    dict(num="4", name="Evaluation & Explainability", color=AMBER,
         tag="Threshold, SHAP, leakage ablation, fairness, model cards",
         body="Turns Model B into an operating decision: the calibrated P(Rejected) threshold is "
              "chosen on validation to maximise net recoverable leakage, verified clean of leakage by "
              "ablation (injecting a forbidden field spikes accuracy to 0.96 — the shipped model sits "
              "at the clean row), and checked for fairness across gender, age band, city and insurer.",
         metric="Operating threshold P(Rejected) ≥ 0.19 → ~850 review alerts/month, ~₹15L/month "
                "recoverable denial leakage (~₹5L net of review cost). No protected attribute fails "
                "the four-fifths parity test.",
         img=f"{CHARTS}/p4_net_recovery_b.png",
         cap="Net recoverable leakage across candidate thresholds — the operating point maximises the business return."),
    dict(num="5", name="Deployment & API", color=PURPLE,
         tag="FastAPI · versioned artefacts · prediction log",
         body="Serves the persisted Phase 3 models behind two endpoints, strict Pydantic validation "
              "bound to the Phase 1 domain constraints, and a Postgres prediction log that becomes "
              "Phase 6's drift baseline. Every response echoes its model, feature-spec and threshold "
              "versions.",
         metric="Two endpoints in production shape — /predict/claim-outcome (review vs submit) and "
                "/predict/visit-risk (a base-rate monitor, not a per-visit decision), plus /model-info and /health.",
         img=f"{CHARTS}/p5_latency.png",
         cap="Served prediction latency — both endpoints benchmarked as part of every build."),
    dict(num="6", name="Monitoring, Drift & Governance", color=RED,
         tag="PSI/KS drift · append-only audit · Grafana · governance docs",
         body="Watches the prediction log a real deployment would generate: a request-validation gate, "
              "feature/prediction/performance drift against the Phase 3 training reference, an "
              "append-only drift_report and audit trail enforced by a database trigger, a Grafana "
              "dashboard, and the governance, retraining-policy and incident-runbook documents.",
         metric="Drift job proven end to end — baseline traffic stays OK; injected drift trips "
                "feature-PSI alerts on billed amount, department and age, plus a ~34-point drop in "
                "Model B's recall on Rejected.",
         img=f"{CHARTS}/p6_feature_psi_drift.png",
         cap="Feature PSI on the drifted traffic window — billed amount, department and age cross the significant-drift line."),
]

for idx, pd in enumerate(phase_details, start=6):
    s = add_slide()
    col = pd["color"]
    kicker(s, "The platform, phase by phase", color=col)
    # badge + name
    rect(s, Inches(0.6), Inches(0.78), Inches(0.55), Inches(0.55), fill=col, radius=True)
    textbox(s, Inches(0.6), Inches(0.78), Inches(0.55), Inches(0.55), pd["num"], size=18,
            color=RGBColor(255, 255, 255), bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, Inches(1.3), Inches(0.78), Inches(9), Inches(0.6), pd["name"], size=27, color=INK, bold=True)
    textbox(s, Inches(0.6), Inches(1.45), Inches(11.9), Inches(0.35), pd["tag"], size=12, color=col, bold=True)

    left_w = Inches(6.5)
    textbox(s, Inches(0.6), Inches(1.95), left_w, Inches(2.0), pd["body"], size=13.5, color=SUB, line_spacing=1.25)

    metric_t = Inches(4.15)
    rect(s, Inches(0.6), metric_t, left_w, Inches(1.55), fill=RGBColor(0xF3, 0xF1, 0xEC), line_color=None, radius=True)
    textbox(s, Inches(0.85), metric_t + Inches(0.15), left_w - Inches(0.5), Inches(1.3), pd["metric"],
            size=13, color=INK, bold=True, line_spacing=1.2)

    links_t = Inches(5.9)
    textbox(s, Inches(0.6), links_t, left_w, Inches(0.4),
            f"solution/phase{pd['num']}_*  ·  {pd['num']}_FINDINGS.md", size=11, color=BLUE, italic=True)

    # image on right
    img_l = Inches(7.35)
    img_t = Inches(1.95)
    img_w = Inches(5.35)
    try:
        pic = s.shapes.add_picture(pd["img"], img_l, img_t, width=img_w)
        # cap height if too tall
        max_h = Inches(4.6)
        if pic.height > max_h:
            ratio = max_h / pic.height
            pic.height = max_h
            pic.width = Emu(int(pic.width * ratio))
            pic.left = img_l + (img_w - pic.width) // 2
    except Exception as e:
        rect(s, img_l, img_t, img_w, Inches(4.0), fill=RGBColor(0xEE, 0xEE, 0xEE))
    textbox(s, img_l, Inches(6.65), img_w, Inches(0.6), pd["cap"], size=10.5, color=SUB, italic=True,
            line_spacing=1.15, align=PP_ALIGN.LEFT)

    page_number(s, idx)
    footer_brand(s)

# =====================================================================
# Business impact slide
# =====================================================================
s = add_slide()
kicker(s, "What this means for leadership")
title(s, "Business impact, in money and risk terms.")

impact_items = [
    ("$", "~₹15L/month in recoverable denial leakage",
     "at the chosen operating threshold, against a total ₹91.8M annual denial pool — roughly "
     "~₹5L/month net of the review team's added workload.", GREEN),
    ("◔", "~850 claims/month routed to review",
     "a manageable pre-submission triage queue rather than a blanket policy change, sized "
     "directly from the validation-month sweep.", BLUE),
    ("✓", "Denial drivers are not the obvious ones",
     "rejection rate is non-monotonic in billed amount and nearly flat across department, "
     "provider and risk band — the mid billed-band is the lever, not any one team.", ORANGE),
    ("△", "Visit-risk scoring is not yet a usable signal",
     "Model A sits at the class-prior ceiling on this year's data; shipped as a risk-mix "
     "monitor, not a staffing decision — an honest ceiling, not a hidden one.", AMBER),
]
top = Inches(1.85)
for i, (ic, h, body, col) in enumerate(impact_items):
    row_t = top + Inches(1.3) * i
    rect(s, Inches(0.6), row_t, Inches(0.7), Inches(0.7), fill=col, radius=True)
    textbox(s, Inches(0.6), row_t, Inches(0.7), Inches(0.7), ic, size=22, color=RGBColor(255, 255, 255),
            bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, Inches(1.5), row_t - Inches(0.02), Inches(10.8), Inches(0.4), h, size=16, color=INK, bold=True)
    textbox(s, Inches(1.5), row_t + Inches(0.38), Inches(10.8), Inches(0.8), body, size=12.5, color=SUB,
            line_spacing=1.2)
page_number(s, 12)
footer_brand(s)

# =====================================================================
# Governance slide
# =====================================================================
s = add_slide()
kicker(s, "Responsible deployment")
title(s, "Governed the way a healthcare decision system should be.")

gov_items = [
    ("Leakage discipline, enforced in code",
     "visit_date is the only temporal key; no post-outcome field can reach either model, "
     "checked by leakage_violations() before every fit and proven by ablation at evaluation time.", BLUE),
    ("Fairness checked at the operating threshold",
     "selection rate, recall, FPR and calibration gap measured per gender, age band, city and "
     "insurer against a four-fifths parity bar; gaps are monitored, not silently accepted.", GREEN),
    ("Append-only audit trail",
     "prediction_log, drift_report and prediction_override reject UPDATE/DELETE at the database "
     "level — every prediction and every manual override is permanent.", ORANGE),
    ("Drift-triggered retraining policy",
     "written triggers (sustained PSI > 0.25, recall drop > 10pts, quarterly cadence), a "
     "shadow-then-promote procedure, and a one-config rollback via serving_config.json.", RED),
]
card_w = Inches(5.75)
card_h = Inches(2.15)
gap = Inches(0.3)
left0 = Inches(0.6)
top0 = Inches(1.85)
for i, (h, body, col) in enumerate(gov_items):
    row = i // 2
    col_i = i % 2
    l = left0 + (card_w + gap) * col_i
    t = top0 + (card_h + gap) * row
    rect(s, l, t, card_w, card_h, fill=CARD_BG, line_color=GRAY_LINE, line_w=Pt(1))
    rect(s, l, t, Inches(0.08), card_h, fill=col, radius=False)
    textbox(s, l + Inches(0.35), t + Inches(0.2), card_w - Inches(0.6), Inches(0.5), h, size=15,
            color=INK, bold=True)
    textbox(s, l + Inches(0.35), t + Inches(0.75), card_w - Inches(0.6), Inches(1.3), body, size=12,
            color=SUB, line_spacing=1.2)
page_number(s, 13)
footer_brand(s)

# =====================================================================
# Build status slide
# =====================================================================
s = add_slide()
kicker(s, "Build status")
title(s, "All six phases built, verified, and reproducible from a clean checkout.", size=25)

rows = [
    ("1", "SQL analytics layer (Postgres)", "Built"),
    ("2", "EDA & data quality", "Built"),
    ("3", "Model development (classification)", "Built"),
    ("4", "Evaluation & explainability", "Built"),
    ("5", "Deployment & API (FastAPI)", "Built"),
    ("6", "Monitoring, drift & governance", "Built"),
    ("—", "Executive presentation (site + deck)", "Built"),
]
table_l, table_t = Inches(0.6), Inches(1.85)
table_w, row_h = Inches(11.9), Inches(0.55)
header_h = Inches(0.5)
rect(s, table_l, table_t, table_w, header_h, fill=RGBColor(0x2A, 0x2A, 0x28), radius=False)
textbox(s, table_l + Inches(0.2), table_t + Inches(0.06), Inches(1), header_h, "Phase", size=13,
        color=RGBColor(255, 255, 255), bold=True, anchor=MSO_ANCHOR.MIDDLE)
textbox(s, table_l + Inches(1.5), table_t + Inches(0.06), Inches(8), header_h, "Scope", size=13,
        color=RGBColor(255, 255, 255), bold=True, anchor=MSO_ANCHOR.MIDDLE)
textbox(s, table_l + Inches(10.2), table_t + Inches(0.06), Inches(1.5), header_h, "State", size=13,
        color=RGBColor(255, 255, 255), bold=True, anchor=MSO_ANCHOR.MIDDLE)

for i, (ph, scope, state) in enumerate(rows):
    rt = table_t + header_h + row_h * i
    fill = CARD_BG if i % 2 == 0 else RGBColor(0xF3, 0xF1, 0xEC)
    rect(s, table_l, rt, table_w, row_h, fill=fill, radius=False)
    textbox(s, table_l + Inches(0.2), rt, Inches(1), row_h, ph, size=13, color=INK, bold=True,
            anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, table_l + Inches(1.5), rt, Inches(8), row_h, scope, size=13, color=INK,
            anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, table_l + Inches(10.2), rt, Inches(1.5), row_h, "✓ " + state, size=13, color=GOOD, bold=True,
            anchor=MSO_ANCHOR.MIDDLE)

foot_t = table_t + header_h + row_h * len(rows) + Inches(0.35)
textbox(s, Inches(0.6), foot_t, Inches(11.8), Inches(0.6),
        "One command reproduces the whole stack from a clean checkout — docker compose up -d, "
        "then each phase's run_phase<n>.py.", size=13, color=SUB, italic=True)
page_number(s, 14)
footer_brand(s)

# =====================================================================
# Closing slide
# =====================================================================
s = add_slide()
band = rect(s, 0, 0, SW, Inches(0.14), fill=BLUE, radius=False)
textbox(s, Inches(0.9), Inches(2.3), Inches(11), Inches(1.2),
        "One cohesive, deployable platform —\nready to propose to hospital leadership.",
        size=32, color=INK, bold=True, line_spacing=1.1)
textbox(s, Inches(0.9), Inches(4.0), Inches(10.5), Inches(1.3),
        "Analytics, machine learning, and MLOps deployment working as a single system: a trusted "
        "SQL foundation, calibrated and fair classifiers, a served API, and a governed monitoring "
        "loop — all reproducible from a clean checkout.",
        size=15, color=SUB, line_spacing=1.3)
textbox(s, Inches(0.9), Inches(5.7), Inches(10), Inches(0.4),
        "github.com/code-4-fun/capstone-healthcare-analytics", size=13, color=BLUE, bold=True)
textbox(s, Inches(0.9), Inches(6.15), Inches(10), Inches(0.4),
        "Thank you.", size=16, color=INK, italic=True)

prs.save(OUT)
print("Saved:", OUT)
