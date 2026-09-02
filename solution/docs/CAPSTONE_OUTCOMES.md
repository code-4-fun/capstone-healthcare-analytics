# Capstone — Key Outcomes & How to Approach It

Verbatim intent from the assignment brief (`objectives/Capstone-Statement.pdf`,
final page). This is the acceptance bar for the whole platform. Every phase is
judged against it, and **hawk-eye must check the phase's work against this file
before signalling OKAY for commit & push.**

## Key Learning Outcomes

By completing this capstone, the solution must demonstrate the ability to:

1. **Design end-to-end analytics and AI solutions for healthcare operations** —
   each phase produces an artefact the next phase consumes; the whole is one
   coherent platform, not disconnected assignments.
2. **Build and evaluate classification models with business impact in mind** —
   Model A (visit risk) and Model B (claim outcome) are judged on business
   metrics first: recall on High-risk visits and on Rejected claims.
3. **Detect and mitigate data quality and leakage risks** — missingness,
   distribution, outlier and timeline problems are quantified and handled; the
   leakage register is enforced (no post-outcome field in Model A or the
   pre-submission Model B; splits key on `visit_date`).
4. **Translate predictive outputs into operational and financial decisions** —
   every finding and every model output is expressed in money and risk terms
   (leakage recoverable, alert volume, staffing/bed implications), not just
   accuracy.
5. **Deploy, monitor, and govern AI systems using modern MLOps practices** —
   FastAPI services with schema validation and versioned prediction logging;
   drift detection; audit logs; documented assumptions, limitations and
   retraining strategy.
6. **Understand how analytics, machine learning, and deployment coexist within a
   single hospital intelligence platform** — the SQL layer, the models and the
   API/monitoring share one data foundation and one set of feature definitions.

## How to Approach This Capstone

- Treat it as **a simulation of professional healthcare analytics work**, not a
  collection of disconnected assignments.
- Every phase must be **clearly documented, technically justified, and
  interpreted in both business and clinical context** to show real-world
  applicability.
- The final deliverable is **a cohesive Hospital Operations & Revenue Risk
  Intelligence Platform** that integrates analytics, ML and deployment
  workflows and could realistically be proposed to hospital leadership as a
  deployable solution.

## Executive Business Presentation — what leadership must see at the end

The final phase must present:

- Hospital business problems and operational risks
- End-to-end system architecture and data flow
- Key insights from SQL and EDA
- Model performance **in business terms**
- Financial impact and revenue-optimization potential
- Deployment, scaling, and risk-management strategy

Phases 1–6 should be built so this story assembles itself — each phase leaves
behind the business-framed finding, chart, or metric the deck will need.
