# Agent instructions — Capstone solution

Project: **Hospital Operations & Revenue Risk Intelligence Platform** (see
`docs/PLAN.md` for the full phased plan and `README.md` for status).

The git repo root is `capstone-healthcare-analytics/`; **all solution work lives
in `capstone-healthcare-analytics/solution/` — treat that as the root folder for
the solution.** `objectives/` (the brief and raw data) is a sibling of
`solution/`, referenced as `../objectives/data`.

## Environment

- **Package manager is `uv`.** Add deps with `uv add`, run code with
  `uv run python ...`. Never call `pip` or edit `pyproject.toml` deps by hand.
- Postgres runs in Docker (container `capstone-project-postgres`, port 5432,
  trust auth). All work goes in database `capstone_hospital_analytics`,
  schema `capstone_solution`. Connection config is in `.env`; read it through
  `capstone.db.SETTINGS` / `connect()` / `engine()` — do not hard-code
  credentials.
- Schema/identifier names use `snake_case` (no hyphens — hyphenated identifiers
  need quoting everywhere in Postgres).

## Reporting standard — every finding is backed by a chart

This is a C-suite deliverable. **Any quantitative finding stated in a report,
notebook, or summary must be backed by an appropriate chart.** A number in prose
without a visual is not an acceptable finding.

Rules for every chart, in every phase:

1. **Use the shared house style** — `from capstone import viz`, call
   `viz.apply_house_style()`, build with `viz.new_figure()` and
   `viz.finalize(fig, ax, title=..., subtitle=..., source=..., out_path=...)`.
   Do not restyle per chart; extend `capstone/viz.py` if something is missing.
2. **Pick the form from the data's job** before colour — magnitude → bar;
   trend → line; distinct series → categorical grouped/stacked; one series is
   the point → emphasis (highlight + gray); part-to-whole → stacked bar;
   above/below a baseline → diverging. A single headline number is a stat tile,
   not a one-bar chart.
3. **Colour comes from `viz`** — `viz.CATEGORICAL` assigned in fixed slot order
   (never cycle past 8, never recolour on filter), `viz.SEQUENTIAL_BLUE` for
   magnitude, `viz.STATUS` / `viz.CLAIM_STATUS_COLORS` / `viz.SEVERITY_COLORS`
   **only** when colour means state. The palette is CVD-validated; do not
   introduce new hues without re-validating
   (dataviz skill `scripts/validate_palette.js`).
4. **One y-axis, ever.** Two measures of different scale → two charts or index
   to a common base. No dual-axis.
5. **Title states the takeaway** ("Rejections peak in the mid-value band"), not
   the mechanic ("Rejection rate by band"). Add a one-line subtitle and a
   `Source:` footer.
6. **Direct-label the marks** (the yellow status fill is below 3:1 contrast, so
   labels are mandatory, not optional); legend whenever ≥2 series; recessive
   hairline grid; no top/right spines.
7. **Ship the table too** — every chart's underlying numbers appear in a CSV
   export and/or a markdown table appendix, so the finding is verifiable and
   accessible.
8. **Look at the rendered PNG** before calling it done — check for label
   collisions, clipped text, legend overlap, axis overflow.

Per-phase pattern: a `make_charts.py` with one function per finding returning
`(key, path, caption)` and a `build_all()`; the phase's `run_*.py` calls it and
embeds the images in `PHASE<n>_FINDINGS.md`. Phase 1 is the reference
implementation (`phase1_sql_analytics/make_charts.py`).

## Reproducibility

- Each phase has one entrypoint (`phase<n>_*/run_*.py`) that rebuilds
  everything from scratch and is safe to re-run.
- Outputs (`output/`, `*_FINDINGS.md`) are generated, never hand-edited.
- Models and feature pipelines carry a version recorded in a manifest and
  echoed in API responses / the prediction log.

## Leakage discipline

`visit_date` is the temporal key for all splits and as-of features. No
post-outcome field (`approved_amount`, `payment_days`, `claim_status`) may feed
Model A or the pre-submission Model B. The Phase 2 leakage register is the
contract; Phase 3 enforces it; Phase 4 verifies by ablation.
