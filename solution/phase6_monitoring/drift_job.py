"""Phase 6 :: the scheduled drift job.

One monitoring pass: for each model, pull a window of ``prediction_log``,
compare it to the Phase 3 test-window reference, write the metrics to
``capstone_solution.drift_report``, and print a summary.

    uv run python -m phase6_monitoring drift-job --window drift --model both

Idempotent: every run is a fresh ``run_id`` time-series point; nothing is
updated or deleted. Exit code is 0 unless ``--fail-on-alert`` is set and the
run raised at least one alert (useful for cron / CI wiring).
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid

from capstone import monitoring as mon
from capstone import serving as S
from capstone.db import connect

log = logging.getLogger("capstone.phase6.drift_job")


def run(*, models: list[str], window: str, reference=None,
        serving_config=None) -> list[mon.DriftResult]:
    bundle = S.load_serving_bundle(serving_config_path=serving_config)
    ref = mon.reference_frame() if reference is None else reference
    run_id = str(uuid.uuid4())
    results: list[mon.DriftResult] = []
    with connect() as conn:
        mon.ensure_tables(conn)
        try:
            from capstone import eda
            spine = eda.load_spine()
        except Exception as exc:  # noqa: BLE001
            log.warning("spine unavailable - performance drift skipped this run: %s", exc)
            spine = None
        for model in models:
            res = mon.run_drift(conn, bundle, model=model, window=window,
                                reference=ref, spine=spine, run_id=run_id)
            n = mon.write_drift_report(conn, res)
            results.append(res)
            _print_summary(res, n)
    return results


def _print_summary(res: mon.DriftResult, n_rows: int) -> None:
    print(f"\n[{res.model}] window={res.window!r} n={res.window_n} "
          f"status={res.status}  ({n_rows} drift_report rows, run_id={res.run_id[:8]})")
    if not res.feature.empty:
        top = res.feature.head(5)
        for _, r in top.iterrows():
            print(f"    PSI {r['psi']:.3f} [{r['band']:<11}] {r['feature']}")
    if not res.prediction.empty:
        print(f"    prediction-mix PSI {res.prediction['mix_psi'].iloc[0]:.3f}")
    if res.gate.n:
        print(f"    validation gate: fail_rate {res.gate.fail_rate:.3%} "
              f"({'pass' if res.gate.passed else 'FAIL'})")
    if res.performance.get("model") == "B":
        p = res.performance
        print(f"    Model B recall on Rejected {p['recall_costly']:.2f} "
              f"(baseline {p['recall_costly_baseline']:.2f}, "
              f"delta {p['recall_costly_delta']:+.2f})")
    if not res.alerts.empty:
        print(f"    ALERTS: {len(res.alerts)}")
        for _, r in res.alerts.iterrows():
            feat = f" {r['feature']}" if r["feature"] else ""
            print(f"      - {r['metric_kind']}{feat}: {r['value']:.3f}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="phase6_monitoring drift-job")
    ap.add_argument("--model", choices=["A", "B", "both"], default="both")
    ap.add_argument("--window", default="drift",
                    help="baseline | drift | last-week | last-day | <ISO>..<ISO>")
    ap.add_argument("--serving-config", default=None)
    ap.add_argument("--fail-on-alert", action="store_true")
    args = ap.parse_args(argv)

    models = ["A", "B"] if args.model == "both" else [args.model]
    results = run(models=models, window=args.window, serving_config=args.serving_config)

    any_alert = any(not r.alerts.empty for r in results)
    print(f"\ndrift job complete - {'ALERTS RAISED' if any_alert else 'no alerts'}")
    return 1 if (any_alert and args.fail_on_alert) else 0


if __name__ == "__main__":
    sys.exit(main())
