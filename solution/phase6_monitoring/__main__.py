"""``python -m phase6_monitoring <command>`` - the Phase 6 CLI.

Commands:
  * ``drift-job``  - one drift monitoring pass (see ``drift_job.main``)
  * ``seed``       - (re)seed the demo baseline + drift traffic windows
  * ``gate``       - run the validation gate over a window and print the report
  * ``scheduler``  - the sidecar loop (seed once, then drift job on an interval)
"""
from __future__ import annotations

import sys


def _seed(argv: list[str]) -> int:
    import argparse

    from capstone.db import connect
    from phase6_monitoring import seed_traffic

    ap = argparse.ArgumentParser(prog="phase6_monitoring seed")
    ap.add_argument("--n-baseline", type=int, default=900)
    ap.add_argument("--n-drift", type=int, default=1400)
    args = ap.parse_args(argv)
    with connect() as conn:
        from capstone import monitoring as mon

        mon.ensure_tables(conn)
        summary = seed_traffic.seed_all(conn, n_baseline=args.n_baseline, n_drift=args.n_drift)
    print(summary)
    return 0


def _gate(argv: list[str]) -> int:
    import argparse

    from capstone import monitoring as mon
    from capstone.db import connect

    ap = argparse.ArgumentParser(prog="phase6_monitoring gate")
    ap.add_argument("--model", choices=["A", "B"], default="B")
    ap.add_argument("--window", default="drift")
    args = ap.parse_args(argv)
    with connect() as conn:
        win = mon.load_window(conn, model=args.model, window=args.window)
        res = mon.validation_gate(mon._gate_payload_frame(win)) if len(win) else None
    if res is None:
        print("no rows in window")
        return 0
    print(f"window n={res.n}  fail_rate={res.fail_rate:.3%}  passed={res.passed}")
    if not res.offences.empty:
        print(res.offences.to_string(index=False))
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "drift-job":
        from phase6_monitoring import drift_job

        return drift_job.main(rest)
    if cmd == "seed":
        return _seed(rest)
    if cmd == "gate":
        return _gate(rest)
    if cmd == "scheduler":
        from phase6_monitoring import scheduler

        scheduler.main()
        return 0
    print(f"unknown command: {cmd}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
