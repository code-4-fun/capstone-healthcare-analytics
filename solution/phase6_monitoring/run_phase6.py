"""Phase 6 entrypoint - regenerate and execute the Phase 6 notebook.

    uv run python phase6_monitoring/run_phase6.py

The notebook (`phase6.ipynb`) is the deliverable; this script is the headless,
idempotent way to rebuild it and everything it emits (`drift_report` rows,
`output/` CSVs + charts, `PHASE6_FINDINGS.md`). Steps:

  1. ensure the Phase 3 model artefacts + the Phase 5 serving_config exist
  2. require Postgres; apply the prediction_log + drift_report DDL
  3. regenerate phase6.ipynb from notebook.py and execute it top-to-bottom
     (it seeds the demo traffic, runs the drift job, writes the report)
  4. run the pytest suite
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTION = HERE.parent
sys.path.insert(0, str(SOLUTION / "src"))
sys.path.insert(0, str(HERE))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

import notebook as nb_builder  # noqa: E402

MANIFEST = SOLUTION / "phase3_models" / "models" / "training_manifest.json"
SERVING_CONFIG = SOLUTION / "phase5_api" / "serving_config.json"


def step(n: int, msg: str) -> None:
    print(f"[{n}] {msg}")


def ensure_upstream() -> None:
    if not MANIFEST.exists():
        step(1, "Phase 3 artefacts missing; rebuilding via run_phase3.py")
        subprocess.run([sys.executable, str(SOLUTION / "phase3_models" / "run_phase3.py")], check=True)
    if not SERVING_CONFIG.exists():
        step(1, "Phase 5 serving_config missing; rebuilding via run_phase5.py")
        subprocess.run([sys.executable, str(SOLUTION / "phase5_api" / "run_phase5.py")], check=True)
    step(1, "Phase 3 + Phase 5 artefacts present")


def apply_ddl() -> None:
    step(2, "apply prediction_log + drift_report DDL")
    from capstone import monitoring as mon
    from capstone.db import connect

    try:
        with connect() as conn:
            mon.ensure_tables(conn)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Postgres is required for Phase 6 and is unreachable ({exc}).\n"
            "Start it with:  docker compose up -d   (from solution/)"
        ) from exc
    print("    tables ready: prediction_log, drift_report, prediction_override, v_prediction_audit")


def run_notebook() -> Path:
    step(3, "regenerate + execute phase6.ipynb (seeds traffic, runs drift job, writes findings)")
    nb_path = nb_builder.build(HERE / "phase6.ipynb")
    nb = nbformat.read(str(nb_path), as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, str(nb_path))
    return nb_path


def run_tests() -> str:
    step(4, "run pytest suite")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(HERE / "tests"), "-p", "no:cacheprovider",
         "-q", "--no-header", "-o", "addopts="],
        cwd=SOLUTION, capture_output=True, text=True,
    )
    summary = ""
    for line in reversed(proc.stdout.strip().splitlines()):
        if re.search(r"\d+ (passed|failed|error)", line):
            summary = line.strip("= ").strip()
            break
    print("   ", summary or (proc.stdout.strip().splitlines() or ["(no output)"])[-1])
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", summary)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", summary)) else 0
    if failed or errors or (proc.returncode != 0 and not summary):
        print(proc.stdout[-4000:])
        raise SystemExit(f"pytest: {failed} failed, {errors} errors")
    return summary


def summarise() -> None:
    from capstone import monitoring as mon
    from capstone.db import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*), count(*) FILTER (WHERE alert), "
                    "count(DISTINCT run_id) FROM capstone_solution.drift_report")
        rows, alerts, runs = cur.fetchone()
    print(f"\nDone.")
    print(f"  notebook : {(HERE / 'phase6.ipynb').relative_to(SOLUTION)}")
    print(f"  findings : {(HERE / 'PHASE6_FINDINGS.md').relative_to(SOLUTION)}")
    print(f"  drift_report: {rows} rows, {alerts} alerts, {runs} runs")
    print(f"  charts   : {(HERE / 'output' / 'charts').relative_to(SOLUTION)}")


def main() -> None:
    ensure_upstream()
    apply_ddl()
    run_notebook()
    run_tests()
    summarise()


if __name__ == "__main__":
    main()
