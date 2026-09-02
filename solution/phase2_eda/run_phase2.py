"""Phase 2 entrypoint - regenerate and execute the Phase 2 notebook.

    uv run python phase2_eda/run_phase2.py

The notebook (`phase2.ipynb`) is the deliverable; this script is the headless,
idempotent way to rebuild it and everything it emits (CSVs, charts,
`feature_spec.yaml`, `PHASE2_FINDINGS.md`). Steps:

  1. ensure the Phase 1 analytics layer is reachable (rebuild it if not)
  2. regenerate phase2.ipynb from `notebook.py`
  3. execute it top-to-bottom (nbclient), in place
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

from capstone.db import SETTINGS, engine  # noqa: E402
import notebook as nb_builder  # noqa: E402


def ensure_analytics_layer() -> None:
    try:
        eng = engine()
        import pandas as pd
        pd.read_sql("SELECT 1 FROM v_visit_billing LIMIT 1", eng)
        eng.dispose()
        print(f"[1] analytics layer reachable (db={SETTINGS.database}, schema={SETTINGS.schema})")
    except Exception as exc:  # noqa: BLE001
        print(f"[1] analytics layer unreachable ({exc}); rebuilding via run_phase1")
        subprocess.run(
            [sys.executable, str(HERE.parents[0] / "phase1_sql_analytics" / "run_phase1.py")],
            check=True,
        )


def main() -> None:
    ensure_analytics_layer()

    print("[2] regenerate phase2.ipynb from notebook.py")
    nb_path = nb_builder.build(HERE / "phase2.ipynb")

    print("[3] execute the notebook (this rebuilds output/, feature_spec.yaml, PHASE2_FINDINGS.md)")
    nb = nbformat.read(str(nb_path), as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, str(nb_path))

    print(f"\nDone.\n  notebook : {nb_path}\n  findings : {HERE / 'PHASE2_FINDINGS.md'}")


if __name__ == "__main__":
    main()
