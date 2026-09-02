"""Phase 3 entrypoint - regenerate and execute the Phase 3 notebook.

    uv run python phase3_models/run_phase3.py

The notebook (`phase3.ipynb`) is the deliverable; this script is the headless,
idempotent way to rebuild it and everything it emits (model artefacts, CSVs,
charts, `PHASE3_FINDINGS.md`). Steps:

  1. ensure the Phase 2 feature frame exists (rebuild Phase 2 if not)
  2. regenerate phase3.ipynb from `notebook.py`
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

import notebook as nb_builder  # noqa: E402

PARQUET = HERE.parent / "phase2_eda" / "output" / "feature_frame.parquet"


def ensure_feature_frame() -> None:
    if PARQUET.exists():
        print(f"[1] Phase 2 feature frame present ({PARQUET.relative_to(HERE.parents[1])})")
        return
    print("[1] Phase 2 feature frame missing; rebuilding via run_phase2.py")
    subprocess.run(
        [sys.executable, str(HERE.parents[0] / "phase2_eda" / "run_phase2.py")],
        check=True,
    )


def main() -> None:
    ensure_feature_frame()

    print("[2] regenerate phase3.ipynb from notebook.py")
    nb_path = nb_builder.build(HERE / "phase3.ipynb")

    print("[3] execute the notebook (rebuilds models/, output/, PHASE3_FINDINGS.md)")
    nb = nbformat.read(str(nb_path), as_version=4)
    NotebookClient(
        nb, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    ).execute()
    nbformat.write(nb, str(nb_path))

    print(f"\nDone.\n  notebook : {nb_path}\n  findings : {HERE / 'PHASE3_FINDINGS.md'}")


if __name__ == "__main__":
    main()
