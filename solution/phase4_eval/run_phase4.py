"""Phase 4 entrypoint - regenerate and execute the Phase 4 notebook.

    uv run python phase4_eval/run_phase4.py

The notebook (`phase4.ipynb`) is the deliverable; this script is the headless,
idempotent way to rebuild it and everything it emits (CSVs, charts,
`PHASE4_FINDINGS.md`, `model_card_A.md`, `model_card_B.md`). Steps:

  1. ensure the Phase 3 model artefacts exist (rebuild Phase 3 if not)
  2. regenerate phase4.ipynb from `notebook.py`
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

MANIFEST = HERE.parent / "phase3_models" / "models" / "training_manifest.json"


def ensure_phase3_artefacts() -> None:
    if MANIFEST.exists():
        print(f"[1] Phase 3 artefacts present ({MANIFEST.relative_to(HERE.parents[1])})")
        return
    print("[1] Phase 3 artefacts missing; rebuilding via run_phase3.py")
    subprocess.run(
        [sys.executable, str(HERE.parents[0] / "phase3_models" / "run_phase3.py")],
        check=True,
    )


def main() -> None:
    ensure_phase3_artefacts()

    print("[2] regenerate phase4.ipynb from notebook.py")
    nb_path = nb_builder.build(HERE / "phase4.ipynb")

    print("[3] execute the notebook (rebuilds output/, PHASE4_FINDINGS.md, model cards)")
    nb = nbformat.read(str(nb_path), as_version=4)
    NotebookClient(
        nb, timeout=1200, kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    ).execute()
    nbformat.write(nb, str(nb_path))

    print(f"\nDone.\n  notebook : {nb_path}\n  findings : {HERE / 'PHASE4_FINDINGS.md'}\n"
          f"  cards    : {HERE / 'model_card_A.md'}, {HERE / 'model_card_B.md'}")


if __name__ == "__main__":
    main()
