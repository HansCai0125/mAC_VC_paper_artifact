"""PyCharm entry point for the collective-fragmentation experiment."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.collective_fragmentation import (
    preflight_collective_fragmentation,
    run_collective_fragmentation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = PROJECT_ROOT / "results" / "detection"
OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "fragmentation"
)

# False runs the full offline experiment. True performs a fast cache check.
PREFLIGHT_ONLY = False


if __name__ == "__main__":
    if PREFLIGHT_ONLY:
        preflight_collective_fragmentation(matrix_root=MATRIX_ROOT)
    else:
        run_collective_fragmentation(
            matrix_root=MATRIX_ROOT,
            output=OUTPUT,
        )
