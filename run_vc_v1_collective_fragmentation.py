"""PyCharm entry point for the collective-fragmentation experiment."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.collective_fragmentation import (
    preflight_collective_fragmentation,
    run_collective_fragmentation,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MATRIX_ROOT = PROJECT_ROOT / "outputs" / "vc_v1" / "formal_matrix_v1"
OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "vc_v1"
    / "collective_fragmentation_v1"
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
