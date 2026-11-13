"""Entry point for continuous anti-gaming mAP curves.

The normal run fills missing diagnostic AP values from existing prediction
caches.  It performs no training and no model inference.  Completed source
runs are resumable and are never written over the formal sparse-AP results.
"""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.gaming_full_ap import (
    preflight_full_ap,
    run_full_ap_supplement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = PROJECT_ROOT / "results" / "detection"
FORMAL_OUTPUT = PROJECT_ROOT / "results" / "gaming_main"
OUTPUT = PROJECT_ROOT / "results" / "perturbations"

# Set this to True only when you want a fast integrity check without AP work.
PREFLIGHT_ONLY = False


if __name__ == "__main__":
    if PREFLIGHT_ONLY:
        preflight_full_ap(
            matrix_root=MATRIX_ROOT,
            formal_output=FORMAL_OUTPUT,
        )
    else:
        run_full_ap_supplement(
            matrix_root=MATRIX_ROOT,
            formal_output=FORMAL_OUTPUT,
            output=OUTPUT,
        )

