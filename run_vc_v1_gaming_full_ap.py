"""PyCharm entry point for continuous anti-gaming mAP curves.

The normal run fills missing diagnostic AP values from existing prediction
caches.  It performs no training and no model inference.  Completed source
runs are resumable and are never written over the formal sparse-AP results.
"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.gaming_full_ap import (
    preflight_full_ap,
    run_full_ap_supplement,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MATRIX_ROOT = PROJECT_ROOT / "outputs" / "vc_v1" / "formal_matrix_v1"
FORMAL_OUTPUT = PROJECT_ROOT / "outputs" / "vc_v1" / "gaming_formal_v1"
OUTPUT = PROJECT_ROOT / "outputs" / "vc_v1" / "gaming_full_ap_v1"

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
        # The aggregate now contains mAP at every severity, so this produces
        # the connected paper figures immediately after AP completion.
        from run_vc_v1_gaming_full_map_figures import main as write_figures

        write_figures()

