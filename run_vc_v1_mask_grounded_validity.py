"""PyCharm entry point for the third-round foreground-mask analysis."""

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.mask_grounded_validity import (
    preflight_mask_grounded_validity,
    run_mask_grounded_validity,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "outputs" / "vc_v1" / "mask_grounded_validity_v1"

# Keep False for the normal PyCharm run. Set True only to validate inputs.
PREFLIGHT_ONLY = False


if __name__ == "__main__":
    if PREFLIGHT_ONLY:
        result = preflight_mask_grounded_validity(output=OUTPUT)
    else:
        result = {
            "status": "complete",
            "output": str(run_mask_grounded_validity(output=OUTPUT)),
            "training": False,
            "inference": False,
        }
    print(json.dumps(result, ensure_ascii=False))
