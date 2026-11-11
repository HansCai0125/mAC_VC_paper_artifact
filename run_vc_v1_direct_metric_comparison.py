"""PyCharm entry point for the second-round direct metric comparison."""

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.direct_metric_comparison import (
    preflight_direct_metric_comparison,
    run_direct_metric_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "outputs" / "vc_v1" / "direct_metric_comparison_v1"

# Keep False for the normal PyCharm run. Set True only to validate inputs.
PREFLIGHT_ONLY = False


if __name__ == "__main__":
    if PREFLIGHT_ONLY:
        result = preflight_direct_metric_comparison(output=OUTPUT)
    else:
        result = {
            "status": "complete",
            "output": str(run_direct_metric_comparison(output=OUTPUT)),
            "training": False,
            "inference": False,
        }
    print(json.dumps(result, ensure_ascii=False))

