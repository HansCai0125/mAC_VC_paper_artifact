"""Entry point for the VC incremental-value experiment."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.vc_incremental_value import (
    preflight_vc_incremental_value,
    run_vc_incremental_value,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = PROJECT_ROOT / "results" / "detection"
OUTPUT = PROJECT_ROOT / "results" / "vc_components"

# False runs the complete offline property suite. True only validates inputs.
PREFLIGHT_ONLY = False


if __name__ == "__main__":
    if PREFLIGHT_ONLY:
        result = preflight_vc_incremental_value(
            matrix_root=MATRIX_ROOT,
            output=OUTPUT,
        )
    else:
        result = {
            "status": "complete",
            "output": str(run_vc_incremental_value(
                matrix_root=MATRIX_ROOT,
                output=OUTPUT,
            )),
            "training": False,
            "inference": False,
        }
    print(json.dumps(result, ensure_ascii=False))

