"""PyCharm entry point for the offline natural collective-gain analysis."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.natural_collective_gain import (
    preflight_natural_collective_gain,
    run_natural_collective_gain,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "results" / "collective_gain"

# Keep False for the normal PyCharm run.  Set True only to validate all 35
# source caches without loading the large prediction JSON files.
PREFLIGHT_ONLY = False


if __name__ == "__main__":
    if PREFLIGHT_ONLY:
        result = preflight_natural_collective_gain(output=OUTPUT)
    else:
        result = {
            "status": "complete",
            "output": str(run_natural_collective_gain(output=OUTPUT)),
            "training": False,
            "inference": False,
        }
    print(json.dumps(result, ensure_ascii=False))
