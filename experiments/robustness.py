"""Entry point for ownership and F1-operating-point robustness.

Reads existing full-validation prediction caches and completed threshold CSVs.
No training, model loading, or inference is performed.  Results go to a new
directory and the expensive ownership pass resumes completed seeds safely.
"""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.reviewer_robustness import run_analysis


# Set True to verify all 35 caches without recalculating either analysis.
PREFLIGHT_ONLY = False
# Set True to finish just the fast F1-optimal-threshold analysis.
OPERATING_POINT_ONLY = False


if __name__ == "__main__":
    output = run_analysis(preflight_only=PREFLIGHT_ONLY,
                          operating_only=OPERATING_POINT_ONLY)
    print(f"SUCCESS: reviewer robustness outputs: {Path(output).resolve()}")
