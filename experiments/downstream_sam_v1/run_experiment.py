"""Run in PyCharm using the existing yolo interpreter. Resumes automatically."""
from pathlib import Path
import argparse
import os
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT / "third_party"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_runtime/matplotlib"))

# full = technical pilot on calibration images, then the held-out experiment.
MODE = "full"  # full, pilot, preflight, report
RUN_NAME = "formal_v1"
DEVICE = "auto"  # CUDA; CPU must be explicitly selected with "cpu".
AUTO_PREPARE = True  # Downloads missing public SAM weights ONLY; never installs packages.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "pilot", "preflight", "report"), default=MODE)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--no-prepare", action="store_true")
    args = parser.parse_args()
    from downstream.pipeline import execute
    from downstream.protocol import Settings
    execute(ROOT, Settings(device=args.device), mode=args.mode, run_name=args.run_name,
            auto_prepare=AUTO_PREPARE and not args.no_prepare)


if __name__ == "__main__":
    main()
