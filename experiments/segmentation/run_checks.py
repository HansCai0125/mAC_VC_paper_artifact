"""Synthetic correctness checks. No model inference or downloads."""
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT / "third_party"))
TEMP_DIRECTORY = ROOT / "validation/tmp"
TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(TEMP_DIRECTORY)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_runtime/matplotlib"))

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
