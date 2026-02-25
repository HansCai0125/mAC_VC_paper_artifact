from pathlib import Path
import os

# Project root = repo root
PROJECT_ROOT = Path(__file__).resolve().parent

# Allow overrides via environment variables
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data"))
RESULT_ROOT = Path(os.getenv("RESULT_ROOT", PROJECT_ROOT / "result"))
MAC_ROOT = Path(os.getenv("MAC_ROOT", PROJECT_ROOT / "mAC"))
