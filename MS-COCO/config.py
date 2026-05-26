from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
MAC_ROOT    = Path(os.getenv("MAC_ROOT", PROJECT_ROOT / "mAC")).resolve()