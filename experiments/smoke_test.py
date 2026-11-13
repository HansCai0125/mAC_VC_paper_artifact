"""Smoke test: two datasets, existing weights, no training."""

import os as _os
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.vc_smoke import run_smoke_suite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_COUNT = 20
SUBSET_SEED = 0
DEVICE = "auto"  # CUDA device 0 when available; otherwise CPU.
IMGSZ = 640
CONFIDENCE_FLOOR = 0.001
NMS_IOU = 0.7
MAX_DET = 1000
COCO_ROOT = Path(_os.environ.get("MAC_COCO_ROOT", r"E:\Project\Database\MS-COCO"))
VISDRONE_ROOT = Path(_os.environ.get("MAC_VISDRONE_ROOT", r"E:\Project\Database\VisDrone"))

CONFIGURATIONS = [
    dict(
        kind="mscoco",
        dataset_root=COCO_ROOT,
        weights=COCO_ROOT / "yolo11n" / "yolo11n_s0" / "weights" / "best.pt",
        model_id="YOLO11n_seed0",
    ),
    dict(
        kind="visdrone",
        dataset_root=VISDRONE_ROOT,
        weights=PROJECT_ROOT / "training_logs" / "VisDrone" / "yolo11n" / "yolo11n_s0" / "weights" / "best.pt",
        model_id="YOLO11n_seed0",
    ),
]


if __name__ == "__main__":
    shared = dict(
        sample_count=SAMPLE_COUNT,
        subset_seed=SUBSET_SEED,
        device=DEVICE,
        imgsz=IMGSZ,
        confidence_floor=CONFIDENCE_FLOOR,
        nms_iou=NMS_IOU,
        max_det=MAX_DET,
    )
    run_smoke_suite([dict(configuration, **shared) for configuration in CONFIGURATIONS])
