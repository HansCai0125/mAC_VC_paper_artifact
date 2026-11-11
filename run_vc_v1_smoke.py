"""PyCharm: right-click -> Run. Two datasets, existing weights, no training."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.vc_smoke import run_smoke_suite


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_COUNT = 20
SUBSET_SEED = 0
DEVICE = "auto"  # CUDA device 0 when available; otherwise CPU.
IMGSZ = 640
CONFIDENCE_FLOOR = 0.001
NMS_IOU = 0.7
MAX_DET = 1000

CONFIGURATIONS = [
    dict(
        kind="mscoco",
        dataset_root=Path(r"E:\Project\Database\MS-COCO"),
        weights=Path(r"E:\Project\Database\MS-COCO\yolo11n\yolo11n_s0\weights\best.pt"),
        model_id="YOLO11n_seed0",
    ),
    dict(
        kind="visdrone",
        dataset_root=Path(r"E:\Project\Database\VisDrone"),
        weights=PROJECT_ROOT / "plot" / "VisDrone" / "yolo11n" / "yolo11n_s0" / "weights" / "best.pt",
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
