"""PyCharm: run/resume RT-DETR-L on complete COCO and VisDrone val sets."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.vc_rtdetr_run import run_rtdetr_suite


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "rtdetr_full_v1"
OUTPUT = PROJECT_ROOT / "outputs" / "vc_v1" / RUN_NAME

DEVICE = "auto"
IMGSZ = 640
CONFIDENCE_FLOOR = 0.001
MAX_DET = 1000

COCO_ROOT = Path(r"E:\Project\Database\MS-COCO")
VISDRONE_ROOT = Path(r"E:\Project\Database\VisDrone")

CONFIGURATIONS = [
    dict(
        kind="mscoco",
        dataset_root=COCO_ROOT,
        weights=(
            PROJECT_ROOT / "anti_gaming" / "Best" / "MSCOCO"
            / "RTDETR" / "best.pt"
        ),
        sample_count=5000,
        seed=0,
        model_id="RT-DETR-L_seed0",
    ),
    *[
        dict(
            kind="visdrone",
            dataset_root=VISDRONE_ROOT,
            weights=(
                PROJECT_ROOT / "plot" / "VisDrone" / "RTDETR-l"
                / f"rtdetr-l.pt_s{seed}" / "weights" / "best.pt"
            ),
            sample_count=548,
            seed=seed,
            model_id=f"RT-DETR-L_seed{seed}",
        )
        for seed in range(5)
    ],
]


if __name__ == "__main__":
    run_rtdetr_suite(
        CONFIGURATIONS,
        output=OUTPUT,
        device=DEVICE,
        imgsz=IMGSZ,
        confidence_floor=CONFIDENCE_FLOOR,
        max_det=MAX_DET,
    )
