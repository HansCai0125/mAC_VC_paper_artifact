"""PyCharm: run/resume the formal 3-model x 5-seed x 2-dataset VC matrix."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from region_eval.vc_matrix import run_formal_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COCO_ROOT = Path(r"E:\Project\Database\MS-COCO")
VISDRONE_ROOT = Path(r"E:\Project\Database\VisDrone")
MATRIX_OUTPUT = PROJECT_ROOT / "results" / "detection"
COMPLETED_SEED0 = PROJECT_ROOT / "results" / "full_seed0_v1"

MODELS = ("yolov8n", "yolo11n", "yolo11s")
SEEDS = (0, 1, 2, 3, 4)
DATASETS = ("mscoco", "visdrone")

DEVICE = "auto"
IMGSZ = 640
CONFIDENCE_FLOOR = 0.001
NMS_IOU = 0.7
MAX_DET = 1000
EXPECTED_MATRIX_FINGERPRINT = "916baa3f128416d7f82a8bd680511a11e8586944669af951b5811aa67deb5141"


def checkpoint(dataset, model, seed):
    directory = "yolo8n" if model == "yolov8n" else model
    run_name = f"yolov8n_s{seed}" if model == "yolov8n" else f"{model}_s{seed}"
    if dataset == "mscoco":
        return COCO_ROOT / directory / run_name / "weights" / "best.pt"
    if dataset == "visdrone":
        return (
            PROJECT_ROOT / "training_logs" / "VisDrone" / directory / run_name
            / "weights" / "best.pt")
    raise ValueError(f"unknown dataset: {dataset}")


def model_id(model, seed):
    display = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo11s": "YOLO11s"}
    return f"{display[model]}_seed{seed}"


def specifications():
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for dataset in DATASETS:
                reuse = model == "yolo11n" and seed == 0
                output = (
                    COMPLETED_SEED0 / dataset if reuse else
                    MATRIX_OUTPUT / "runs" / dataset / model / f"seed_{seed}"
                )
                rows.append(dict(
                    kind=dataset,
                    dataset_root=(COCO_ROOT if dataset == "mscoco" else VISDRONE_ROOT),
                    weights=checkpoint(dataset, model, seed),
                    sample_count=(5000 if dataset == "mscoco" else 548),
                    model=model,
                    seed=seed,
                    model_id=model_id(model, seed),
                    output=output,
                    reuse_existing=reuse,
                ))
    return rows


if __name__ == "__main__":
    run_formal_matrix(
        specifications(),
        inference=dict(
            device=DEVICE,
            imgsz=IMGSZ,
            confidence_floor=CONFIDENCE_FLOOR,
            nms_iou=NMS_IOU,
            max_det=MAX_DET,
        ),
        matrix_output=MATRIX_OUTPUT,
        verify_checkpoints=True,
        expected_matrix_fingerprint=EXPECTED_MATRIX_FINGERPRINT,
    )
