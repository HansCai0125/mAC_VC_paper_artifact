"""Five-seed, full-validation MS COCO evaluation for the three YOLO models."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from .case_analysis import _file_hash
from .offline import PROJECT_ROOT
from .vc_full import run_dataset_full
from .vc_matrix import _atomic_json_with_retry as _atomic_json
from .vc_v1 import VC_VERSION


YOLO_COCO5_SUITE_VERSION = "vc-v1-yolo-coco5-0.1.0"
MODELS = ("yolov8n", "yolo11n", "yolo11s")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fingerprint(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def configurations_from_training(training_root, coco_root):
    """Build 15 evaluation configurations from the audited training suite."""
    training_root = Path(training_root).resolve()
    complete_path = training_root / "suite_complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"YOLO five-seed training is incomplete: {complete_path}")
    complete = _read_json(complete_path)
    if complete.get("status") != "complete":
        raise ValueError("YOLO training suite is not complete")
    records = complete.get("results", [])
    expected = {(model, seed) for model in MODELS for seed in range(5)}
    found = {(row["model"], int(row["seed"])) for row in records}
    if found != expected:
        raise ValueError("YOLO training suite does not contain 3 models x 5 seeds")
    if len({row["parameter_sha256"] for row in records}) != 15:
        raise ValueError("YOLO checkpoints are not parameter-distinct")
    coco_root = Path(coco_root).resolve()
    return [dict(
        kind="mscoco",
        dataset_root=coco_root,
        weights=Path(record["checkpoint"]).resolve(),
        sample_count=5000,
        subset_seed=0,
        model=record["model"],
        seed=int(record["seed"]),
        model_id=(
            f"{'YOLOv8n' if record['model'] == 'yolov8n' else record['model'].upper()}"
            f"_seed{int(record['seed'])}"
        ),
    ) for record in sorted(
        records, key=lambda row: (MODELS.index(row["model"]), int(row["seed"])))]


def run_yolo_coco5_suite(
        configurations, *, output, device="auto", imgsz=640,
        confidence_floor=0.001, nms_iou=0.7, max_det=1000):
    """Run or resume 15 complete COCO validations and VC analyses."""
    expected = {(model, seed) for model in MODELS for seed in range(5)}
    found = {(row["model"], int(row["seed"])) for row in configurations}
    if found != expected:
        raise ValueError("evaluation suite must contain 3 models x 5 seeds")
    hashes = {_file_hash(Path(row["weights"])) for row in configurations}
    if len(hashes) != 15:
        raise ValueError("evaluation checkpoints must have distinct file hashes")

    runtime_dir = PROJECT_ROOT / "outputs" / "vc_v1" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    serialised = [{
        key: str(value) if isinstance(value, Path) else value
        for key, value in row.items()
    } for row in configurations]
    plan = dict(
        yolo_coco5_suite_version=YOLO_COCO5_SUITE_VERSION,
        vc_version=VC_VERSION,
        device=device,
        imgsz=imgsz,
        confidence_floor=confidence_floor,
        nms_iou=nms_iou,
        max_det=max_det,
        configurations=serialised,
        source_sha256={
            "region_eval/vc_yolo_coco5.py": _file_hash(Path(__file__)),
        },
    )
    plan["suite_fingerprint"] = _fingerprint(plan)
    manifest_path = output / "suite_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("suite_fingerprint") != plan["suite_fingerprint"]:
            raise ValueError("YOLO COCO5 evaluation changed; use a new output")
    else:
        if any(output.iterdir()):
            raise ValueError(f"nonempty output has no manifest: {output}")
        _atomic_json(manifest_path, dict(
            status="started",
            created_at=datetime.now(timezone.utc).isoformat(),
            **plan,
        ))

    results = []
    try:
        for index, configuration in enumerate(configurations, 1):
            print(
                f"YOLO COCO {index:02d}/15 START: "
                f"{configuration['model']} seed{configuration['seed']}",
                flush=True,
            )
            run_output = (
                output / "runs" / configuration["model"]
                / f"seed_{configuration['seed']}")
            arguments = {
                key: value for key, value in configuration.items()
                if key not in {"model", "seed"}
            }
            result = run_dataset_full(
                **arguments,
                output=run_output,
                device=device,
                imgsz=imgsz,
                confidence_floor=confidence_floor,
                nms_iou=nms_iou,
                max_det=max_det,
            )
            result = dict(result, model=configuration["model"],
                          seed=configuration["seed"])
            results.append(result)
            _atomic_json(output / "suite_status.json", dict(
                status="running",
                suite_fingerprint=plan["suite_fingerprint"],
                completed_runs=len(results),
                total_runs=15,
                last_completed=configuration["model_id"],
                updated_at=datetime.now(timezone.utc).isoformat(),
            ))
    except BaseException as exc:
        _atomic_json(output / "suite_status.json", dict(
            status="failed_or_interrupted",
            suite_fingerprint=plan["suite_fingerprint"],
            completed_runs=len(results),
            total_runs=15,
            error=repr(exc),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ))
        raise

    complete = dict(
        status="complete",
        yolo_coco5_suite_version=YOLO_COCO5_SUITE_VERSION,
        vc_version=VC_VERSION,
        suite_fingerprint=plan["suite_fingerprint"],
        completed_runs=15,
        total_runs=15,
        results=results,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_json(output / "suite_complete.json", complete)
    _atomic_json(output / "suite_status.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    print(f"SUCCESS: YOLO five-seed COCO evaluation complete: {output}")
    return output
