"""Five-seed, full-validation MS COCO evaluation for RT-DETR-L."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .case_analysis import _file_hash
from .offline import PROJECT_ROOT
from .vc_full import _read_json
from .vc_matrix import _atomic_json_with_retry as _atomic_json
from .vc_rtdetr_full import (
    RTDETR_RUN_VERSION,
    _configuration_record,
    _fingerprint,
    _expected_names,
)
from .vc_rtdetr_run import run_rtdetr_dataset_full
from .vc_smoke import _model_names
from .vc_v1 import VC_VERSION


COCO5_SUITE_VERSION = "vc-v1-rtdetr-coco5-0.1.0"


def _training_records(training_root):
    training_root = Path(training_root).resolve()
    complete_path = training_root / "suite_complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"five-seed RT-DETR training is incomplete: {complete_path}")
    complete = _read_json(complete_path)
    if complete.get("status") != "complete":
        raise ValueError("RT-DETR training suite is not complete")
    results = complete.get("results", [])
    if {int(row["seed"]) for row in results} != set(range(5)):
        raise ValueError("RT-DETR training suite does not contain seeds 0--4")
    if len({row["parameter_sha256"] for row in results}) != 5:
        raise ValueError("RT-DETR training checkpoints are not parameter-distinct")
    return complete_path, sorted(results, key=lambda row: int(row["seed"]))


def configurations_from_training(training_root, coco_root):
    """Build the five evaluation configurations from audited training output."""
    _, records = _training_records(training_root)
    return [
        dict(
            kind="mscoco",
            dataset_root=Path(coco_root).resolve(),
            weights=Path(record["checkpoint"]).resolve(),
            sample_count=5000,
            seed=int(record["seed"]),
            model_id=f"RT-DETR-L_seed{int(record['seed'])}",
        )
        for record in records
    ]


def preflight_rtdetr_coco5(
        configurations, *, imgsz=640, confidence_floor=0.001,
        max_det=1000, verify_checkpoints=True):
    """Validate five distinct RT-DETR checkpoints and the full COCO split."""
    if {(row["kind"], int(row["seed"])) for row in configurations} != {
            ("mscoco", seed) for seed in range(5)}:
        raise ValueError("COCO RT-DETR suite must contain seeds 0--4")
    records = [
        _configuration_record(
            **dict(configuration), imgsz=imgsz,
            confidence_floor=confidence_floor, max_det=max_det)
        for configuration in configurations
    ]
    if len({row["weights_sha256"] for row in records}) != 5:
        raise ValueError("RT-DETR checkpoints must have distinct SHA-256 values")
    for row in records:
        if row["sample_count"] != 5000:
            raise ValueError("COCO sample_count must be 5000")
        found = row["dataset_source_signature"]["images"]["count"]
        if found != 5000:
            raise ValueError(f"COCO validation set has {found} images, not 5000")

    if verify_checkpoints:
        import torch
        from ultralytics import RTDETR

        names = _expected_names("mscoco", records[0]["dataset_root"])
        for index, record in enumerate(records, 1):
            model = RTDETR(record["weights"])
            _model_names(model, names)
            predictor = model._smart_load("predictor")
            if predictor.__name__ != "RTDETRPredictor":
                raise ValueError(
                    f"unexpected predictor for {record['model_id']}: "
                    f"{predictor.__name__}")
            print(
                f"PREFLIGHT {index}/5: {record['model_id']} "
                f"sha256={record['weights_sha256'][:12]}... OK",
                flush=True,
            )
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


def run_rtdetr_coco5_suite(
        configurations, *, output, device="auto", imgsz=640,
        confidence_floor=0.001, max_det=1000,
        expected_suite_fingerprint=None):
    """Run or resume five complete COCO validations and VC analyses."""
    runtime_dir = PROJECT_ROOT / "results" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"

    records = preflight_rtdetr_coco5(
        configurations, imgsz=imgsz, confidence_floor=confidence_floor,
        max_det=max_det, verify_checkpoints=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        coco5_suite_version=COCO5_SUITE_VERSION,
        rtdetr_run_version=RTDETR_RUN_VERSION,
        vc_version=VC_VERSION,
        device=device,
        imgsz=imgsz,
        confidence_floor=confidence_floor,
        max_det=max_det,
        configurations=records,
        source_sha256={
            "region_eval/vc_rtdetr_coco5.py": _file_hash(Path(__file__)),
        },
    )
    plan["suite_fingerprint"] = _fingerprint(plan)
    if (expected_suite_fingerprint is not None
            and plan["suite_fingerprint"] != expected_suite_fingerprint):
        raise ValueError("COCO5 evaluation differs from its frozen fingerprint")
    manifest_path = output / "suite_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("suite_fingerprint") != plan["suite_fingerprint"]:
            raise ValueError("COCO5 evaluation changed; use a new output name")
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
                f"RT-DETR COCO {index}/5 START: {configuration['model_id']}",
                flush=True,
            )
            run_output = output / "runs" / f"seed_{configuration['seed']}"
            result = run_rtdetr_dataset_full(
                **configuration,
                output=run_output,
                device=device,
                imgsz=imgsz,
                confidence_floor=confidence_floor,
                max_det=max_det,
            )
            results.append(result)
            _atomic_json(output / "suite_status.json", dict(
                status="running",
                suite_fingerprint=plan["suite_fingerprint"],
                completed_runs=len(results),
                total_runs=5,
                last_completed=configuration["model_id"],
                updated_at=datetime.now(timezone.utc).isoformat(),
            ))
    except BaseException as exc:
        _atomic_json(output / "suite_status.json", dict(
            status="failed_or_interrupted",
            suite_fingerprint=plan["suite_fingerprint"],
            completed_runs=len(results),
            total_runs=5,
            error=repr(exc),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ))
        raise

    complete = dict(
        status="complete",
        coco5_suite_version=COCO5_SUITE_VERSION,
        rtdetr_run_version=RTDETR_RUN_VERSION,
        vc_version=VC_VERSION,
        suite_fingerprint=plan["suite_fingerprint"],
        completed_runs=5,
        total_runs=5,
        results=results,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_json(output / "suite_complete.json", complete)
    _atomic_json(output / "suite_status.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    print(f"SUCCESS: five-seed RT-DETR COCO evaluation complete: {output}")
    return output
