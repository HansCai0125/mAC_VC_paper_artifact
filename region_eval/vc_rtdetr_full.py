"""Resumable full-validation VC evaluation for Ultralytics RT-DETR weights.

The historical YOLO runner is left unchanged. RT-DETR must be constructed
with :class:`ultralytics.RTDETR` so Ultralytics selects ``RTDETRPredictor``
and its native set-prediction post-processing (no NMS).
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter

from .case_analysis import _file_hash
from .offline import PROJECT_ROOT
from .vc_full import (
    FULL_RUN_VERSION,
    _completed_run,
    _dataset_source_signature,
    _read_json,
    _source_hashes,
    _validate_shard,
    load_existing_shards,
    shard_path,
)
from .vc_matrix import _atomic_json_with_retry as _atomic_json
from .vc_smoke import _model_names, _prepare, _yaml_names
from .vc_v1 import (
    PRIMARY_THETA,
    THETA_GRID,
    VC_VERSION,
    analyze_vc_document,
    write_vc_analysis,
)


RTDETR_RUN_VERSION = "vc-v1-rtdetr-full-0.1.0"
EXPECTED_COMBINATIONS = {
    ("mscoco", 0),
    *(("visdrone", seed) for seed in range(5)),
}


def _fingerprint(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _runner_source_hashes():
    hashes = _source_hashes()
    hashes["region_eval/vc_rtdetr_full.py"] = _file_hash(Path(__file__))
    run_module = PROJECT_ROOT / "region_eval" / "vc_rtdetr_run.py"
    if run_module.is_file():
        hashes["region_eval/vc_rtdetr_run.py"] = _file_hash(run_module)
    return hashes


def _configuration_record(
        *, kind, dataset_root, weights, sample_count, seed, model_id,
        imgsz, confidence_floor, max_det):
    dataset_root = Path(dataset_root).resolve()
    weights = Path(weights).resolve()
    record = dict(
        full_run_version=FULL_RUN_VERSION,
        rtdetr_run_version=RTDETR_RUN_VERSION,
        vc_version=VC_VERSION,
        model_family="rtdetr",
        kind=kind,
        dataset_root=str(dataset_root),
        weights=str(weights),
        weights_sha256=_file_hash(weights),
        sample_count=sample_count,
        subset_seed=0,
        training_seed=seed,
        model_id=model_id,
        inference=dict(
            backend="ultralytics.RTDETR",
            predictor="RTDETRPredictor",
            imgsz=imgsz,
            confidence_floor=confidence_floor,
            max_det=max_det,
            native_nms=False,
            nms_iou=None,
            batch=1,
            half=False,
            augment=False,
            rect=True,
        ),
        primary_theta=PRIMARY_THETA,
        theta_grid=list(THETA_GRID),
        dataset_source_signature=_dataset_source_signature(kind, dataset_root),
        source_sha256=_runner_source_hashes(),
    )
    record["run_fingerprint"] = _fingerprint(record)
    return record


def _manifest_metadata(prepared, configuration, *, device, torch, ultralytics):
    return dict(
        dataset=prepared["dataset"],
        split=f"val_full_{configuration['sample_count']}",
        model_id=configuration["model_id"],
        model_family="rtdetr",
        training_seed=configuration["training_seed"],
        checkpoint_id=(
            f"{Path(configuration['weights']).name}:sha256:"
            f"{configuration['weights_sha256']}"),
        coordinate_frame="original_image_xyxy",
        ignore_crowd_policy="already_filtered",
        class_ids=list(range(len(prepared["class_names"]))),
        class_names=prepared["class_names"],
        postprocessing=dict(
            stage="native_rtdetr_postprocess_no_nms",
            **configuration["inference"],
        ),
        preprocessing=prepared["preprocessing"],
        subset=dict(
            method="full validation set; deterministic filename order",
            seed=configuration["subset_seed"],
            available_count=len(prepared["available_names"]),
            count=len(prepared["items"]),
            available_names_sha256=sha256(
                json.dumps(prepared["available_names"]).encode()).hexdigest(),
        ),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            ultralytics=ultralytics.__version__,
            ultralytics_path=str(Path(ultralytics.__file__).parent),
            device=str(device),
        ),
        weights_file=configuration["weights"],
        weights_sha256=configuration["weights_sha256"],
        dataset_yaml_file=str(prepared["data_yaml"]),
        dataset_yaml_sha256=_file_hash(prepared["data_yaml"]),
        full_run_version=FULL_RUN_VERSION,
        rtdetr_run_version=RTDETR_RUN_VERSION,
        vc_version=VC_VERSION,
        run_fingerprint=configuration["run_fingerprint"],
        purpose=(
            "complete validation cache and VC v1 RT-DETR evaluation; "
            "existing checkpoint only; no training"),
    )


def _predict_rtdetr(
        model, path, *, width, height, device, imgsz, confidence_floor,
        max_det, output):
    """Predict with RTDETRPredictor and retain native post-processed boxes."""
    results = model.predict(
        source=str(path),
        imgsz=imgsz,
        conf=confidence_floor,
        max_det=max_det,
        device=device,
        batch=1,
        half=False,
        augment=False,
        rect=True,
        classes=None,
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show=False,
        verbose=False,
        stream=False,
        project=str(output),
        name="inference",
        exist_ok=True,
    )
    if len(results) != 1 or tuple(results[0].orig_shape) != (height, width):
        raise ValueError(f"unexpected RT-DETR image shape/count for {path.name}")
    if Path(results[0].path).resolve() != Path(path).resolve():
        raise ValueError("RT-DETR prediction source differs from selected image")
    boxes = results[0].boxes
    coordinates = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    if len(coordinates) > max_det or not all(
            float(value).is_integer() for value in classes):
        raise ValueError("invalid RT-DETR prediction count/class")
    return [
        dict(
            id=f"pred_{index:06d}", xyxy=rect, score=score,
            class_id=int(class_id))
        for index, (rect, score, class_id) in enumerate(
            zip(coordinates, scores, classes))
    ]


def _expected_names(kind, dataset_root):
    return _prepare(kind, Path(dataset_root), 1, 0)["class_names"]


def preflight_rtdetr_suite(
        configurations, *, imgsz=640, confidence_floor=0.001,
        max_det=1000, verify_checkpoints=True):
    """Validate the exact six-run plan without inference or training."""
    if confidence_floor > min(THETA_GRID):
        raise ValueError("confidence floor exceeds the theta-grid minimum")
    combinations = set()
    hashes = set()
    records = []
    for configuration in configurations:
        configuration = dict(configuration)
        key = (configuration["kind"], configuration["seed"])
        if key in combinations:
            raise ValueError(f"duplicate RT-DETR combination: {key}")
        combinations.add(key)
        record = _configuration_record(
            **configuration,
            imgsz=imgsz,
            confidence_floor=confidence_floor,
            max_det=max_det,
        )
        if record["weights_sha256"] in hashes:
            raise ValueError("RT-DETR checkpoints must have distinct SHA-256 values")
        hashes.add(record["weights_sha256"])
        records.append(record)
    if combinations != EXPECTED_COMBINATIONS:
        raise ValueError(
            "suite must contain MSCOCO seed 0 and VisDrone seeds 0--4")

    expected_counts = {"mscoco": 5000, "visdrone": 548}
    for record in records:
        found = record["dataset_source_signature"]["images"]["count"]
        expected = expected_counts[record["kind"]]
        if found != expected or record["sample_count"] != expected:
            raise ValueError(
                f"full validation count mismatch for {record['kind']}: "
                f"found={found}, configured={record['sample_count']}, "
                f"expected={expected}")

    if verify_checkpoints:
        import torch
        from ultralytics import RTDETR

        names_by_kind = {}
        for record in records:
            names_by_kind.setdefault(
                record["kind"],
                _expected_names(record["kind"], record["dataset_root"]),
            )
        for index, record in enumerate(records, 1):
            model = RTDETR(record["weights"])
            _model_names(model, names_by_kind[record["kind"]])
            predictor = model._smart_load("predictor")
            if predictor.__name__ != "RTDETRPredictor":
                raise ValueError(
                    f"wrong backend for {record['model_id']}: "
                    f"{predictor.__name__}")
            print(
                f"PREFLIGHT {index:02d}/{len(records)}: "
                f"{record['kind']}:{record['model_id']} "
                f"sha256={record['weights_sha256'][:12]}... "
                "RTDETRPredictor OK",
                flush=True,
            )
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    existing = PROJECT_ROOT / "results" / "full_seed0_v1"
    estimated = 0
    for kind, multiplier in (("mscoco", 1), ("visdrone", 5)):
        reference = existing / kind
        if reference.is_dir():
            reference_size = sum(
                path.stat().st_size for path in reference.rglob("*")
                if path.is_file())
            estimated += reference_size * multiplier
    free = shutil.disk_usage(PROJECT_ROOT).free
    if estimated and free < estimated * 1.2:
        raise OSError(
            f"insufficient disk space: free={free}, estimated={estimated}")
    summary = dict(
        status="passed",
        combinations=len(records),
        mscoco_images=5000,
        visdrone_images=548,
        distinct_checkpoint_hashes=len(hashes),
        predictor="RTDETRPredictor",
        native_nms=False,
        estimated_output_bytes=estimated,
        free_bytes=free,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("PREFLIGHT SUCCESS: no training or model inference performed.", flush=True)
    return records, summary
