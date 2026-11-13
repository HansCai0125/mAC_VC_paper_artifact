"""Resumable formal VC matrix runner and cross-seed aggregation."""

from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
import statistics
from time import sleep
from uuid import uuid4

from . import vc_full as _vc_full_module
from .case_analysis import _file_hash
from .coco import COCO_NAMES
from .offline import PROJECT_ROOT
from .vc_full import (
    FULL_RUN_VERSION,
    _atomic_json as _original_atomic_json,
    _completed_run,
    _dataset_source_signature,
    _read_json,
    _source_hashes,
    run_dataset_full,
)
from .vc_smoke import _model_names
from .vc_v1 import PRIMARY_THETA, THETA_GRID, VC_VERSION
from .visdrone import CLASS_NAMES as VISDRONE_NAMES


MATRIX_VERSION = "vc-v1-formal-matrix-0.1.0"
FORMAL_MODELS = ("yolov8n", "yolo11n", "yolo11s")
FORMAL_SEEDS = (0, 1, 2, 3, 4)
FORMAL_DATASETS = ("mscoco", "visdrone")
EXPECTED_COMBINATIONS = {
    (dataset, model, seed)
    for dataset in FORMAL_DATASETS
    for model in FORMAL_MODELS
    for seed in FORMAL_SEEDS
}

WINDOWS_ATOMIC_WRITE_ATTEMPTS = 20
WINDOWS_ATOMIC_WRITE_INITIAL_DELAY = 0.05
WINDOWS_ATOMIC_WRITE_MAX_DELAY = 2.0


def _retry_delay(failed_attempt):
    return min(
        WINDOWS_ATOMIC_WRITE_INITIAL_DELAY * (2 ** (failed_attempt - 1)),
        WINDOWS_ATOMIC_WRITE_MAX_DELAY,
    )


def _atomic_json_with_retry(
        path, value, attempts=WINDOWS_ATOMIC_WRITE_ATTEMPTS):
    """Retry atomic JSON writes when Windows briefly locks the target file."""
    for attempt in range(1, attempts + 1):
        try:
            return _original_atomic_json(path, value)
        except PermissionError:
            if attempt == attempts:
                raise
            if attempt == 1:
                print(
                    f"WINDOWS FILE LOCK: retrying atomic write: {Path(path)}",
                    flush=True,
                )
            sleep(_retry_delay(attempt))


def _replace_with_retry(
        temporary, path, attempts=WINDOWS_ATOMIC_WRITE_ATTEMPTS):
    """Retry os.replace while preserving the already-flushed temporary file."""
    for attempt in range(1, attempts + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            if attempt == 1:
                print(
                    f"WINDOWS FILE LOCK: retrying atomic replace: {Path(path)}",
                    flush=True,
                )
            sleep(_retry_delay(attempt))


_atomic_json = _atomic_json_with_retry


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _fingerprint(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _expected_names(kind):
    if kind == "mscoco":
        return COCO_NAMES
    if kind == "visdrone":
        return VISDRONE_NAMES
    raise ValueError(f"unknown matrix dataset: {kind}")


def _expected_inference(inference):
    return dict(
        imgsz=inference["imgsz"],
        confidence_floor=inference["confidence_floor"],
        nms_iou=inference["nms_iou"],
        max_det=inference["max_det"],
        class_agnostic_nms=False,
        batch=1,
        half=False,
        augment=False,
        rect=True,
    )


def _directory_size(path):
    return sum(
        candidate.stat().st_size
        for candidate in Path(path).rglob("*")
        if candidate.is_file()
    )


def _dataset_file_count(kind, dataset_root):
    dataset_root = Path(dataset_root)
    if kind == "mscoco":
        image_dir = dataset_root / "images" / "val"
        annotation = dataset_root / "annotations" / "instances_val2017.json"
        yaml_path = dataset_root / "data.yaml"
        if not annotation.is_file() or not yaml_path.is_file():
            raise FileNotFoundError(f"incomplete MSCOCO dataset: {dataset_root}")
        return sum(
            path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for path in image_dir.iterdir())
    if kind == "visdrone":
        split = dataset_root / "VisDrone2019-DET-val"
        image_dir = split / "images"
        annotation_dir = split / "annotations"
        yaml_path = dataset_root / "visdrone.yaml"
        if not annotation_dir.is_dir() or not yaml_path.is_file():
            raise FileNotFoundError(f"incomplete VisDrone dataset: {dataset_root}")
        image_names = {
            path.stem for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        annotation_names = {
            path.stem for path in annotation_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }
        if image_names != annotation_names:
            raise ValueError("VisDrone image and annotation names differ")
        return len(image_names)
    raise ValueError(f"unknown matrix dataset: {kind}")


def _validate_reused_run(record, inference):
    output = Path(record["output"])
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"reused run manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    expected = {
        "kind": record["kind"],
        "dataset_root": str(Path(record["dataset_root"]).resolve()),
        "weights": str(Path(record["weights"]).resolve()),
        "weights_sha256": record["weights_sha256"],
        "sample_count": record["sample_count"],
        "subset_seed": 0,
        "model_id": record["model_id"],
        "inference": _expected_inference(inference),
        "primary_theta": PRIMARY_THETA,
        "theta_grid": list(THETA_GRID),
        "dataset_source_signature": record["dataset_source_signature"],
        "source_sha256": record["source_sha256"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"reused run differs at {key}: {record['label']}")
    complete = _completed_run(output, manifest["run_fingerprint"])
    expected_dataset = "MSCOCO2017" if record["kind"] == "mscoco" else "VisDrone2019-DET"
    if complete.get("dataset") != expected_dataset:
        raise ValueError(f"reused run dataset mismatch: {record['label']}")
    return complete


def _normalise_records(specifications, inference):
    records = []
    combinations = set()
    output_paths = set()
    weight_hashes = set()
    required = {
        "kind", "dataset_root", "weights", "sample_count", "model", "seed",
        "model_id", "output", "reuse_existing",
    }
    for specification in specifications:
        missing = required - set(specification)
        if missing:
            raise ValueError(f"matrix specification is missing: {sorted(missing)}")
        record = dict(specification)
        record["dataset_root"] = Path(record["dataset_root"]).resolve()
        record["weights"] = Path(record["weights"]).resolve()
        record["output"] = Path(record["output"]).resolve()
        if not record["weights"].is_file() or record["weights"].suffix.lower() != ".pt":
            raise FileNotFoundError(f"checkpoint is missing: {record['weights']}")
        combination = (record["kind"], record["model"], record["seed"])
        if combination in combinations:
            raise ValueError(f"duplicate matrix combination: {combination}")
        combinations.add(combination)
        if record["output"] in output_paths:
            raise ValueError(f"duplicate matrix output: {record['output']}")
        output_paths.add(record["output"])
        record["weights_sha256"] = _file_hash(record["weights"])
        if record["weights_sha256"] in weight_hashes:
            raise ValueError("formal checkpoints must have distinct SHA-256 values")
        weight_hashes.add(record["weights_sha256"])
        record["label"] = (
            f"{record['kind']}:{record['model']}:seed{record['seed']}")
        records.append(record)
    if combinations != EXPECTED_COMBINATIONS:
        missing = sorted(EXPECTED_COMBINATIONS - combinations)
        extra = sorted(combinations - EXPECTED_COMBINATIONS)
        raise ValueError(f"formal matrix mismatch; missing={missing}, extra={extra}")
    if inference["confidence_floor"] > min(THETA_GRID):
        raise ValueError("confidence floor exceeds the theta-grid minimum")
    return records


def _plan_value(records, inference):
    serialised = []
    for record in records:
        serialised.append({
            key: str(value) if isinstance(value, Path) else value
            for key, value in record.items()
            if key != "label"
        })
    value = dict(
        matrix_version=MATRIX_VERSION,
        full_run_version=FULL_RUN_VERSION,
        vc_version=VC_VERSION,
        primary_theta=PRIMARY_THETA,
        theta_grid=list(THETA_GRID),
        inference=inference,
        specifications=serialised,
    )
    value["matrix_fingerprint"] = _fingerprint(value)
    return value


def preflight_matrix(
        specifications, *, inference, matrix_output, verify_checkpoints=True):
    """Validate the exact 30-run plan without model inference or training."""
    records = _normalise_records(specifications, inference)
    counts = {}
    roots = {}
    for record in records:
        kind = record["kind"]
        root = record["dataset_root"]
        if kind in roots and roots[kind] != root:
            raise ValueError(f"multiple dataset roots configured for {kind}")
        roots[kind] = root
    for kind, root in roots.items():
        counts[kind] = _dataset_file_count(kind, root)
    expected_counts = {
        record["kind"]: record["sample_count"] for record in records
    }
    if counts != expected_counts:
        raise ValueError(
            f"dataset image counts differ: found={counts}, expected={expected_counts}")

    dataset_signatures = {
        kind: _dataset_source_signature(kind, root)
        for kind, root in roots.items()
    }
    source_hashes = _source_hashes()
    for record in records:
        record["dataset_source_signature"] = dataset_signatures[record["kind"]]
        record["source_sha256"] = source_hashes

    reused = [record for record in records if record["reuse_existing"]]
    if {(row["kind"], row["model"], row["seed"]) for row in reused} != {
            ("mscoco", "yolo11n", 0), ("visdrone", "yolo11n", 0)}:
        raise ValueError("only the two completed YOLO11n seed-0 runs may be reused")
    for record in reused:
        _validate_reused_run(record, inference)

    if verify_checkpoints:
        import torch
        from ultralytics import YOLO

        for index, record in enumerate(records, 1):
            model = YOLO(str(record["weights"]), task="detect")
            _model_names(model, _expected_names(record["kind"]))
            del model
            print(
                f"PREFLIGHT {index:02d}/{len(records)}: {record['label']} "
                f"sha256={record['weights_sha256'][:12]}... OK",
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    size_by_kind = {
        record["kind"]: _directory_size(record["output"])
        for record in reused
    }
    remaining_by_kind = {
        kind: sum(
            record["kind"] == kind and not record["reuse_existing"]
            for record in records)
        for kind in FORMAL_DATASETS
    }
    estimated_remaining = sum(
        size_by_kind[kind] * remaining_by_kind[kind]
        for kind in FORMAL_DATASETS
    )
    free = shutil.disk_usage(PROJECT_ROOT).free
    if free < estimated_remaining * 1.2:
        raise OSError(
            f"insufficient free space: free={free}, estimated={estimated_remaining}")
    plan = _plan_value(records, inference)
    plan["preflight"] = dict(
        status="passed",
        combinations=len(records),
        reused_combinations=len(reused),
        pending_combinations=len(records) - len(reused),
        dataset_image_counts=counts,
        distinct_checkpoint_hashes=len({row["weights_sha256"] for row in records}),
        estimated_remaining_bytes=estimated_remaining,
        free_bytes=free,
        checkpoint_class_mapping_verified=bool(verify_checkpoints),
    )
    print(json.dumps(plan["preflight"], ensure_ascii=False), flush=True)
    print("PREFLIGHT SUCCESS: no training or model inference performed.", flush=True)
    return records, plan


def _convert_threshold_row(row):
    integer_fields = {
        "n_images", "n_gt", "n_pred", "assigned_pred_count",
        "orphan_pred_count", "standard_TP", "standard_FP", "standard_FN",
    }
    nonnumeric = {"metric_version"}
    converted = {}
    for key, value in row.items():
        if key in nonnumeric:
            converted[key] = value
        elif key in integer_fields:
            converted[key] = int(value)
        else:
            converted[key] = float(value)
    return converted


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summarize_primary_gt(path):
    """Stream one primary GT table into interpretable collective diagnostics."""
    summary = dict(
        n_gt=0,
        missed_gt_count=0,
        multi_assigned_gt_count=0,
        positive_coverage_gain_gt_count=0,
        positive_coverage_gain_sum=0.0,
        multigt_source_gt_count=0,
        assigned_prediction_sum=0,
    )
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            summary["n_gt"] += 1
            assigned = int(row["assigned_pred_count"])
            gain = float(row["responsibility_gain"])
            multigt = int(row["assigned_multigt_source_count"])
            summary["assigned_prediction_sum"] += assigned
            summary["missed_gt_count"] += assigned == 0
            summary["multi_assigned_gt_count"] += assigned >= 2
            summary["positive_coverage_gain_gt_count"] += gain > 1e-12
            if gain > 1e-12:
                summary["positive_coverage_gain_sum"] += gain
            summary["multigt_source_gt_count"] += multigt > 0
    detected = summary["n_gt"] - summary["missed_gt_count"]
    summary.update(
        missed_gt_rate=(summary["missed_gt_count"] / summary["n_gt"]),
        multi_assigned_gt_rate=(
            summary["multi_assigned_gt_count"] / summary["n_gt"]),
        multi_assigned_detected_gt_rate=(
            summary["multi_assigned_gt_count"] / detected if detected else 0.0),
        positive_coverage_gain_gt_rate=(
            summary["positive_coverage_gain_gt_count"] / summary["n_gt"]),
        positive_coverage_gain_mean=(
            summary["positive_coverage_gain_sum"]
            / summary["positive_coverage_gain_gt_count"]
            if summary["positive_coverage_gain_gt_count"] else 0.0),
        multigt_source_gt_rate=(
            summary["multigt_source_gt_count"] / summary["n_gt"]),
    )
    return summary


def _run_summary(record):
    output = Path(record["output"])
    complete = _read_json(output / "run_complete.json")
    manifest = _read_json(output / "run_manifest.json")
    if manifest.get("weights_sha256") != record["weights_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {record['label']}")
    if manifest.get("dataset_source_signature") != record["dataset_source_signature"]:
        raise ValueError(f"dataset provenance mismatch: {record['label']}")
    if manifest.get("source_sha256") != record["source_sha256"]:
        raise ValueError(f"core source provenance mismatch: {record['label']}")
    analysis = Path(complete["analysis"])
    thresholds = [
        _convert_threshold_row(row)
        for row in _read_csv(analysis / "vc_thresholds.csv")
    ]
    if [row["confidence_threshold"] for row in thresholds] != list(THETA_GRID):
        raise ValueError(f"theta grid mismatch: {record['label']}")
    primary = next(
        row for row in thresholds
        if row["confidence_threshold"] == PRIMARY_THETA)
    if primary != complete["primary"]:
        raise ValueError(f"primary result mismatch: {record['label']}")
    if not all(
            row["assigned_pred_count"] + row["orphan_pred_count"] == row["n_pred"]
            and row["standard_TP"] + row["standard_FP"] == row["n_pred"]
            and row["standard_TP"] + row["standard_FN"] == row["n_gt"]
            for row in thresholds):
        raise ValueError(f"count conservation failed: {record['label']}")
    best_vc = max(thresholds, key=lambda row: row["VC"])
    best_f1 = max(thresholds, key=lambda row: row["standard_F1"])
    best_lrp = min(thresholds, key=lambda row: row["standard_LRP"])
    recall = {
        float(row["coverage_threshold"]): float(row["RecallACtau"])
        for row in _read_csv(analysis / "recall_ac_tau.csv")
    }
    ap = _read_json(analysis / "diagnostic_ap_summary.json")
    gt = summarize_primary_gt(analysis / "vc_gt_primary.csv")
    if gt["n_gt"] != primary["n_gt"]:
        raise ValueError(f"GT table length mismatch: {record['label']}")
    if gt["assigned_prediction_sum"] != primary["assigned_pred_count"]:
        raise ValueError(f"assigned prediction mismatch: {record['label']}")
    row = dict(
        dataset=record["kind"],
        model=record["model"],
        seed=record["seed"],
        model_id=record["model_id"],
        reused_existing=record["reuse_existing"],
        output=str(output),
        weights_sha256=manifest["weights_sha256"],
        max_det_reached_images=complete["max_det_reached_images"],
        invalid_prediction_count=complete["invalid_prediction_count"],
        diagnostic_AP50=ap["AP50"],
        diagnostic_AP75=ap["AP75"],
        diagnostic_AP50_95=ap["AP50_95"],
        diagnostic_AR50_95=ap["AR50_95"],
        RecallAC025=recall[.25],
        RecallAC050=recall[.5],
        RecallAC075=recall[.75],
        best_VC_theta=best_vc["confidence_threshold"],
        best_VC=best_vc["VC"],
        best_F1_theta=best_f1["confidence_threshold"],
        best_F1=best_f1["standard_F1"],
        best_LRP_theta=best_lrp["confidence_threshold"],
        best_LRP=best_lrp["standard_LRP"],
    )
    row.update(primary)
    row.update(gt)
    combined_thresholds = [dict(
        dataset=record["kind"],
        model=record["model"],
        seed=record["seed"],
        model_id=record["model_id"],
        weights_sha256=manifest["weights_sha256"],
        **threshold,
    ) for threshold in thresholds]
    return row, combined_thresholds


def aggregate_rows(rows, group_fields, metric_fields):
    """Return deterministic mean/sample-std rows for numeric run metrics."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    result = []
    for key in sorted(groups):
        group = groups[key]
        output = dict(zip(group_fields, key))
        output["n_seeds"] = len(group)
        for field in metric_fields:
            values = [float(row[field]) for row in group]
            output[f"{field}_mean"] = statistics.fmean(values)
            output[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0)
        result.append(output)
    return result


PRIMARY_AGGREGATE_METRICS = (
    "VC", "mAC", "mean_responsibility_coverage", "mean_spatial_precision",
    "mean_local_VC", "orphan_rate", "standard_F1", "standard_LRP",
    "diagnostic_AP50", "diagnostic_AP75", "diagnostic_AP50_95",
    "diagnostic_AR50_95", "RecallAC025", "RecallAC050", "RecallAC075",
    "best_VC_theta", "best_F1_theta", "best_LRP_theta",
    "missed_gt_rate", "multi_assigned_gt_rate",
    "multi_assigned_detected_gt_rate", "positive_coverage_gain_gt_rate",
    "positive_coverage_gain_mean", "multigt_source_gt_rate",
)
THRESHOLD_AGGREGATE_METRICS = (
    "VC", "mAC", "mean_responsibility_coverage", "mean_spatial_precision",
    "mean_local_VC", "orphan_rate", "standard_F1", "standard_LRP",
)


def aggregate_matrix(records, matrix_output):
    """Validate all completed combinations and write cross-seed reports."""
    run_rows = []
    threshold_rows = []
    for record in records:
        row, thresholds = _run_summary(record)
        run_rows.append(row)
        threshold_rows.extend(thresholds)
    if len(run_rows) != len(EXPECTED_COMBINATIONS):
        raise AssertionError("formal matrix does not contain 30 completed runs")
    primary_aggregate = aggregate_rows(
        run_rows, ("dataset", "model"), PRIMARY_AGGREGATE_METRICS)
    threshold_aggregate = aggregate_rows(
        threshold_rows, ("dataset", "model", "confidence_threshold"),
        THRESHOLD_AGGREGATE_METRICS)
    if any(row["n_seeds"] != 5 for row in primary_aggregate):
        raise AssertionError("primary aggregate requires five seeds per group")
    if any(row["n_seeds"] != 5 for row in threshold_aggregate):
        raise AssertionError("threshold aggregate requires five seeds per group")

    matrix_output = Path(matrix_output)
    run_fields = list(run_rows[0])
    threshold_fields = list(threshold_rows[0])
    primary_fields = list(primary_aggregate[0])
    aggregate_threshold_fields = list(threshold_aggregate[0])
    _write_csv(matrix_output / "matrix_runs.csv", run_rows, run_fields)
    _write_csv(
        matrix_output / "matrix_thresholds.csv", threshold_rows,
        threshold_fields)
    _write_csv(
        matrix_output / "aggregate_primary.csv", primary_aggregate,
        primary_fields)
    _write_csv(
        matrix_output / "aggregate_thresholds.csv", threshold_aggregate,
        aggregate_threshold_fields)
    return dict(
        run_count=len(run_rows),
        threshold_row_count=len(threshold_rows),
        primary_group_count=len(primary_aggregate),
        threshold_group_count=len(threshold_aggregate),
        files=[
            "matrix_runs.csv", "matrix_thresholds.csv",
            "aggregate_primary.csv", "aggregate_thresholds.csv",
        ],
    )


def run_formal_matrix(
        specifications, *, inference, matrix_output, verify_checkpoints=True,
        expected_matrix_fingerprint=None):
    """Preflight, run/resume 30 combinations, then aggregate five seeds."""
    runtime_dir = PROJECT_ROOT / "results" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    matrix_output = Path(matrix_output).resolve()
    records, plan = preflight_matrix(
        specifications,
        inference=inference,
        matrix_output=matrix_output,
        verify_checkpoints=verify_checkpoints,
    )
    if (expected_matrix_fingerprint is not None
            and plan["matrix_fingerprint"] != expected_matrix_fingerprint):
        raise ValueError(
            "formal matrix differs from the prevalidated fingerprint; "
            "do not start inference")
    # run_dataset_full resolves this name in region_eval.vc_full at runtime.
    # Patch only the callable, not the source file, so completed shards and the
    # frozen source fingerprint remain valid while transient Windows locks retry.
    _vc_full_module._atomic_json = _atomic_json_with_retry
    matrix_output.mkdir(parents=True, exist_ok=True)
    manifest_path = matrix_output / "matrix_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("matrix_fingerprint") != plan["matrix_fingerprint"]:
            raise ValueError(
                f"matrix configuration changed; choose a new output: {matrix_output}")
    else:
        unexpected = list(matrix_output.iterdir())
        if unexpected:
            raise ValueError(
                f"matrix output is nonempty without a manifest: {matrix_output}")
        _atomic_json(manifest_path, dict(
            status="started",
            created_at=datetime.now(timezone.utc).isoformat(),
            **plan,
        ))

    completed = []
    try:
        for index, record in enumerate(records, 1):
            if _source_hashes() != record["source_sha256"]:
                raise ValueError("core VC source files changed after matrix preflight")
            print(
                f"MATRIX {index:02d}/{len(records)} START: {record['label']}",
                flush=True)
            if record["reuse_existing"]:
                result = _validate_reused_run(record, inference)
                print(
                    f"MATRIX {index:02d}/{len(records)} REUSE: "
                    f"{record['label']} -> {record['output']}", flush=True)
            else:
                result = run_dataset_full(
                    kind=record["kind"],
                    dataset_root=record["dataset_root"],
                    weights=record["weights"],
                    output=record["output"],
                    sample_count=record["sample_count"],
                    subset_seed=0,
                    model_id=record["model_id"],
                    **inference,
                )
            completed.append(dict(
                label=record["label"],
                output=str(record["output"]),
                reused=record["reuse_existing"],
                run_fingerprint=result["run_fingerprint"],
            ))
            _atomic_json(matrix_output / "matrix_status.json", dict(
                status="running",
                matrix_fingerprint=plan["matrix_fingerprint"],
                completed_combinations=len(completed),
                total_combinations=len(records),
                last_completed=record["label"],
                updated_at=datetime.now(timezone.utc).isoformat(),
            ))
        aggregation = aggregate_matrix(records, matrix_output)
    except BaseException as exc:
        _atomic_json(matrix_output / "matrix_status.json", dict(
            status="failed_or_interrupted",
            matrix_fingerprint=plan["matrix_fingerprint"],
            completed_combinations=len(completed),
            total_combinations=len(records),
            error=repr(exc),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ))
        raise

    complete = dict(
        status="complete",
        matrix_version=MATRIX_VERSION,
        full_run_version=FULL_RUN_VERSION,
        vc_version=VC_VERSION,
        matrix_fingerprint=plan["matrix_fingerprint"],
        completed_combinations=len(completed),
        reused_combinations=sum(row["reuse_existing"] for row in records),
        newly_managed_combinations=sum(not row["reuse_existing"] for row in records),
        aggregation=aggregation,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_json(matrix_output / "matrix_complete.json", complete)
    _atomic_json(matrix_output / "matrix_status.json", complete)
    print(json.dumps(complete, ensure_ascii=False, allow_nan=False), flush=True)
    print(
        f"SUCCESS: formal 3-model x 5-seed x 2-dataset VC matrix complete: "
        f"{matrix_output}", flush=True)
    return matrix_output
