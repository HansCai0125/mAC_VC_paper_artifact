"""Resumable full-validation cache and VC v1 analysis for existing weights."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

from .case_analysis import _file_hash
from .offline import PROJECT_ROOT
from .vc_smoke import _model_names, _predict, _prepare, _yaml_names
from .vc_v1 import (
    PRIMARY_THETA,
    THETA_GRID,
    VC_VERSION,
    analyze_vc_document,
    write_vc_analysis,
)


FULL_RUN_VERSION = "vc-v1-full-cache-0.1.0"


def _read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON number: {value}")),
    )


def _atomic_json(path, value):
    """Write one complete JSON value, then atomically publish it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return data


def _fingerprint(value):
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return sha256(canonical).hexdigest()


def _safe_image_id(value):
    if not isinstance(value, str) or not value:
        raise ValueError("image id must be a nonempty string")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
           for character in value):
        raise ValueError(f"unsafe image id for shard path: {value}")
    return value


def shard_path(shard_dir, index, image_id):
    if type(index) is not int or index < 0:
        raise ValueError("shard index must be a nonnegative integer")
    return Path(shard_dir) / f"{index:06d}_{_safe_image_id(image_id)}.json"


def _validate_shard(value, *, run_fingerprint, index, item):
    expected_id = item["path"].stem
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid prediction shard schema")
    if value.get("full_run_version") != FULL_RUN_VERSION:
        raise ValueError("prediction shard version mismatch")
    if value.get("run_fingerprint") != run_fingerprint:
        raise ValueError("prediction shard belongs to a different experiment")
    if value.get("index") != index or value.get("image_id") != expected_id:
        raise ValueError("prediction shard index/image mismatch")
    filtered = value.get("filtered_image")
    raw = value.get("raw_image")
    audit = value.get("audit")
    if not all(isinstance(row, dict) for row in (filtered, raw, audit)):
        raise ValueError("prediction shard is incomplete")
    if filtered.get("image_id") != expected_id or raw.get("image_id") != expected_id:
        raise ValueError("prediction shard payload image mismatch")
    if filtered.get("width") != item["width"] or filtered.get("height") != item["height"]:
        raise ValueError("prediction shard dimensions changed")
    return value


def load_existing_shards(items, shard_dir, run_fingerprint):
    """Load valid completed shards and return their indexes plus pending work."""
    completed = {}
    pending = []
    for index, item in enumerate(items):
        path = shard_path(shard_dir, index, item["path"].stem)
        if path.is_file():
            completed[index] = _validate_shard(
                _read_json(path),
                run_fingerprint=run_fingerprint,
                index=index,
                item=item,
            )
        else:
            pending.append(index)
    return completed, pending


def _source_hashes():
    names = (
        "coco.py", "diagnostics.py", "geometry.py", "matching_analysis.py",
        "offline.py", "responsibility_analysis.py", "vc_full.py", "vc_smoke.py",
        "vc_v1.py", "visdrone.py",
    )
    return {
        f"region_eval/{name}": _file_hash(PROJECT_ROOT / "region_eval" / name)
        for name in names
    }


def _directory_signature(path, suffix):
    """Hash names, sizes, and contents for a small annotation directory."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"dataset directory is missing: {path}")
    digest = sha256()
    files = sorted(
        (candidate for candidate in path.iterdir()
         if candidate.is_file() and candidate.suffix.lower() == suffix),
        key=lambda candidate: candidate.name,
    )
    for candidate in files:
        raw = candidate.read_bytes()
        digest.update(candidate.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(raw)).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return dict(count=len(files), sha256=digest.hexdigest())


def _image_manifest_signature(path):
    """Hash image filenames and sizes without rereading the large image corpus."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"dataset image directory is missing: {path}")
    digest = sha256()
    files = sorted(
        (candidate for candidate in path.iterdir()
         if candidate.is_file()
         and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda candidate: candidate.name,
    )
    for candidate in files:
        digest.update(candidate.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(candidate.stat().st_size).encode("ascii"))
        digest.update(b"\0")
    return dict(count=len(files), sha256=digest.hexdigest())


def _dataset_source_signature(kind, dataset_root):
    """Record label contents and the image manifest in the resume fingerprint."""
    dataset_root = Path(dataset_root)
    if kind == "mscoco":
        yaml_path = dataset_root / "data.yaml"
        annotation_path = dataset_root / "annotations" / "instances_val2017.json"
        return dict(
            dataset_yaml_sha256=_file_hash(yaml_path),
            annotations_sha256=_file_hash(annotation_path),
            images=_image_manifest_signature(dataset_root / "images" / "val"),
        )
    if kind == "visdrone":
        split_root = dataset_root / "VisDrone2019-DET-val"
        return dict(
            dataset_yaml_sha256=_file_hash(dataset_root / "visdrone.yaml"),
            annotations=_directory_signature(
                split_root / "annotations", ".txt"),
            images=_image_manifest_signature(split_root / "images"),
        )
    raise ValueError(f"unknown full-run dataset: {kind}")

def _configuration_record(
        *, kind, dataset_root, weights, sample_count, subset_seed, model_id,
        imgsz, confidence_floor, nms_iou, max_det):
    dataset_root = Path(dataset_root).resolve()
    weights = Path(weights).resolve()
    record = dict(
        full_run_version=FULL_RUN_VERSION,
        vc_version=VC_VERSION,
        kind=kind,
        dataset_root=str(dataset_root),
        weights=str(weights),
        weights_sha256=_file_hash(weights),
        sample_count=sample_count,
        subset_seed=subset_seed,
        model_id=model_id,
        inference=dict(
            imgsz=imgsz,
            confidence_floor=confidence_floor,
            nms_iou=nms_iou,
            max_det=max_det,
            class_agnostic_nms=False,
            batch=1,
            half=False,
            augment=False,
            rect=True,
        ),
        primary_theta=PRIMARY_THETA,
        theta_grid=list(THETA_GRID),
        dataset_source_signature=_dataset_source_signature(kind, dataset_root),
        source_sha256=_source_hashes(),
    )
    record["run_fingerprint"] = _fingerprint(record)
    return record


def _manifest_metadata(prepared, configuration, *, device, torch, ultralytics):
    return dict(
        dataset=prepared["dataset"],
        split=f"val_full_{configuration['sample_count']}",
        model_id=configuration["model_id"],
        checkpoint_id=(
            f"{Path(configuration['weights']).name}:sha256:"
            f"{configuration['weights_sha256']}"),
        coordinate_frame="original_image_xyxy",
        ignore_crowd_policy="already_filtered",
        class_ids=list(range(len(prepared["class_names"]))),
        class_names=prepared["class_names"],
        postprocessing=dict(stage="post_nms", **configuration["inference"]),
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
        vc_version=VC_VERSION,
        run_fingerprint=configuration["run_fingerprint"],
        purpose="complete validation cache and VC v1 seed-0 evaluation; no training",
    )


def _completed_run(output, run_fingerprint):
    path = Path(output) / "run_complete.json"
    if not path.is_file():
        return None
    value = _read_json(path)
    if value.get("status") != "complete":
        raise ValueError(f"invalid completion marker: {path}")
    if value.get("run_fingerprint") != run_fingerprint:
        raise ValueError("completed output belongs to a different experiment")
    source = Path(output) / "filtered_predictions.json"
    analysis = Path(value["analysis"])
    analysis_marker = analysis / "analysis_complete.json"
    if not source.is_file() or not analysis_marker.is_file():
        raise ValueError("completion marker references an incomplete analysis")
    recorded = _read_json(analysis_marker)
    if recorded.get("source_sha256") != _file_hash(source):
        raise ValueError("completed prediction cache changed after analysis")
    return value


def run_dataset_full(
        *, kind, dataset_root, weights, output, sample_count, subset_seed=0,
        model_id=None, device="auto", imgsz=640, confidence_floor=0.001,
        nms_iou=0.7, max_det=1000):
    """Run or resume one full validation cache, then analyze it offline."""
    if confidence_floor > min(THETA_GRID):
        raise ValueError("confidence floor must not exceed the theta-grid minimum")
    dataset_root = Path(dataset_root).resolve()
    weights = Path(weights).resolve()
    output = Path(output).resolve()
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise FileNotFoundError(f"checkpoint is missing: {weights}")
    model_id = model_id or weights.parent.parent.name
    configuration = _configuration_record(
        kind=kind,
        dataset_root=dataset_root,
        weights=weights,
        sample_count=sample_count,
        subset_seed=subset_seed,
        model_id=model_id,
        imgsz=imgsz,
        confidence_floor=confidence_floor,
        nms_iou=nms_iou,
        max_det=max_det,
    )
    output.mkdir(parents=True, exist_ok=True)
    existing_manifest = output / "run_manifest.json"
    if existing_manifest.is_file():
        old = _read_json(existing_manifest)
        if old.get("run_fingerprint") != configuration["run_fingerprint"]:
            raise ValueError(
                f"output configuration changed; choose a new RUN_NAME: {output}")
    else:
        unexpected = list(output.iterdir())
        if unexpected:
            raise ValueError(
                f"output is nonempty but has no run manifest: {output}")
        _atomic_json(existing_manifest, dict(
            status="started",
            created_at=datetime.now(timezone.utc).isoformat(),
            **configuration,
        ))
    completed = _completed_run(output, configuration["run_fingerprint"])
    if completed is not None:
        print(
            f"SKIP: {kind} full run is already complete: {output}", flush=True)
        return completed

    prepared = _prepare(kind, dataset_root, sample_count, subset_seed)
    if len(prepared["items"]) != len(prepared["available_names"]):
        raise ValueError("full run must include every validation image")
    _yaml_names(prepared["data_yaml"], prepared["class_names"])

    import torch
    import ultralytics
    from ultralytics import YOLO

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    metadata = _manifest_metadata(
        prepared, configuration, device=device, torch=torch,
        ultralytics=ultralytics)
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards, pending = load_existing_shards(
        prepared["items"], shard_dir, configuration["run_fingerprint"])
    print(
        f"[{prepared['dataset']}] resume check: {len(shards)} completed, "
        f"{len(pending)} pending, total={len(prepared['items'])}", flush=True)

    model = None
    start = perf_counter()
    try:
        if pending:
            model = YOLO(str(weights), task="detect")
            _model_names(model, prepared["class_names"])
            print(f"Loading existing checkpoint: {weights}", flush=True)
        for completed_in_turn, index in enumerate(pending, 1):
            item = prepared["items"][index]
            predictions = _predict(
                model,
                item["path"],
                width=item["width"],
                height=item["height"],
                device=device,
                imgsz=imgsz,
                confidence_floor=confidence_floor,
                nms_iou=nms_iou,
                max_det=max_det,
                output=output,
            )
            kept, prediction_audit = prepared["filter_predictions"](
                predictions, item)
            minimum_score = min(
                (row["score"] for row in predictions), default=None)
            reached = len(predictions) == max_det
            audit = dict(
                image_id=item["path"].stem,
                **item["audit"],
                **prediction_audit,
                max_det_reached=reached,
                minimum_returned_score=minimum_score,
                relevant_cap_risk=bool(
                    reached and minimum_score is not None
                    and minimum_score >= min(THETA_GRID)),
            )
            filtered_image = dict(
                image_id=item["path"].stem,
                width=item["width"],
                height=item["height"],
                gts=item["gts"],
                predictions=kept,
            )
            raw_image = dict(
                image_id=item["path"].stem,
                image_file=str(item["path"]),
                width=item["width"],
                height=item["height"],
                annotations=item["raw_annotations"],
                predictions=predictions,
            )
            shard = dict(
                schema_version=1,
                full_run_version=FULL_RUN_VERSION,
                run_fingerprint=configuration["run_fingerprint"],
                index=index,
                image_id=item["path"].stem,
                raw_image=raw_image,
                filtered_image=filtered_image,
                audit=audit,
            )
            path = shard_path(shard_dir, index, item["path"].stem)
            _atomic_json(path, shard)
            shards[index] = _validate_shard(
                _read_json(path),
                run_fingerprint=configuration["run_fingerprint"],
                index=index,
                item=item,
            )
            total_complete = len(shards)
            print(
                f"[{prepared['dataset']} {total_complete}/{len(prepared['items'])}] "
                f"{item['path'].name}: GT={len(item['gts'])}, "
                f"pred={len(predictions)} -> {len(kept)}; "
                f"this_run={completed_in_turn}/{len(pending)}, "
                f"elapsed={perf_counter() - start:.1f}s",
                flush=True,
            )
            if completed_in_turn % 10 == 0 or completed_in_turn == len(pending):
                _atomic_json(output / "progress.json", dict(
                    status="inference",
                    run_fingerprint=configuration["run_fingerprint"],
                    completed_images=total_complete,
                    total_images=len(prepared["items"]),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                ))
        if len(shards) != len(prepared["items"]):
            raise AssertionError("not all prediction shards are present")

        ordered = [shards[index] for index in range(len(prepared["items"]))]
        audit_rows = [row["audit"] for row in ordered]
        if any(row["relevant_cap_risk"] for row in audit_rows):
            risky = [row["image_id"] for row in audit_rows if row["relevant_cap_risk"]]
            raise ValueError(
                "max_det truncated predictions at or above theta=0.05: "
                + ", ".join(risky[:20]))
        metadata["max_det_reached_images"] = sum(
            row["max_det_reached"] for row in audit_rows)
        metadata["invalid_prediction_count"] = sum(
            len(row["invalid_predictions"]) for row in audit_rows)
        document = dict(
            schema_version=1,
            metadata=metadata,
            images=[row["filtered_image"] for row in ordered],
        )
        source_file = output / "filtered_predictions.json"
        _atomic_json(source_file, document)
        _atomic_json(output / "preprocessing_audit.json", audit_rows)
        _atomic_json(output / "cache_complete.json", dict(
            status="complete",
            run_fingerprint=configuration["run_fingerprint"],
            n_images=len(ordered),
            filtered_predictions_sha256=_file_hash(source_file),
            completed_at=datetime.now(timezone.utc).isoformat(),
        ))

        print(
            f"[{prepared['dataset']}] inference cache complete; starting offline "
            f"{len(THETA_GRID)}-threshold VC analysis...", flush=True)
        analysis = analyze_vc_document(document)
        analysis_dir = output / "analysis"
        if (analysis_dir / "analysis_complete.json").is_file():
            recorded = _read_json(analysis_dir / "analysis_complete.json")
            if recorded.get("source_sha256") != _file_hash(source_file):
                analysis_dir = output / (
                    "analysis_retry_" +
                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        elif analysis_dir.exists():
            analysis_dir = output / (
                "analysis_retry_" +
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        if not (analysis_dir / "analysis_complete.json").is_file():
            write_vc_analysis(
                analysis_dir,
                document=document,
                source_file=source_file,
                analysis=analysis,
            )
        primary = next(
            row for row in analysis["threshold_rows"]
            if row["confidence_threshold"] == PRIMARY_THETA)
        complete = dict(
            status="complete",
            full_run_version=FULL_RUN_VERSION,
            vc_version=VC_VERSION,
            run_fingerprint=configuration["run_fingerprint"],
            dataset=prepared["dataset"],
            output=str(output),
            analysis=str(analysis_dir),
            n_images=len(ordered),
            elapsed_seconds_this_invocation=perf_counter() - start,
            max_det_reached_images=metadata["max_det_reached_images"],
            invalid_prediction_count=metadata["invalid_prediction_count"],
            primary=primary,
        )
        _atomic_json(output / "run_complete.json", complete)
        _atomic_json(output / "progress.json", dict(
            status="complete",
            run_fingerprint=configuration["run_fingerprint"],
            completed_images=len(ordered),
            total_images=len(ordered),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ))
    except Exception as exc:
        _atomic_json(output / "last_failure.json", dict(
            status="failed",
            run_fingerprint=configuration["run_fingerprint"],
            error=repr(exc),
            completed_shards=len(shards),
            total_images=len(prepared["items"]),
            failed_at=datetime.now(timezone.utc).isoformat(),
        ))
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(complete, ensure_ascii=False, allow_nan=False), flush=True)
    print(
        f"SUCCESS: {prepared['dataset']} full seed-0 VC v1 run complete. "
        "No training performed.", flush=True)
    return complete


def run_full_suite(configurations, *, output):
    """Run/resume each dataset and write a suite completion marker."""
    runtime_dir = PROJECT_ROOT / "results" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite_manifest = output / "suite_manifest.json"
    manifest_value = dict(
        status="started",
        full_run_version=FULL_RUN_VERSION,
        vc_version=VC_VERSION,
        configurations=[
            {key: str(value) if isinstance(value, Path) else value
             for key, value in configuration.items()}
            for configuration in configurations
        ],
    )
    if suite_manifest.is_file():
        existing = _read_json(suite_manifest)
        if (existing.get("full_run_version") != FULL_RUN_VERSION
                or existing.get("vc_version") != VC_VERSION):
            raise ValueError("suite output belongs to a different implementation")
    else:
        _atomic_json(suite_manifest, manifest_value)
    results = []
    try:
        for configuration in configurations:
            results.append(run_dataset_full(
                **configuration,
                output=output / configuration["kind"],
            ))
    except Exception as exc:
        _atomic_json(output / "suite_last_failure.json", dict(
            status="failed",
            error=repr(exc),
            completed_datasets=len(results),
            failed_at=datetime.now(timezone.utc).isoformat(),
        ))
        raise
    complete = dict(
        status="complete",
        full_run_version=FULL_RUN_VERSION,
        vc_version=VC_VERSION,
        results=results,
    )
    _atomic_json(output / "suite_complete.json", complete)
    print(f"SUCCESS: full seed-0 two-dataset suite complete: {output}", flush=True)
    return output
