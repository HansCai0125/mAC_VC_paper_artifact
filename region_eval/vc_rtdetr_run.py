"""Inference orchestration for the complete-validation RT-DETR VC suite."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter

from .case_analysis import _file_hash
from .offline import PROJECT_ROOT
from .vc_full import (
    FULL_RUN_VERSION,
    _completed_run,
    _read_json,
    _validate_shard,
    load_existing_shards,
    shard_path,
)
from .vc_matrix import _atomic_json_with_retry as _atomic_json
from .vc_rtdetr_full import (
    RTDETR_RUN_VERSION,
    _configuration_record,
    _fingerprint,
    _manifest_metadata,
    _predict_rtdetr,
    preflight_rtdetr_suite,
)
from .vc_smoke import _model_names, _prepare, _yaml_names
from .vc_v1 import (
    PRIMARY_THETA,
    THETA_GRID,
    VC_VERSION,
    analyze_vc_document,
    write_vc_analysis,
)


_RESUME_MIGRATION_OLD_SOURCE_SHA256 = {
    "region_eval/offline.py":
        "2c59d1269e9793596e080327af256253169893eb19237a2476944f01d6a60616",
    "region_eval/vc_rtdetr_run.py":
        "9aec0d53fe45dcd2cb2202d992d4bd082c27a12d8dbb77521c5fc07887598aae",
}


def _resume_configuration_view(record):
    """Return cache-affecting fields, excluding the approved analysis fix."""
    view = {
        key: value for key, value in record.items()
        if key not in {"status", "created_at", "run_fingerprint"}
    }
    sources = dict(view.get("source_sha256", {}))
    for key in _RESUME_MIGRATION_OLD_SOURCE_SHA256:
        sources.pop(key, None)
    view["source_sha256"] = sources
    return view


def _accepted_source_sha256(record):
    sources = record.get("source_sha256", {})
    return {
        key: sources.get(key) for key in _RESUME_MIGRATION_OLD_SOURCE_SHA256
    }


def _resume_configuration_compatible(existing, current, marker=None):
    """Allow only the known post-inference analysis compatibility repair."""
    if _resume_configuration_view(existing) != _resume_configuration_view(current):
        return False
    old_sources = existing.get("source_sha256", {})
    current_sources = current.get("source_sha256", {})
    changed = {
        key for key in set(old_sources) | set(current_sources)
        if old_sources.get(key) != current_sources.get(key)
    }
    if not changed or not changed.issubset(_RESUME_MIGRATION_OLD_SOURCE_SHA256):
        return False
    if any(
            old_sources.get(key) != _RESUME_MIGRATION_OLD_SOURCE_SHA256[key]
            for key in changed):
        return False
    if marker is not None:
        if marker.get("status") != "accepted_analysis_resume_migration":
            return False
        if marker.get("accepted_source_sha256") != _accepted_source_sha256(current):
            return False
    return True


def _suite_resume_compatible(existing, current, marker=None):
    scalar_keys = (
        "rtdetr_run_version", "vc_version", "device", "imgsz",
        "confidence_floor", "max_det",
    )
    if any(existing.get(key) != current.get(key) for key in scalar_keys):
        return False
    old_configs = existing.get("configurations")
    new_configs = current.get("configurations")
    if not isinstance(old_configs, list) or len(old_configs) != len(new_configs):
        return False
    return all(
        _resume_configuration_compatible(old, new, marker)
        for old, new in zip(old_configs, new_configs)
    )


def _write_resume_migration(path, *, old_fingerprint, new_fingerprint,
                            current_configuration, scope):
    _atomic_json(path, dict(
        status="accepted_analysis_resume_migration",
        scope=scope,
        reason=(
            "Preserve completed inference shards after adding support for the "
            "native RT-DETR no-NMS postprocessing stage."),
        cached_fingerprint=old_fingerprint,
        current_fingerprint=new_fingerprint,
        accepted_source_sha256=_accepted_source_sha256(current_configuration),
        migrated_at=datetime.now(timezone.utc).isoformat()))


def run_rtdetr_dataset_full(
        *, kind, dataset_root, weights, output, sample_count, seed,
        model_id, device="auto", imgsz=640, confidence_floor=0.001,
        max_det=1000):
    """Run or resume one complete RT-DETR validation cache and VC analysis."""
    dataset_root = Path(dataset_root).resolve()
    weights = Path(weights).resolve()
    output = Path(output).resolve()
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise FileNotFoundError(f"checkpoint is missing: {weights}")
    configuration = _configuration_record(
        kind=kind, dataset_root=dataset_root, weights=weights,
        sample_count=sample_count, seed=seed, model_id=model_id,
        imgsz=imgsz, confidence_floor=confidence_floor, max_det=max_det)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        old = _read_json(manifest_path)
        if old.get("run_fingerprint") != configuration["run_fingerprint"]:
            migration_path = output / "analysis_resume_migration.json"
            marker = _read_json(migration_path) if migration_path.is_file() else None
            current_configuration = configuration
            if not _resume_configuration_compatible(old, configuration, marker):
                raise ValueError(
                    f"output configuration changed; choose a new RUN_NAME: {output}")
            if marker is None:
                _write_resume_migration(
                    migration_path,
                    old_fingerprint=old["run_fingerprint"],
                    new_fingerprint=configuration["run_fingerprint"],
                    current_configuration=current_configuration,
                    scope="run")
            configuration = {
                key: old[key] for key in current_configuration
            }
            print(
                "RESUME COMPATIBILITY: reusing verified inference shards after "
                "the RT-DETR offline-analysis fix.", flush=True)
    else:
        if list(output.iterdir()):
            raise ValueError(f"output is nonempty without a run manifest: {output}")
        _atomic_json(manifest_path, dict(
            status="started", created_at=datetime.now(timezone.utc).isoformat(),
            **configuration))
    completed = _completed_run(output, configuration["run_fingerprint"])
    if completed is not None:
        print(f"SKIP: RT-DETR run already complete: {output}", flush=True)
        return completed

    prepared = _prepare(kind, dataset_root, sample_count, 0)
    if len(prepared["items"]) != len(prepared["available_names"]):
        raise ValueError("RT-DETR full run must include every validation image")
    _yaml_names(prepared["data_yaml"], prepared["class_names"])

    import torch
    import ultralytics
    from ultralytics import RTDETR

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
        f"[{prepared['dataset']} {model_id}] resume check: "
        f"{len(shards)} completed, {len(pending)} pending, "
        f"total={len(prepared['items'])}", flush=True)

    model = None
    start = perf_counter()
    try:
        if pending:
            model = RTDETR(str(weights))
            _model_names(model, prepared["class_names"])
            predictor = model._smart_load("predictor")
            if predictor.__name__ != "RTDETRPredictor":
                raise ValueError(f"unexpected RT-DETR predictor: {predictor.__name__}")
            print(f"Loading existing RT-DETR checkpoint: {weights}", flush=True)
        for completed_in_turn, index in enumerate(pending, 1):
            item = prepared["items"][index]
            predictions = _predict_rtdetr(
                model, item["path"], width=item["width"], height=item["height"],
                device=device, imgsz=imgsz, confidence_floor=confidence_floor,
                max_det=max_det, output=output)
            kept, prediction_audit = prepared["filter_predictions"](
                predictions, item)
            minimum_score = min(
                (row["score"] for row in predictions), default=None)
            reached = len(predictions) == max_det
            audit = dict(
                image_id=item["path"].stem, **item["audit"], **prediction_audit,
                max_det_reached=reached,
                minimum_returned_score=minimum_score,
                relevant_cap_risk=bool(
                    reached and minimum_score is not None
                    and minimum_score >= min(THETA_GRID)))
            filtered_image = dict(
                image_id=item["path"].stem, width=item["width"],
                height=item["height"], gts=item["gts"], predictions=kept)
            raw_image = dict(
                image_id=item["path"].stem, image_file=str(item["path"]),
                width=item["width"], height=item["height"],
                annotations=item["raw_annotations"], predictions=predictions)
            shard = dict(
                schema_version=1, full_run_version=FULL_RUN_VERSION,
                rtdetr_run_version=RTDETR_RUN_VERSION,
                run_fingerprint=configuration["run_fingerprint"], index=index,
                image_id=item["path"].stem, raw_image=raw_image,
                filtered_image=filtered_image, audit=audit)
            path = shard_path(shard_dir, index, item["path"].stem)
            _atomic_json(path, shard)
            shards[index] = _validate_shard(
                _read_json(path), run_fingerprint=configuration["run_fingerprint"],
                index=index, item=item)
            total_complete = len(shards)
            print(
                f"[{prepared['dataset']} {model_id} "
                f"{total_complete}/{len(prepared['items'])}] "
                f"{item['path'].name}: GT={len(item['gts'])}, "
                f"pred={len(predictions)} -> {len(kept)}; "
                f"this_run={completed_in_turn}/{len(pending)}, "
                f"elapsed={perf_counter() - start:.1f}s", flush=True)
            if completed_in_turn % 10 == 0 or completed_in_turn == len(pending):
                _atomic_json(output / "progress.json", dict(
                    status="inference",
                    run_fingerprint=configuration["run_fingerprint"],
                    completed_images=total_complete,
                    total_images=len(prepared["items"]),
                    updated_at=datetime.now(timezone.utc).isoformat()))

        if len(shards) != len(prepared["items"]):
            raise AssertionError("not all RT-DETR prediction shards are present")
        ordered = [shards[index] for index in range(len(prepared["items"]))]
        audit_rows = [row["audit"] for row in ordered]
        risky = [row["image_id"] for row in audit_rows if row["relevant_cap_risk"]]
        if risky:
            raise ValueError(
                "max_det truncated RT-DETR predictions at or above theta=0.05: "
                + ", ".join(risky[:20]))
        metadata["max_det_reached_images"] = sum(
            row["max_det_reached"] for row in audit_rows)
        metadata["invalid_prediction_count"] = sum(
            len(row["invalid_predictions"]) for row in audit_rows)
        document = dict(
            schema_version=1, metadata=metadata,
            images=[row["filtered_image"] for row in ordered])
        source_file = output / "filtered_predictions.json"
        _atomic_json(source_file, document)
        _atomic_json(output / "preprocessing_audit.json", audit_rows)
        _atomic_json(output / "cache_complete.json", dict(
            status="complete", run_fingerprint=configuration["run_fingerprint"],
            n_images=len(ordered),
            filtered_predictions_sha256=_file_hash(source_file),
            completed_at=datetime.now(timezone.utc).isoformat()))

        print(
            f"[{prepared['dataset']} {model_id}] cache complete; starting "
            f"offline {len(THETA_GRID)}-threshold VC analysis...", flush=True)
        analysis = analyze_vc_document(document)
        analysis_dir = output / "analysis"
        if analysis_dir.exists() and not (
                analysis_dir / "analysis_complete.json").is_file():
            analysis_dir = output / (
                "analysis_retry_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        if not (analysis_dir / "analysis_complete.json").is_file():
            write_vc_analysis(
                analysis_dir, document=document, source_file=source_file,
                analysis=analysis)
        primary = next(
            row for row in analysis["threshold_rows"]
            if row["confidence_threshold"] == PRIMARY_THETA)
        complete = dict(
            status="complete", full_run_version=FULL_RUN_VERSION,
            rtdetr_run_version=RTDETR_RUN_VERSION, vc_version=VC_VERSION,
            run_fingerprint=configuration["run_fingerprint"],
            dataset=prepared["dataset"], model_id=model_id,
            training_seed=seed, output=str(output), analysis=str(analysis_dir),
            n_images=len(ordered),
            elapsed_seconds_this_invocation=perf_counter() - start,
            max_det_reached_images=metadata["max_det_reached_images"],
            invalid_prediction_count=metadata["invalid_prediction_count"],
            primary=primary)
        _atomic_json(output / "run_complete.json", complete)
        _atomic_json(output / "progress.json", dict(
            status="complete", run_fingerprint=configuration["run_fingerprint"],
            completed_images=len(ordered), total_images=len(ordered),
            updated_at=datetime.now(timezone.utc).isoformat()))
    except Exception as exc:
        _atomic_json(output / "last_failure.json", dict(
            status="failed", run_fingerprint=configuration["run_fingerprint"],
            error=repr(exc), completed_shards=len(shards),
            total_images=len(prepared["items"]),
            failed_at=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(complete, ensure_ascii=False, allow_nan=False), flush=True)
    print(
        f"SUCCESS: {prepared['dataset']} {model_id} RT-DETR VC run complete. "
        "No training performed.", flush=True)
    return complete


def run_rtdetr_suite(
        configurations, *, output, device="auto", imgsz=640,
        confidence_floor=0.001, max_det=1000):
    """Preflight and run/resume the six full-validation RT-DETR evaluations."""
    runtime_dir = PROJECT_ROOT / "outputs" / "vc_v1" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"

    records, preflight = preflight_rtdetr_suite(
        configurations, imgsz=imgsz, confidence_floor=confidence_floor,
        max_det=max_det, verify_checkpoints=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        rtdetr_run_version=RTDETR_RUN_VERSION, vc_version=VC_VERSION,
        device=device, imgsz=imgsz, confidence_floor=confidence_floor,
        max_det=max_det, configurations=records)
    plan["suite_fingerprint"] = _fingerprint(plan)
    plan["preflight"] = preflight
    manifest_path = output / "suite_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("suite_fingerprint") != plan["suite_fingerprint"]:
            migration_path = output / "suite_resume_migration.json"
            marker = _read_json(migration_path) if migration_path.is_file() else None
            current_suite_fingerprint = plan["suite_fingerprint"]
            if not _suite_resume_compatible(existing, plan, marker):
                raise ValueError("RT-DETR suite changed; use a new RUN_NAME")
            if marker is None:
                _write_resume_migration(
                    migration_path,
                    old_fingerprint=existing["suite_fingerprint"],
                    new_fingerprint=current_suite_fingerprint,
                    current_configuration=plan["configurations"][0],
                    scope="suite")
            plan["suite_fingerprint"] = existing["suite_fingerprint"]
            print(
                "RESUME COMPATIBILITY: suite inputs are unchanged; accepting "
                "the recorded RT-DETR offline-analysis fix.", flush=True)
    else:
        _atomic_json(manifest_path, dict(
            status="started", created_at=datetime.now(timezone.utc).isoformat(),
            **plan))

    results = []
    try:
        for index, configuration in enumerate(configurations, 1):
            print(
                f"RT-DETR RUN {index:02d}/{len(configurations):02d} START: "
                f"{configuration['kind']}:{configuration['model_id']}",
                flush=True)
            run_output = (
                output / "runs" / configuration["kind"] / "rtdetr_l"
                / f"seed_{configuration['seed']}")
            result = run_rtdetr_dataset_full(
                **configuration, output=run_output, device=device, imgsz=imgsz,
                confidence_floor=confidence_floor, max_det=max_det)
            results.append(result)
            _atomic_json(output / "suite_status.json", dict(
                status="running", suite_fingerprint=plan["suite_fingerprint"],
                completed_runs=len(results), total_runs=len(configurations),
                last_completed=configuration["model_id"],
                updated_at=datetime.now(timezone.utc).isoformat()))
    except BaseException as exc:
        _atomic_json(output / "suite_status.json", dict(
            status="failed_or_interrupted",
            suite_fingerprint=plan["suite_fingerprint"],
            completed_runs=len(results), total_runs=len(configurations),
            error=repr(exc), updated_at=datetime.now(timezone.utc).isoformat()))
        raise

    complete = dict(
        status="complete", rtdetr_run_version=RTDETR_RUN_VERSION,
        vc_version=VC_VERSION, suite_fingerprint=plan["suite_fingerprint"],
        completed_runs=len(results), total_runs=len(configurations),
        results=results, completed_at=datetime.now(timezone.utc).isoformat())
    _atomic_json(output / "suite_complete.json", complete)
    _atomic_json(output / "suite_status.json", complete)
    print(f"SUCCESS: RT-DETR full-validation suite complete: {output}", flush=True)
    return output
