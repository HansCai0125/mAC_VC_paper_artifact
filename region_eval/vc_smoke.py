"""Two-dataset, existing-checkpoint smoke runner for VC v1.0."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter

from .case_analysis import _file_hash
from .coco import (
    COCO_CROWD_POLICY,
    COCO_NAMES,
    filter_coco_predictions,
    prepare_coco_image,
    select_coco_images,
)
from .offline import PROJECT_ROOT
from .vc_v1 import (
    PRIMARY_THETA,
    THETA_GRID,
    VC_VERSION,
    analyze_vc_document,
    write_vc_analysis,
)
from .visdrone import (
    CLASS_NAMES as VISDRONE_NAMES,
    IGNORE_POLICY as VISDRONE_IGNORE_POLICY,
    IGNORE_REFERENCE as VISDRONE_IGNORE_REFERENCE,
    filter_predictions as filter_visdrone_predictions,
    parse_annotations as parse_visdrone_annotations,
    prepare_ground_truth as prepare_visdrone_ground_truth,
    select_images as select_visdrone_images,
)
from .visdrone_run import _save_json


SMOKE_VERSION = "vc-v1-smoke-0.1.0"


def _yaml_names(data_yaml, expected):
    import yaml
    value = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8-sig"))
    names = value.get("names") if isinstance(value, dict) else None
    if isinstance(names, dict) and set(names) == set(range(len(expected))):
        names = [names[index] for index in range(len(expected))]
    if names != expected:
        raise ValueError(f"dataset YAML class names do not match: {data_yaml}")


def _model_names(model, expected):
    names = model.names
    if isinstance(names, dict) and set(names) == set(range(len(expected))):
        names = [names[index] for index in range(len(expected))]
    if model.task != "detect" or names != expected:
        raise ValueError("checkpoint task/class names differ from the selected dataset")


def _prepare_visdrone(dataset_root, count, seed):
    from PIL import Image

    split_root = dataset_root / "VisDrone2019-DET-val"
    selected, available_names = select_visdrone_images(
        split_root / "images", split_root / "annotations", count, seed)
    items = []
    for image_path in selected:
        annotation_path = split_root / "annotations" / f"{image_path.stem}.txt"
        raw = annotation_path.read_bytes()
        with Image.open(image_path) as image:
            width, height = image.size
        annotations = parse_visdrone_annotations(raw.decode("utf-8-sig"))
        gts, ignore_rects, audit = prepare_visdrone_ground_truth(
            annotations, width, height)
        items.append(dict(
            path=image_path,
            width=width,
            height=height,
            gts=gts,
            raw_annotations=annotations,
            ignore_data=ignore_rects,
            audit=audit,
        ))
    return dict(
        dataset="VisDrone2019-DET",
        slug="visdrone",
        class_names=VISDRONE_NAMES,
        data_yaml=dataset_root / "visdrone.yaml",
        items=items,
        available_names=available_names,
        preprocessing=dict(
            protocol=VISDRONE_IGNORE_POLICY,
            reference=VISDRONE_IGNORE_REFERENCE,
            ignore_overlap_threshold=0.5,
            geometry="continuous clipped xyxy",
            official_standard_metrics_equivalent=False,
        ),
        filter_predictions=lambda predictions, item: filter_visdrone_predictions(
            predictions, item["ignore_data"], item["width"], item["height"]),
    )


def _prepare_coco(dataset_root, count, seed):
    from PIL import Image

    selected, available_names, category_map, annotation_path = select_coco_images(
        dataset_root, count, seed)
    items = []
    for image_row, annotations in selected:
        image_path = dataset_root / "images" / "val" / image_row["file_name"]
        with Image.open(image_path) as image:
            actual_size = image.size
        declared_size = (image_row["width"], image_row["height"])
        if actual_size != declared_size:
            raise ValueError(f"COCO image size differs from JSON: {image_path}")
        gts, crowd_by_class, audit = prepare_coco_image(
            image_row, annotations, category_map)
        raw_annotations = [
            {key: annotation.get(key) for key in (
                "id", "image_id", "category_id", "bbox", "area", "iscrowd", "ignore")
             if key in annotation}
            for annotation in annotations
        ]
        items.append(dict(
            path=image_path,
            width=image_row["width"],
            height=image_row["height"],
            gts=gts,
            raw_annotations=raw_annotations,
            ignore_data=crowd_by_class,
            audit=audit,
        ))
    return dict(
        dataset="MSCOCO2017",
        slug="mscoco",
        class_names=COCO_NAMES,
        data_yaml=dataset_root / "data.yaml",
        annotation_file=annotation_path,
        items=items,
        available_names=available_names,
        preprocessing=dict(
            protocol=COCO_CROWD_POLICY,
            reference=str(annotation_path),
            ignore_overlap_threshold=0.5,
            apply_to="same-class predictions against crowd/ignore bounding boxes",
            geometry="continuous clipped xyxy; crowd segmentation masks are not rasterized",
            official_standard_metrics_equivalent=False,
        ),
        filter_predictions=lambda predictions, item: filter_coco_predictions(
            predictions, item["ignore_data"], item["width"], item["height"]),
    )


def _prepare(kind, dataset_root, count, seed):
    if kind == "visdrone":
        return _prepare_visdrone(dataset_root, count, seed)
    if kind == "mscoco":
        return _prepare_coco(dataset_root, count, seed)
    raise ValueError(f"unknown smoke dataset: {kind}")


def _predict(model, path, *, width, height, device, imgsz, confidence_floor,
             nms_iou, max_det, output):
    results = model.predict(
        source=str(path),
        imgsz=imgsz,
        conf=confidence_floor,
        iou=nms_iou,
        max_det=max_det,
        device=device,
        batch=1,
        half=False,
        augment=False,
        agnostic_nms=False,
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
        raise ValueError(f"unexpected prediction image shape/count for {path.name}")
    if Path(results[0].path).resolve() != path.resolve():
        raise ValueError("prediction source path differs from the selected image")
    boxes = results[0].boxes
    coordinates = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    if len(coordinates) > max_det or not all(float(value).is_integer() for value in classes):
        raise ValueError("invalid model prediction count/class")
    return [
        dict(id=f"pred_{index:06d}", xyxy=rect, score=score, class_id=int(class_id))
        for index, (rect, score, class_id) in enumerate(
            zip(coordinates, scores, classes))
    ]


def run_dataset_smoke(
        *,
        kind,
        dataset_root,
        weights,
        output,
        sample_count=20,
        subset_seed=0,
        model_id=None,
        device="auto",
        imgsz=640,
        confidence_floor=0.001,
        nms_iou=0.7,
        max_det=1000):
    """Run one small existing-checkpoint inference and full offline VC analysis."""
    if confidence_floor > min(THETA_GRID):
        raise ValueError("confidence floor must not exceed the theta-grid minimum")
    dataset_root = Path(dataset_root).resolve()
    weights = Path(weights).resolve()
    output = Path(output).resolve()
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise FileNotFoundError(f"checkpoint is missing: {weights}")
    if output.exists():
        raise ValueError(f"smoke output already exists: {output}")
    prepared = _prepare(kind, dataset_root, sample_count, subset_seed)
    _yaml_names(prepared["data_yaml"], prepared["class_names"])

    import torch
    import ultralytics
    from ultralytics import YOLO

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(weights), task="detect")
    _model_names(model, prepared["class_names"])
    model_id = model_id or weights.parent.parent.name
    package_root = Path(ultralytics.__file__).parent
    metadata = dict(
        dataset=prepared["dataset"],
        split=f"val_smoke_{sample_count}",
        model_id=model_id,
        checkpoint_id=f"{weights.name}:sha256:{_file_hash(weights)}",
        coordinate_frame="original_image_xyxy",
        ignore_crowd_policy="already_filtered",
        class_ids=list(range(len(prepared["class_names"]))),
        class_names=prepared["class_names"],
        postprocessing=dict(
            stage="post_nms",
            confidence_floor=confidence_floor,
            nms_iou=nms_iou,
            max_det=max_det,
            class_agnostic_nms=False,
            imgsz=imgsz,
            batch=1,
            half=False,
            augment=False,
            rect=True,
        ),
        preprocessing=prepared["preprocessing"],
        subset=dict(
            method="smallest SHA256(seed:filename), then filename order",
            seed=subset_seed,
            available_count=len(prepared["available_names"]),
            count=len(prepared["items"]),
            available_names_sha256=sha256(
                json.dumps(prepared["available_names"]).encode()).hexdigest(),
            selected_names=[item["path"].name for item in prepared["items"]],
        ),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            ultralytics=ultralytics.__version__,
            ultralytics_path=str(package_root),
            device=str(device),
        ),
        weights_file=str(weights),
        weights_sha256=_file_hash(weights),
        dataset_yaml_file=str(prepared["data_yaml"]),
        dataset_yaml_sha256=_file_hash(prepared["data_yaml"]),
        smoke_version=SMOKE_VERSION,
        vc_version=VC_VERSION,
        purpose="small two-dataset pipeline smoke test; not a paper benchmark result",
    )
    _save_json(output / "run_manifest.json", dict(status="started", metadata=metadata))
    raw_images, filtered_images, audit_rows = [], [], []
    start = perf_counter()
    try:
        print(
            f"[{prepared['dataset']}] Python {platform.python_version()}, "
            f"torch {torch.__version__}, ultralytics {ultralytics.__version__}, "
            f"device={device}", flush=True)
        print(f"Loading existing checkpoint: {weights}", flush=True)
        for index, item in enumerate(prepared["items"], 1):
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
            min_score = min((row["score"] for row in predictions), default=None)
            reached = len(predictions) == max_det
            audit = dict(
                image_id=item["path"].stem,
                **item["audit"],
                **prediction_audit,
                max_det_reached=reached,
                minimum_returned_score=min_score,
                relevant_cap_risk=bool(
                    reached and min_score is not None and min_score >= min(THETA_GRID)),
            )
            record = dict(
                image_id=item["path"].stem,
                width=item["width"],
                height=item["height"],
                gts=item["gts"],
                predictions=kept,
            )
            raw_images.append(dict(
                image_id=item["path"].stem,
                width=item["width"],
                height=item["height"],
                annotations=item["raw_annotations"],
                predictions=predictions,
            ))
            filtered_images.append(record)
            audit_rows.append(audit)
            print(
                f"[{prepared['dataset']} {index}/{len(prepared['items'])}] "
                f"{item['path'].name}: GT={len(item['gts'])}, "
                f"pred={len(predictions)} -> {len(kept)}; "
                f"elapsed={perf_counter() - start:.1f}s",
                flush=True,
            )
        if any(row["relevant_cap_risk"] for row in audit_rows):
            raise ValueError(
                "max_det truncated predictions at or above theta=0.05; increase max_det")

        metadata["preprocessing_audit"] = audit_rows
        metadata["max_det_reached_images"] = sum(
            row["max_det_reached"] for row in audit_rows)
        document = dict(
            schema_version=1,
            metadata=metadata,
            images=filtered_images,
        )
        _save_json(output / "raw_predictions.json", dict(
            raw_cache_version=1, metadata=metadata, images=raw_images))
        source_file = output / "filtered_predictions.json"
        _save_json(source_file, document)
        _save_json(output / "preprocessing_audit.json", audit_rows)
        analysis = analyze_vc_document(document)
        analysis_dir = write_vc_analysis(
            output / "analysis",
            document=document,
            source_file=source_file,
            analysis=analysis,
        )
        primary = next(
            row for row in analysis["threshold_rows"]
            if row["confidence_threshold"] == PRIMARY_THETA)
        complete = dict(
            status="complete",
            smoke_version=SMOKE_VERSION,
            vc_version=VC_VERSION,
            dataset=prepared["dataset"],
            output=str(output),
            analysis=str(analysis_dir),
            elapsed_seconds=perf_counter() - start,
            primary=primary,
            max_det_reached_images=metadata["max_det_reached_images"],
        )
        _save_json(output / "run_complete.json", complete)
    except Exception as exc:
        _save_json(output / "run_failed.json", dict(
            status="failed", error=repr(exc), completed_images=len(filtered_images)))
        raise
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(complete, ensure_ascii=False, allow_nan=False), flush=True)
    print(
        f"SUCCESS: {prepared['dataset']} VC v1 smoke test complete. "
        "No training performed.", flush=True)
    return complete


def preflight_smoke_configuration(configuration, *, sample_count=2, subset_seed=0):
    """Validate local data, YAML, checkpoint classes, and imports without inference."""
    runtime_dir = PROJECT_ROOT / "results" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    dataset_root = Path(configuration["dataset_root"]).resolve()
    weights = Path(configuration["weights"]).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {weights}")
    prepared = _prepare(
        configuration["kind"], dataset_root, sample_count, subset_seed)
    _yaml_names(prepared["data_yaml"], prepared["class_names"])
    import torch
    import ultralytics
    from ultralytics import YOLO
    model = YOLO(str(weights), task="detect")
    _model_names(model, prepared["class_names"])
    result = dict(
        dataset=prepared["dataset"],
        available_images=len(prepared["available_names"]),
        checked_images=len(prepared["items"]),
        model_id=configuration.get("model_id") or weights.parent.parent.name,
        weights_sha256=_file_hash(weights),
        torch=torch.__version__,
        ultralytics=ultralytics.__version__,
        cuda_available=torch.cuda.is_available(),
        status="ready",
    )
    del model
    return result

def run_smoke_suite(configurations, *, output=None):
    """Run each configured dataset in one timestamped, non-overwriting suite."""
    runtime_dir = PROJECT_ROOT / "results" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    output = Path(output).resolve() if output else (
        PROJECT_ROOT / "results" /
        ("smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    if output.exists():
        raise ValueError(f"suite output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _save_json(output / "suite_manifest.json", dict(
        status="started",
        smoke_version=SMOKE_VERSION,
        vc_version=VC_VERSION,
        configurations=[
            {key: str(value) if isinstance(value, Path) else value
             for key, value in configuration.items()}
            for configuration in configurations
        ],
    ))
    results = []
    for configuration in configurations:
        results.append(run_dataset_smoke(
            **configuration,
            output=output / configuration["kind"],
        ))
    _save_json(output / "suite_complete.json", dict(
        status="complete",
        smoke_version=SMOKE_VERSION,
        vc_version=VC_VERSION,
        results=results,
    ))
    print(f"SUCCESS: two-dataset smoke suite complete: {output}", flush=True)
    return output
