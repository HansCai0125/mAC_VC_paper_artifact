"""Existing-weight VisDrone smoke evaluation; inference is kept outside the metric core."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter

from .diagnostics import Config, aggregate
from .offline import PROJECT_ROOT, evaluate_document, write_report
from .visdrone import (CLASS_NAMES, IGNORE_POLICY, IGNORE_REFERENCE, filter_predictions,
                       parse_annotations, prepare_ground_truth, select_images)


def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # Write exactly the bytes used by the report hash, including on Windows.
    data = text.encode("utf-8")
    with Path(path).open("xb") as handle:
        handle.write(data)
    return data


def run_visdrone(*, dataset_root, weights, sample_count=50, seed=0, device="auto",
                 imgsz=640, confidence_floor=0.001, nms_iou=0.7, max_det=500,
                 theta=0.25, tau=0.5):
    """Use local assets only; each call owns a new output directory."""
    config = Config(theta=theta, tau=tau, class_ids=tuple(range(10)))
    if not 0 <= confidence_floor <= config.theta or not 0 <= nms_iou <= 1:
        raise ValueError("invalid confidence floor / NMS IoU")
    if type(imgsz) is not int or imgsz <= 0 or type(max_det) is not int or max_det <= 0:
        raise ValueError("imgsz and max_det must be positive integers")
    dataset_root, weights = Path(dataset_root).resolve(), Path(weights).resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise FileNotFoundError(f"local .pt checkpoint required: {weights}")
    data_yaml = dataset_root / "visdrone.yaml"
    split_root = dataset_root / "VisDrone2019-DET-val"
    selected, available_names = select_images(split_root / "images", split_root / "annotations",
                                              sample_count, seed)
    # No automatic installation, settings changes in the user's training setup, or downloads.
    runtime_dir = PROJECT_ROOT / "results" / "region_eval" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import ultralytics
    import yaml
    from PIL import Image
    from ultralytics import YOLO

    yaml_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    names = yaml_config.get("names")
    if isinstance(names, dict):
        names = [names.get(i) for i in range(10)] if set(names) == set(range(10)) else None
    if names != CLASS_NAMES:
        raise ValueError("dataset YAML class mapping does not match VisDrone 1..10 -> model 0..9")
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Environment: Python {platform.python_version()}, torch {torch.__version__}, "
          f"ultralytics {ultralytics.__version__}; device={device}", flush=True)
    print(f"Ultralytics source: {ultralytics.__file__}", flush=True)
    print(f"Checking {len(selected)} images and raw annotations...", flush=True)
    prepared, manifest = [], []
    for image_path in selected:
        annotation_path = split_root / "annotations" / (image_path.stem + ".txt")
        raw = annotation_path.read_bytes()
        with Image.open(image_path) as image:
            width, height = image.size
        rows = parse_annotations(raw.decode("utf-8-sig"))
        gts, ignore_rects, audit = prepare_ground_truth(rows, width, height)
        prepared.append(dict(path=image_path, width=width, height=height, rows=rows,
                             gts=gts, ignore_rects=ignore_rects, audit=audit))
        manifest.append(dict(image_id=image_path.stem, image_file=str(image_path),
                             image_sha256=_hash_file(image_path), width=width, height=height,
                             annotation_file=str(annotation_path), annotation_sha256=sha256(raw).hexdigest()))
    weight_hash = _hash_file(weights)
    print(f"Loading existing checkpoint: {weights}", flush=True)
    model = YOLO(str(weights), task="detect")
    model_names = model.names
    if isinstance(model_names, dict):
        model_names = [model_names.get(i) for i in range(10)] if set(model_names) == set(range(10)) else None
    if model.task != "detect" or model_names != CLASS_NAMES:
        raise ValueError("checkpoint task/class names differ from the dataset; no implicit remapping")
    package_root = Path(ultralytics.__file__).parent
    package_hashes = {str(p.relative_to(package_root)).replace("\\", "/"): _hash_file(p)
                      for p in sorted(package_root.rglob("*.py"))}
    metadata = dict(
        dataset="VisDrone2019-DET", split=f"val_diagnostic_subset_{sample_count}",
        model_id="YOLO11n" if "yolo11n" in str(weights).lower() else weights.parent.parent.name,
        checkpoint_id=f"{weights.name}:sha256:{weight_hash}",
        coordinate_frame="original_image_xyxy", ignore_crowd_policy="already_filtered",
        class_ids=list(range(10)), class_names=CLASS_NAMES,
        postprocessing=dict(stage="post_nms", confidence_floor=confidence_floor,
                            nms_iou=nms_iou, max_det=max_det, class_agnostic_nms=False,
                            imgsz=imgsz, batch=1, half=False, augment=False, rect=True),
        preprocessing=dict(protocol=IGNORE_POLICY, reference=IGNORE_REFERENCE,
                           ignore_region_category=0, ignore_overlap_threshold=0.5,
                           denominator="clipped_box_area", removal_comparison=">=",
                           apply_to="target GT and predictions, independently of class",
                           others_category_11="excluded from target GT, not a spatial ignore mask",
                           score0_target_gt="unsupported: fail explicitly",
                           remaining_boxes="retained whole, including residual ignore overlap",
                           geometry="continuous clipped xyxy; differs from official integer raster implementation",
                           official_ap_ar_equivalent=False),
        subset=dict(method="smallest SHA256(seed:filename), then filename order", seed=seed,
                    available_count=len(available_names), count=len(selected),
                    available_names_sha256=sha256(json.dumps(available_names).encode()).hexdigest(),
                    image_manifest=manifest),
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         ultralytics=ultralytics.__version__, ultralytics_path=str(package_root),
                         ultralytics_source_sha256=sha256(json.dumps(package_hashes, sort_keys=True).encode()).hexdigest(),
                         ultralytics_source_file_count=len(package_hashes), device=str(device)),
        weights_file=str(weights), weights_sha256=weight_hash,
        dataset_yaml_file=str(data_yaml), dataset_yaml_sha256=_hash_file(data_yaml),
        runner_sha256=_hash_file(PROJECT_ROOT / "run_region_visdrone.py"),
        purpose="small-sample regional diagnostics; not benchmark AP/AR or final VC")
    run_root = PROJECT_ROOT / "results" / "region_eval" / ("visdrone_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run_root.mkdir(parents=True, exist_ok=False)
    _save_json(run_root / "run_manifest.json", dict(status="started", metadata=metadata))
    raw_images, filtered_images, diagnostics, audit_rows = [], [], [], []
    start = perf_counter()
    try:
        for index, item in enumerate(prepared, 1):
            path, width, height = item["path"], item["width"], item["height"]
            results = model.predict(source=str(path), imgsz=imgsz, conf=confidence_floor,
                                    iou=nms_iou, max_det=max_det, device=device, batch=1,
                                    half=False, augment=False, agnostic_nms=False, rect=True,
                                    classes=None, save=False, save_txt=False, save_conf=False,
                                    save_crop=False, show=False, verbose=False, stream=False,
                                    project=str(run_root), name="inference", exist_ok=True)
            if len(results) != 1 or tuple(results[0].orig_shape) != (height, width):
                raise ValueError(f"unexpected prediction image shape/count for {path.name}")
            result = results[0]
            if Path(result.path).resolve() != path.resolve():
                raise ValueError("prediction source path differs from selected image")
            tensor = result.boxes
            coords = tensor.xyxy.detach().cpu().tolist()
            scores = tensor.conf.detach().cpu().tolist()
            classes = tensor.cls.detach().cpu().tolist()
            if len(coords) > max_det or not all(float(c).is_integer() for c in classes):
                raise ValueError("invalid model prediction count/class")
            predictions = [dict(id=f"pred_{j:06d}", xyxy=rect, score=score, class_id=int(cls))
                           for j, (rect, score, cls) in enumerate(zip(coords, scores, classes))]
            kept, pred_audit = filter_predictions(predictions, item["ignore_rects"], width, height)
            audit = dict(image_id=path.stem, **item["audit"], **pred_audit,
                         max_det_reached=len(predictions) == max_det)
            record = dict(image_id=path.stem, width=width, height=height,
                          gts=item["gts"], predictions=kept)
            _, measured, _ = evaluate_document(dict(schema_version=1, metadata=metadata, images=[record]),
                                                theta=theta, tau=tau)
            diagnostics.extend(measured)
            audit_rows.append(audit)
            filtered_images.append(record)
            raw_images.append(dict(image_id=path.stem, width=width, height=height,
                                   annotations=item["rows"], predictions=predictions))
            print(f"[{index}/{len(prepared)}] {path.name}: GT={len(item['gts'])}, "
                  f"predictions={len(predictions)} -> {len(kept)} after ignore filtering; "
                  f"elapsed={perf_counter() - start:.1f}s", flush=True)
        # Preserve predictions before writing the report, so geometry can be recomputed without inference.
        _save_json(run_root / "raw_predictions.json", dict(raw_cache_version=1, metadata=metadata, images=raw_images))
        metadata["preprocessing_audit"] = audit_rows
        metadata["max_det_reached_images"] = sum(row["max_det_reached"] for row in audit_rows)
        document = dict(schema_version=1, metadata=metadata, images=filtered_images)
        source = run_root / "filtered_predictions.json"
        source_bytes = _save_json(source, document)
        _save_json(run_root / "preprocessing_audit.json", audit_rows)
        summary = aggregate(diagnostics, config=config)
        output = write_report(run_root / "report", source=source, source_bytes=source_bytes,
                              document=document, config=config, images=diagnostics, summary=summary)
        _save_json(run_root / "run_complete.json", dict(status="complete", report=str(output),
                   elapsed_seconds=perf_counter() - start, summary=summary,
                   max_det_reached_images=metadata["max_det_reached_images"]))
    except Exception as exc:
        _save_json(run_root / "run_failed.json", dict(status="failed", error=str(exc),
                   completed_images=len(filtered_images)))
        raise
    print(json.dumps(dict(output=str(output), **summary), ensure_ascii=False), flush=True)
    print(f"SUCCESS: evaluated {len(prepared)} real images. No training performed.", flush=True)
    print(f"Prediction caches and preprocessing audit: {run_root}", flush=True)
    if metadata["max_det_reached_images"]:
        print(f"NOTE: {metadata['max_det_reached_images']} images reached max_det={max_det}; "
              "the candidate cap is recorded for interpreting these diagnostics.", flush=True)
    return run_root
