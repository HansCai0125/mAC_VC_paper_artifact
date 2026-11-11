"""Strict saved-prediction input and isolated CSV reports (no inference)."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from math import fsum
from pathlib import Path
import platform
import sys

from .diagnostics import (Box, Config, METRIC_VERSION, _class_id, _identifier,
                          _number, aggregate, image_region_diagnostics)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_FIELDS = ["run_id", "image_id", "gt_id", "class_id", "theta", "gt_area", "AC",
             "BestIoGT_i", "CollectiveGain_i", "coverage_hit",
             "intersecting_pred_count", "best_pred_id"]
CLASS_FIELDS = ["run_id", "image_id", "class_id", "theta", "n_gt", "n_pred",
                "gt_union_area", "pred_union_area", "intersection_area", "background_area",
                "SpatialPrecision", "ClassUnionDice", "ClassUnionIoU", "active_pair",
                "orphan_pred_count", "orphan_union_area", "multigt_pred_count"]


def _records(records, *, prediction):
    if not isinstance(records, list):
        raise ValueError("gts and predictions must be arrays")
    boxes = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each box must be an object")
        required = {"id", "xyxy", "class_id"} | ({"score"} if prediction else set())
        if set(record) != required:
            raise ValueError(f"box fields must be exactly {sorted(required)}; "
                             "ignore/crowd and other annotations must be handled upstream")
        boxes.append(Box(record["id"], record["xyxy"], record["class_id"],
                         record.get("score")))
    return boxes


def evaluate_document(document, *, theta=0.25, tau=0.5, cg_epsilon=1e-9):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("input must have schema_version=1")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    for name in ("dataset", "split", "model_id", "checkpoint_id", "ignore_crowd_policy"):
        _identifier(metadata.get(name), f"metadata.{name}")
    if metadata.get("coordinate_frame") != "original_image_xyxy":
        raise ValueError("coordinate_frame must be original_image_xyxy; no implicit scaling")
    if metadata["ignore_crowd_policy"] != "already_filtered":
        raise ValueError("only explicit upstream ignore/crowd preprocessing is supported")
    class_ids = metadata.get("class_ids")
    if not isinstance(class_ids, list) or not class_ids:
        raise ValueError("metadata.class_ids must be a nonempty list")
    config = Config(theta, tau, cg_epsilon, tuple(_class_id(c) for c in class_ids))
    processing = metadata.get("postprocessing")
    allowed_stages = ("post_nms", "native_rtdetr_postprocess_no_nms", "synthetic")
    if not isinstance(processing, dict) or processing.get("stage") not in allowed_stages:
        raise ValueError(
            "postprocessing.stage must be post_nms, "
            "native_rtdetr_postprocess_no_nms, or synthetic"
        )
    floor = _number(processing.get("confidence_floor"), "confidence_floor")
    if not 0 <= floor <= config.theta:
        raise ValueError("cannot evaluate below the exported prediction confidence floor")
    if processing.get("nms_iou") is not None:
        nms_iou = _number(processing["nms_iou"], "nms_iou")
        if not 0 <= nms_iou <= 1:
            raise ValueError("nms_iou must be in [0, 1] or null")
    if "nms_iou" not in processing:
        raise ValueError("postprocessing must record nms_iou (null if no NMS)")
    max_det = processing.get("max_det")
    if isinstance(max_det, bool) or not isinstance(max_det, int) or max_det <= 0:
        raise ValueError("postprocessing.max_det must be a positive integer")
    raw_images = document.get("images")
    if not isinstance(raw_images, list):
        raise ValueError("images must be an array")
    images = []
    for record in raw_images:
        if not isinstance(record, dict) or set(record) != {"image_id", "width", "height", "gts", "predictions"}:
            raise ValueError("image fields must be image_id, width, height, gts, predictions")
        gts = _records(record["gts"], prediction=False)
        preds = _records(record["predictions"], prediction=True)
        if len(preds) > max_det:
            raise ValueError(f"{record['image_id']}: predictions exceed declared max_det")
        for pred in preds:
            if _number(pred.score, "prediction score") < floor:
                raise ValueError("prediction score below declared export confidence floor")
        images.append(image_region_diagnostics(gts, preds, (record["width"], record["height"]),
                                               image_id=record["image_id"], config=config))
    summary = aggregate(images, config=config)
    return config, sorted(images, key=lambda d: d.image_id), summary


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _csv(path, fields, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if row.get(key) is None else row[key] for key in fields})


def write_report(output, *, source, source_bytes, document, config, images, summary):
    output = Path(output).resolve()
    archive = (PROJECT_ROOT / "bin" / "old").resolve()
    if output == archive or output.is_relative_to(archive):
        raise ValueError("reports must not be written into bin/old")
    if output.exists():
        raise ValueError(f"output directory already exists; choose a fresh run directory: {output}")
    run_id = output.name
    metadata = document["metadata"]
    common = dict(run_id=run_id, **{k: metadata[k] for k in ("dataset", "split", "model_id", "checkpoint_id")})
    summary_row = dict(common, **summary)
    gt_rows = [dict(run_id=run_id, **row) for d in images for row in d.per_gt]
    pair_rows = [dict(run_id=run_id, **row) for d in images for row in d.per_class]
    manifest = [dict(image_id=d.image_id, width=d.width, height=d.height, **d.counts) for d in images]
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source_files = sorted((PROJECT_ROOT / "region_eval").glob("*.py"))
    legacy = PROJECT_ROOT / "mean_AC.py"
    if legacy.exists():
        source_files.append(legacy)
    hashes = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha(p.read_bytes()) for p in source_files}
    by_class = []
    for cls in sorted({row["class_id"] for row in pair_rows}):
        rows = [r for r in pair_rows if r["class_id"] == cls]
        by_class.append(dict(class_id=cls, n_active_images=len(rows),
                             ClassUnionDice=fsum(r["ClassUnionDice"] for r in rows) / len(rows),
                             ClassUnionIoU=fsum(r["ClassUnionIoU"] for r in rows) / len(rows)))
    report_metadata = dict(
        status="complete", metric_version=METRIC_VERSION, run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(), source_file=str(Path(source).resolve()),
        input_sha256=_sha(source_bytes), image_manifest_sha256=_sha(manifest_bytes),
        code_sha256=hashes, source_metadata=metadata, configuration=asdict(config),
        clipping_policy="clip_to_image; invalid_gt_error; invalid_pred_skip_and_count; reversed_error",
        geometry="continuous_xyxy_float64_exact_rectangle_union",
        numeric_tolerance="synthetic tests: atol=rtol=1e-9; no denominator epsilon",
        quantile_method="linear interpolation at (n-1)*q", python_version=platform.python_version(),
        image_manifest=manifest, per_class_means=by_class,
        limitations=["No inference, AP, VC, EPI, assignment, or native dataset ignore-region adapter.",
                     "Input predictions and GT must already share the original-image coordinate frame.",
                     "The caller is responsible for upstream ignore/crowd preprocessing.",
                     "Class union cannot distinguish instance merges with identical support; area can mask small misses."])
    # Finish evaluation and metadata serialization before creating any output.
    metadata_text = json.dumps(report_metadata, ensure_ascii=False, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    _csv(output / "region_summary.csv", list(summary_row), [summary_row])
    _csv(output / "region_gt.csv", GT_FIELDS, gt_rows)
    _csv(output / "region_image_class.csv", CLASS_FIELDS, pair_rows)
    # Written last; absent metadata means the output is incomplete.
    (output / "metadata.json").write_text(metadata_text + "\n", encoding="utf-8")
    return output


def _reject_constant(value):
    raise ValueError(f"nonstandard JSON number {value}; use finite numeric coordinates")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate saved boxes with five region diagnostics; no inference.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="A new, nonexistent directory; bin/old is forbidden")
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--cg-epsilon", type=float, default=1e-9)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
        config, images, summary = evaluate_document(document, theta=args.theta, tau=args.tau,
                                                    cg_epsilon=args.cg_epsilon)
        output = args.output or PROJECT_ROOT / "outputs" / "region_eval" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = write_report(output, source=args.input, source_bytes=raw, document=document,
                              config=config, images=images, summary=summary)
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"region_eval: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dict(output=str(output), **summary), ensure_ascii=False, allow_nan=False))
    return 0
