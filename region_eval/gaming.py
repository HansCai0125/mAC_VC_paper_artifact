"""Offline controlled gaming experiments for the frozen VC v1 definition.

The experiment consumes completed post-NMS prediction caches. It never loads a
detector checkpoint and never performs training or inference.
"""

from copy import deepcopy
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

from .matching_analysis import ap_metrics, match_class
from .vc_v1 import MATCHING_IOU, PRIMARY_THETA, VC_VERSION, evaluate_vc


GAMING_VERSION = "vc-v1-gaming-0.2.1"
JITTER_CENTER_FRACTION = 0.02
JITTER_SIZE_FRACTION = 0.01


def _read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON number: {value}")),
    )


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty gaming table")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cache_metadata(path, chunk_size=65536, maximum_bytes=2 * 1024 * 1024):
    """Read only the leading metadata object from a large cache JSON file."""
    path = Path(path)
    decoder = json.JSONDecoder(
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON number: {value}")))
    text = ""
    with path.open("r", encoding="utf-8-sig") as handle:
        while len(text) < maximum_bytes:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            text += chunk
            marker = text.find('"metadata"')
            if marker < 0:
                continue
            colon = text.find(":", marker + len('"metadata"'))
            if colon < 0:
                continue
            start = colon + 1
            while start < len(text) and text[start].isspace():
                start += 1
            try:
                value, _ = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                raise ValueError("cache metadata must be an object")
            return value
    raise ValueError(
        f"could not read cache metadata within {maximum_bytes} bytes: {path}")


def _evenly_spaced(items, count):
    if type(count) is not int or count <= 0:
        raise ValueError("sample count must be a positive integer")
    if count > len(items):
        raise ValueError(
            f"requested {count} images but only {len(items)} shards exist")
    if count == 1:
        return [items[0]]
    indexes = [
        round(index * (len(items) - 1) / (count - 1))
        for index in range(count)
    ]
    if len(set(indexes)) != count:
        raise AssertionError(
            "evenly spaced shard selection produced duplicate indexes")
    return [items[index] for index in indexes]


def load_shard_sample(
        run_directory, sample_count, *, minimum_score=None):
    """Build a canonical document from evenly spaced prediction shards."""
    if (minimum_score is not None
            and (not math.isfinite(minimum_score)
                 or not 0.0 <= minimum_score <= 1.0)):
        raise ValueError("minimum_score must be finite and in [0, 1]")
    run_directory = Path(run_directory).resolve()
    completion = _read_json(run_directory / "run_complete.json")
    cache_completion = _read_json(run_directory / "cache_complete.json")
    if completion.get("status") != "complete":
        raise ValueError(f"source run is not complete: {run_directory}")
    if (cache_completion.get("status") != "complete"
            or cache_completion.get("run_fingerprint")
            != completion.get("run_fingerprint")):
        raise ValueError(f"source cache is not complete: {run_directory}")
    cache_path = run_directory / "filtered_predictions.json"
    metadata = read_cache_metadata(cache_path)
    shard_paths = sorted((run_directory / "shards").glob("*.json"))
    if cache_completion.get("n_images") != len(shard_paths):
        raise ValueError("cache image count does not match prediction shards")
    selected = _evenly_spaced(shard_paths, sample_count)
    images = []
    for path in selected:
        shard = _read_json(path)
        if shard.get("run_fingerprint") != completion.get("run_fingerprint"):
            raise ValueError(f"shard fingerprint mismatch: {path}")
        image = shard.get("filtered_image")
        if not isinstance(image, dict):
            raise ValueError(f"filtered image is missing: {path}")
        if minimum_score is not None:
            image = dict(image)
            image["predictions"] = [
                prediction for prediction in image["predictions"]
                if prediction["score"] >= minimum_score
            ]
        images.append(image)
    if minimum_score is not None:
        metadata = deepcopy(metadata)
        source_floor = metadata["postprocessing"]["confidence_floor"]
        metadata["postprocessing"]["confidence_floor"] = max(
            source_floor, minimum_score)
    document = dict(schema_version=1, metadata=metadata, images=images)
    source = dict(
        run_directory=str(run_directory),
        run_fingerprint=completion["run_fingerprint"],
        cache_file=str(cache_path),
        cache_sha256=cache_completion["filtered_predictions_sha256"],
        available_images=len(shard_paths),
        selected_images=len(images),
        selection="evenly spaced over deterministic shard order",
        analysis_minimum_score=minimum_score,
        selected_shards=[path.name for path in selected],
    )
    return document, source


def baseline_tp_ids(
        document, theta=PRIMARY_THETA, iou_threshold=MATCHING_IOU):
    """Return predictions that are TPs before any manipulation."""
    class_ids = sorted({
        row["class_id"]
        for image in document["images"]
        for kind in ("gts", "predictions")
        for row in image[kind]
    })
    result = set()
    for class_id in class_ids:
        events, _ = match_class(
            document, class_id, iou_threshold, theta)
        result.update(
            (event["image_id"], event["pred_id"])
            for event in events if event["is_tp"]
        )
    return result


def scale_xyxy(xyxy, scale, width, height):
    """Scale a box around its centre and clip it to its image."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("box scale must be finite and positive")
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = (x2 - x1) * scale / 2.0
    half_h = (y2 - y1) * scale / 2.0
    result = [
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(float(width), cx + half_w),
        min(float(height), cy + half_h),
    ]
    if not result[0] < result[2] or not result[1] < result[3]:
        raise ValueError("scaled prediction became degenerate")
    return result


def jitter_xyxy(
        xyxy, copy_index, width, height,
        center_fraction=JITTER_CENTER_FRACTION,
        size_fraction=JITTER_SIZE_FRACTION):
    """Apply deterministic centre and size jitter to one duplicate."""
    if type(copy_index) is not int or copy_index < 1:
        raise ValueError("copy_index must be a positive integer")
    if not 0.0 <= center_fraction < 1.0:
        raise ValueError("center_fraction must be in [0, 1)")
    if not 0.0 <= size_fraction < 1.0:
        raise ValueError("size_fraction must be in [0, 1)")
    patterns = (
        (1, 0, 1, -1), (-1, 0, -1, 1),
        (0, 1, 1, 1), (0, -1, -1, -1),
        (1, 1, 1, 0), (-1, 1, 0, 1),
        (1, -1, -1, 0), (-1, -1, 0, -1),
    )
    dx_sign, dy_sign, dw_sign, dh_sign = patterns[
        (copy_index - 1) % len(patterns)]
    x1, y1, x2, y2 = xyxy
    box_w, box_h = x2 - x1, y2 - y1
    centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    target_w = min(float(width), box_w * (1.0 + dw_sign * size_fraction))
    target_h = min(float(height), box_h * (1.0 + dh_sign * size_fraction))
    shifted_centre_x = centre_x + dx_sign * center_fraction * box_w
    shifted_centre_y = centre_y + dy_sign * center_fraction * box_h
    shifted_x1 = min(
        max(0.0, shifted_centre_x - target_w / 2.0),
        max(0.0, float(width) - target_w),
    )
    shifted_y1 = min(
        max(0.0, shifted_centre_y - target_h / 2.0),
        max(0.0, float(height) - target_h),
    )
    return [
        shifted_x1, shifted_y1,
        shifted_x1 + target_w, shifted_y1 + target_h,
    ]


def manipulate_scale(document, target_ids, scale):
    """Scale baseline-TP predictions; preserve every other prediction."""
    result = deepcopy(document)
    modified = 0
    for image in result["images"]:
        for prediction in image["predictions"]:
            if (image["image_id"], prediction["id"]) in target_ids:
                prediction["xyxy"] = scale_xyxy(
                    prediction["xyxy"], scale,
                    image["width"], image["height"])
                modified += 1
    return result, dict(
        modified_prediction_count=modified,
        added_prediction_count=0,
    )


def manipulate_duplicates(document, target_ids, factor, *, jitter=False):
    """Duplicate baseline TPs exactly or with deterministic centre/size jitter."""
    if type(factor) is not int or factor < 1:
        raise ValueError(
            "duplication factor must be an integer of at least one")
    result = deepcopy(document)
    added = 0
    for image in result["images"]:
        additions = []
        for prediction in image["predictions"]:
            if (image["image_id"], prediction["id"]) not in target_ids:
                continue
            for copy_index in range(1, factor):
                clone = deepcopy(prediction)
                clone["id"] = (
                    f"{prediction['id']}__gaming_dup_{copy_index:02d}")
                if jitter:
                    clone["xyxy"] = jitter_xyxy(
                        prediction["xyxy"], copy_index,
                        image["width"], image["height"])
                additions.append(clone)
                added += 1
        image["predictions"].extend(additions)
    maximum = max(
        (len(image["predictions"]) for image in result["images"]),
        default=1,
    )
    result["metadata"]["postprocessing"]["max_det"] = max(
        result["metadata"]["postprocessing"]["max_det"], maximum)
    return result, dict(
        modified_prediction_count=len(target_ids),
        added_prediction_count=added,
    )


def _thresholded_for_ap(document, theta):
    """Use the same external confidence cutoff for AP and other metrics."""
    source_floor = document["metadata"]["postprocessing"]["confidence_floor"]
    if source_floor >= theta:
        return document
    result = deepcopy(document)
    for image in result["images"]:
        image["predictions"] = [
            prediction for prediction in image["predictions"]
            if prediction["score"] >= theta
        ]
    result["metadata"]["postprocessing"]["confidence_floor"] = theta
    return result


def score_document(
        document, theta=PRIMARY_THETA, *, include_diagnostic_ap=True):
    """Score revised regional and standard metrics on one prediction set."""
    summary, _, _ = evaluate_vc(document, theta=theta, tau=0.50)
    ap_summary = None
    if include_diagnostic_ap:
        ap_summary, _ = ap_metrics(
            _thresholded_for_ap(document, theta),
            document["metadata"]["class_names"],
        )
    return dict(
        theta=theta,
        n_images=summary["n_images"],
        n_gt=summary["n_gt"],
        n_pred=summary["n_pred"],
        assigned_pred_count=summary["assigned_pred_count"],
        orphan_pred_count=summary["orphan_pred_count"],
        orphan_rate=summary["orphan_rate"],
        orphan_per_gt=(
            summary["orphan_pred_count"] / summary["n_gt"]
            if summary["n_gt"] else None),
        mAC=summary["mAC"],
        VC=summary["VC"],
        F1=summary["standard_F1"],
        LRP=summary["standard_LRP"],
        AP50=(ap_summary["AP50"] if ap_summary else None),
        AP75=(ap_summary["AP75"] if ap_summary else None),
        AP50_95=(ap_summary["AP50_95"] if ap_summary else None),
        AR50_95=(ap_summary["AR50_95"] if ap_summary else None),
    )


def _normalizations(row, baseline):
    for field in ("mAC", "VC", "F1", "AP50", "AP50_95"):
        denominator = baseline[field]
        numerator = row[field]
        row[f"normalized_{field}"] = (
            numerator / denominator
            if numerator is not None
            and denominator not in (None, 0.0) else None)
    baseline_quality = (
        None if baseline["LRP"] is None else 1.0 - baseline["LRP"])
    quality = None if row["LRP"] is None else 1.0 - row["LRP"]
    row["normalized_one_minus_LRP"] = (
        quality / baseline_quality
        if baseline_quality not in (None, 0.0) else None)
    row["normalized_n_pred"] = (
        row["n_pred"] / baseline["n_pred"]
        if baseline["n_pred"] else None)
    row["normalized_orphan_count"] = (
        row["orphan_pred_count"] / baseline["orphan_pred_count"]
        if baseline["orphan_pred_count"] else None)
    return row


def run_gaming_grid(
        document, *, shrink_scales, expand_scales, duplicate_factors,
        threshold_grid, primary_theta=PRIMARY_THETA,
        diagnostic_ap_levels=None):
    """Run four controlled families on one canonical cached document."""
    include_all_diagnostic_ap = diagnostic_ap_levels is None
    diagnostic_ap_levels = set(diagnostic_ap_levels or ())

    def score(value, family, variant, level, theta=primary_theta):
        return score_document(
            value, theta,
            include_diagnostic_ap=(
                include_all_diagnostic_ap
                or (family, variant, level) in diagnostic_ap_levels
            ),
        )

    target_ids = baseline_tp_ids(
        document, primary_theta, MATCHING_IOU)
    baseline = score(
        document, "baseline", "unmodified", 1.0)
    rows = []

    def append(
            manipulation, variant, level, label, metrics, audit):
        row = dict(
            manipulation=manipulation,
            variant=variant,
            level=level,
            level_label=label,
            primary_theta=primary_theta,
            baseline_tp_count=len(target_ids),
            **audit,
            **metrics,
        )
        rows.append(_normalizations(row, baseline))

    baseline_audit = dict(
        modified_prediction_count=0,
        added_prediction_count=0,
    )
    for family, scales in (
            ("shrink", shrink_scales), ("expand", expand_scales)):
        for scale in scales:
            if scale == 1.0:
                metrics, audit = dict(baseline), dict(baseline_audit)
            else:
                manipulated, audit = manipulate_scale(
                    document, target_ids, scale)
                metrics = score(
                    manipulated, family, "center_scale", scale)
            append(
                family, "center_scale", scale,
                f"side scale {scale:g}", metrics, audit)

    for jitter in (False, True):
        variant = (
            "controlled_jitter_2pct_center_1pct_size"
            if jitter else "exact")
        for factor in duplicate_factors:
            if factor == 1:
                metrics, audit = dict(baseline), dict(baseline_audit)
            else:
                manipulated, audit = manipulate_duplicates(
                    document, target_ids, factor, jitter=jitter)
                metrics = score(
                    manipulated, "duplicate", variant, factor)
            append(
                "duplicate", variant, factor,
                f"total multiplicity {factor}", metrics, audit)

    for theta in threshold_grid:
        metrics = (
            baseline if theta == primary_theta
            else score(
                document, "threshold", "confidence_cutoff", theta,
                theta=theta))
        append(
            "threshold", "confidence_cutoff", theta,
            f"theta {theta:.2f}", dict(metrics), dict(baseline_audit))
    return rows, dict(
        baseline_tp_count=len(target_ids),
        baseline_metrics=baseline,
        target_rule=(
            f"baseline one-to-one true positives at theta={primary_theta} "
            f"and IoU={MATCHING_IOU}"),
    )


def _plot_rows(output, rows, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to write gaming figures") from exc

    panels = (
        ("shrink", "center_scale", "Shrink", "Side-length scale"),
        ("expand", "center_scale", "Expand", "Side-length scale"),
        ("duplicate", "exact", "Exact duplicate", "Total multiplicity"),
        (
            "duplicate", "controlled_jitter_2pct_center_1pct_size",
            "Controlled-jitter duplicate", "Total multiplicity"),
        ("threshold", "confidence_cutoff", "Confidence threshold", "Theta"),
    )
    panels = tuple(
        panel for panel in panels
        if any(
            row["manipulation"] == panel[0]
            and row["variant"] == panel[1]
            for row in rows
        )
    )
    fields = (
        ("normalized_VC", "VC"),
        ("normalized_mAC", "mAC"),
        ("normalized_F1", "F1"),
        ("normalized_one_minus_LRP", "1-LRP"),
        ("normalized_AP50_95", "AP50:95"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = list(axes.flat)
    for axis, (
            family, variant, panel_title, xlabel) in zip(axes, panels):
        selected = [
            row for row in rows
            if row["manipulation"] == family
            and row["variant"] == variant
        ]
        selected.sort(key=lambda row: row["level"])
        for field, label in fields:
            axis.plot(
                [row["level"] for row in selected],
                [row[field] for row in selected],
                marker="o", linewidth=1.7, label=label)
        axis.axhline(
            1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(panel_title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Normalized score")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    for axis in axes[len(panels):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(
        Path(output) / "gaming_normalized_quality.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = list(axes.flat)
    for axis, (
            family, variant, panel_title, xlabel) in zip(axes, panels):
        selected = [
            row for row in rows
            if row["manipulation"] == family
            and row["variant"] == variant
        ]
        selected.sort(key=lambda row: row["level"])
        axis.plot(
            [row["level"] for row in selected],
            [row["normalized_orphan_count"] for row in selected],
            marker="o", label="Orphan count")
        axis.plot(
            [row["level"] for row in selected],
            [row["normalized_n_pred"] for row in selected],
            marker="o", label="Prediction count")
        axis.axhline(
            1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(panel_title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Ratio to baseline")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for axis in axes[len(panels):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(
        Path(output) / "gaming_diagnostics.png", dpi=180)
    plt.close(figure)


def write_gaming_report(
        output, *, document, source, rows, audit, protocol):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(
            f"gaming output already exists: {output}")
    output.mkdir(parents=True)
    metadata = document["metadata"]
    common = dict(
        dataset=metadata["dataset"],
        model_id=metadata["model_id"],
        source_run_fingerprint=source["run_fingerprint"],
    )
    complete_rows = [
        dict(common, **row) for row in rows]
    _write_csv(
        output / "gaming_metrics.csv", complete_rows)
    _plot_rows(
        output, complete_rows,
        f"{metadata['dataset']} / {metadata['model_id']} gaming analysis")
    manifest = dict(
        status="complete",
        gaming_version=GAMING_VERSION,
        vc_version=VC_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset=metadata["dataset"],
        model_id=metadata["model_id"],
        source=source,
        protocol=protocol,
        audit=audit,
        result_files={},
    )
    purpose = protocol.get("purpose", "controlled gaming analysis")
    ap_note = (
        "Diagnostic AP fields are populated only at the protocol's "
        "declared levels. Blank AP cells were intentionally not recomputed."
        if protocol.get("diagnostic_ap_levels") else
        "Diagnostic AP was not recomputed in this report."
    )
    readme = f"""# VC v1 controlled gaming result

- Dataset: `{metadata['dataset']}`
- Model: `{metadata['model_id']}`
- Images: `{len(document['images'])}` cached images
- Purpose: `{purpose}`
- Baseline target: `{audit['target_rule']}`
- Baseline matched predictions manipulated: `{audit['baseline_tp_count']}`
- Training: none
- Model inference: none

`gaming_metrics.csv` contains raw and baseline-normalized VC, mAC, F1,
LRP, orphan count, and prediction count. {ap_note}
"""
    (output / "README.md").write_text(
        readme, encoding="utf-8")
    for name in (
            "gaming_metrics.csv",
            "gaming_normalized_quality.png",
            "gaming_diagnostics.png",
            "README.md"):
        manifest["result_files"][name] = _file_hash(
            output / name)
    _write_json(
        output / "gaming_complete.json", manifest)
    return output
