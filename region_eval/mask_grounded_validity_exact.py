"""Exact continuous-polygon validation of box-level coverage on MS COCO.

The module reuses 15 saved MS COCO prediction caches. COCO segmentation
polygons are continuous regions, so pixel rasterization and boundary rounding
are not part of the quantitative computation. No training or inference occurs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
from pathlib import Path
import statistics

import numpy as np

from .cache_registry import MODEL_ORDER, YOLO_MODELS, CachedRun, paired_yolo_sources
from .mask_grounded_validity import (
    AC_BINS, AC_THRESHOLD, ANNOTATION_FILE, GAIN_BINS, GAIN_PLOT_MAX,
    GAIN_THRESHOLD, IMAGE_ROOT, MODEL_LABELS, PolygonMaskCache,
    _annotation_id, _fraction, _latex_table, _load_coco, _pearson,
    _plot_cases, _plot_relationships, _plot_validity_rates, _sha256, _spearman,
)
from .natural_collective_gain import (
    ANALYSIS_VERSION as NATURAL_ANALYSIS_VERSION,
    DEFAULT_OUTPUT as NATURAL_OUTPUT,
    _aggregate_seed_rows, _atomic_csv, _atomic_json, _atomic_text,
    _read_csv, _read_json,
)
from .offline import PROJECT_ROOT
from .vc_v1 import PRIMARY_THETA, VC_VERSION


EXACT_VERSION = "vc-v1-mask-grounded-exact-0.2.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "mask_polygon"
OLD_OUTPUT = PROJECT_ROOT / "results" / "mask_raster"
PAPER_GAIN_THRESHOLDS = (0.02, 0.05, 0.10)
GAIN_THRESHOLD_GRID = tuple(round(index / 100.0, 2) for index in range(1, 21))
GEOMETRY_EPSILON = 1e-10
AREA_AUDIT_TOLERANCE = 1e-8
SIZE_ORDER = ("small", "medium", "large")


@dataclass(frozen=True)
class PolygonPart:
    vertices: np.ndarray
    bbox: tuple[float, float, float, float]
    area: float


@dataclass(frozen=True)
class PolygonGeometry:
    annotation_id: int
    parts: tuple[PolygonPart, ...]
    bbox: tuple[float, float, float, float]
    area: float
    coco_area: float


class PolygonGeometryCache:
    def __init__(self, annotations: dict[int, dict]):
        self.annotations = annotations
        self._cache: dict[int, PolygonGeometry] = {}

    def get(self, annotation_id: int) -> PolygonGeometry:
        geometry = self._cache.get(annotation_id)
        if geometry is None:
            geometry = geometry_from_annotation(self.annotations[annotation_id])
            self._cache[annotation_id] = geometry
        return geometry

    @property
    def size(self) -> int:
        return len(self._cache)


def _signed_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clean_vertices(flat_coordinates) -> np.ndarray:
    vertices = np.asarray(flat_coordinates, dtype=np.float64).reshape(-1, 2)
    cleaned = []
    for point in vertices:
        if not cleaned or float(np.linalg.norm(point - cleaned[-1])) > GEOMETRY_EPSILON:
            cleaned.append(point)
    if len(cleaned) >= 2 and float(np.linalg.norm(cleaned[0] - cleaned[-1])) <= GEOMETRY_EPSILON:
        cleaned.pop()
    if len(cleaned) < 3:
        raise ValueError("polygon contains fewer than three distinct vertices")
    return np.asarray(cleaned, dtype=np.float64)


def geometry_from_annotation(annotation: dict) -> PolygonGeometry:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError(f"annotation {annotation.get('id')} is not polygon encoded")
    parts = []
    for flat in segmentation:
        if not isinstance(flat, list) or len(flat) < 6 or len(flat) % 2:
            raise ValueError(f"invalid polygon for annotation {annotation.get('id')}")
        vertices = _clean_vertices(flat)
        part_area = abs(_signed_area(vertices))
        if part_area <= GEOMETRY_EPSILON:
            raise ValueError(f"zero-area polygon for annotation {annotation.get('id')}")
        parts.append(PolygonPart(
            vertices=vertices,
            bbox=(float(np.min(vertices[:, 0])), float(np.min(vertices[:, 1])),
                  float(np.max(vertices[:, 0])), float(np.max(vertices[:, 1]))),
            area=part_area,
        ))
    area = float(sum(part.area for part in parts))
    bbox = (min(part.bbox[0] for part in parts), min(part.bbox[1] for part in parts),
            max(part.bbox[2] for part in parts), max(part.bbox[3] for part in parts))
    return PolygonGeometry(
        annotation_id=int(annotation["id"]), parts=tuple(parts), bbox=bbox,
        area=area, coco_area=float(annotation["area"]),
    )


def _rect_intersection(first, second):
    rectangle = (max(float(first[0]), float(second[0])),
                 max(float(first[1]), float(second[1])),
                 min(float(first[2]), float(second[2])),
                 min(float(first[3]), float(second[3])))
    if rectangle[2] - rectangle[0] <= GEOMETRY_EPSILON:
        return None
    if rectangle[3] - rectangle[1] <= GEOMETRY_EPSILON:
        return None
    return rectangle


def _clip_half_plane(vertices, *, axis, boundary, keep_greater):
    if len(vertices) == 0:
        return np.empty((0, 2), dtype=np.float64)

    def inside(point):
        if keep_greater:
            return float(point[axis]) >= boundary - GEOMETRY_EPSILON
        return float(point[axis]) <= boundary + GEOMETRY_EPSILON

    def crossing(start, end):
        denominator = float(end[axis] - start[axis])
        if abs(denominator) <= GEOMETRY_EPSILON:
            point = start.copy()
            point[axis] = boundary
            return point
        ratio = (boundary - float(start[axis])) / denominator
        point = start + min(1.0, max(0.0, ratio)) * (end - start)
        point[axis] = boundary
        return point

    output = []
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(crossing(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(crossing(previous, current))
        previous, previous_inside = current, current_inside
    return np.asarray(output, dtype=np.float64).reshape(-1, 2) if output else np.empty((0, 2), dtype=np.float64)


def clip_polygon_to_rectangle(vertices, rectangle):
    x1, y1, x2, y2 = rectangle
    output = np.asarray(vertices, dtype=np.float64)
    for axis, boundary, keep_greater in (
        (0, x1, True), (0, x2, False), (1, y1, True), (1, y2, False)
    ):
        output = _clip_half_plane(
            output, axis=axis, boundary=boundary, keep_greater=keep_greater
        )
    return output


def _part_rectangle_area(part, rectangle):
    if _rect_intersection(part.bbox, rectangle) is None:
        return 0.0
    x1, y1, x2, y2 = rectangle
    bx1, by1, bx2, by2 = part.bbox
    if (x1 <= bx1 + GEOMETRY_EPSILON and y1 <= by1 + GEOMETRY_EPSILON
            and x2 >= bx2 - GEOMETRY_EPSILON and y2 >= by2 - GEOMETRY_EPSILON):
        return part.area
    return abs(_signed_area(clip_polygon_to_rectangle(part.vertices, rectangle)))


def polygon_rectangle_area(geometry, rectangle):
    clipped_rectangle = _rect_intersection(geometry.bbox, rectangle)
    if clipped_rectangle is None:
        return 0.0
    value = sum(_part_rectangle_area(part, clipped_rectangle) for part in geometry.parts)
    if value > geometry.area * (1.0 + 1e-8):
        raise AssertionError(f"polygon-rectangle area exceeds target: {value}/{geometry.area}")
    return min(geometry.area, max(0.0, float(value)))


def rectangle_union_cells(rectangles):
    normalized = []
    for rectangle in rectangles:
        x1, y1, x2, y2 = map(float, rectangle)
        if x2 - x1 > GEOMETRY_EPSILON and y2 - y1 > GEOMETRY_EPSILON:
            normalized.append((x1, y1, x2, y2))
    if not normalized:
        return []
    x_values = sorted({value for row in normalized for value in (row[0], row[2])})
    cells = []
    for left, right in zip(x_values, x_values[1:]):
        if right - left <= GEOMETRY_EPSILON:
            continue
        midpoint = 0.5 * (left + right)
        intervals = sorted((row[1], row[3]) for row in normalized if row[0] < midpoint < row[2])
        if not intervals:
            continue
        start, end = intervals[0]
        merged = []
        for next_start, next_end in intervals[1:]:
            if next_start <= end + GEOMETRY_EPSILON:
                end = max(end, next_end)
            else:
                merged.append((start, end))
                start, end = next_start, next_end
        merged.append((start, end))
        cells.extend((left, a, right, b) for a, b in merged if b - a > GEOMETRY_EPSILON)
    return cells


def polygon_rectangle_union_area(geometry, rectangles):
    clipped = [value for rectangle in rectangles
               if (value := _rect_intersection(geometry.bbox, rectangle)) is not None]
    value = sum(polygon_rectangle_area(geometry, cell)
                for cell in rectangle_union_cells(clipped))
    if value > geometry.area * (1.0 + 2e-8):
        raise AssertionError(f"rectangle union exceeds target: {value}/{geometry.area}")
    return min(geometry.area, max(0.0, float(value)))


def foreground_metrics_exact(geometry, predictions, *, image_width, image_height):
    candidates = []
    image_rectangle = (0.0, 0.0, float(image_width), float(image_height))
    for prediction in predictions:
        full_rectangle = _rect_intersection(prediction["xyxy"], image_rectangle)
        if full_rectangle is None:
            continue
        local_rectangle = _rect_intersection(full_rectangle, geometry.bbox)
        if local_rectangle is None:
            continue
        overlap_area = polygon_rectangle_area(geometry, local_rectangle)
        if overlap_area <= GEOMETRY_EPSILON:
            continue
        pred_area = ((full_rectangle[2] - full_rectangle[0])
                     * (full_rectangle[3] - full_rectangle[1]))
        denominator = geometry.area + pred_area - overlap_area
        candidates.append({
            "id": str(prediction["id"]), "score": float(prediction.get("score", 0.0)),
            "rectangle": local_rectangle, "overlap_area": overlap_area,
            "mask_iou": overlap_area / denominator if denominator else 0.0,
            "foreground_precision": overlap_area / pred_area if pred_area else 0.0,
        })
    candidates.sort(key=lambda row: (-row["overlap_area"], -row["score"], row["id"]))
    rectangles = [row["rectangle"] for row in candidates]
    covered_area = polygon_rectangle_union_area(geometry, rectangles)
    best_area = candidates[0]["overlap_area"] if candidates else 0.0
    # Enforce the mathematical invariant against sub-ULP clipping noise.
    covered_area = max(covered_area, best_area)
    retained, marginal_ids, marginal_fractions = [], [], []
    current_area = 0.0
    for candidate in candidates:
        updated = polygon_rectangle_union_area(geometry, [*retained, candidate["rectangle"]])
        marginal = max(0.0, updated - current_area)
        if marginal > GEOMETRY_EPSILON * max(1.0, geometry.area):
            retained.append(candidate["rectangle"])
            current_area = updated
            marginal_ids.append(candidate["id"])
            marginal_fractions.append(marginal / geometry.area)
    mask_ac = min(1.0, max(0.0, covered_area / geometry.area))
    mask_best = min(1.0, max(0.0, best_area / geometry.area))
    mask_ac = max(mask_ac, mask_best)
    return {
        "MaskAC": mask_ac,
        "MaskBestCoverage": mask_best,
        "MaskCollectiveGain": mask_ac - mask_best,
        "MaskABO": max((row["mask_iou"] for row in candidates), default=0.0),
        "MaskBestPredPrecision": max((row["foreground_precision"] for row in candidates), default=0.0),
        "mask_intersecting_pred_count": len(candidates),
        "mask_marginal_contributor_count": len(marginal_ids),
        "mask_substantial_marginal_contributor_count": sum(value >= 0.01 for value in marginal_fractions),
        "mask_marginal_pred_ids": ";".join(marginal_ids),
        "mask_best_pred_id": candidates[0]["id"] if candidates else "",
    }


def _mscoco_sources():
    sources = [row for row in paired_yolo_sources() if row.dataset == "mscoco"]
    expected = {(model, seed) for model in YOLO_MODELS for seed in range(5)}
    actual = {(row.model, row.seed) for row in sources}
    if len(sources) != 15 or actual != expected:
        raise ValueError(f"exact analysis requires 3 models x 5 runs; got {sorted(actual)}")
    return sorted(sources, key=lambda row: (MODEL_ORDER.index(row.model), row.seed))


def _natural_source(root, source):
    return Path(root) / "runs" / "mscoco" / source.model / f"seed_{source.seed}"


def _source_output(root, source):
    return Path(root) / "runs" / "mscoco" / source.model / f"seed_{source.seed}"


def summarize_rows(rows):
    box_ac = np.asarray([float(row["AC"]) for row in rows], dtype=float)
    mask_ac = np.asarray([float(row["MaskAC"]) for row in rows], dtype=float)
    box_gain = np.asarray([float(row["CollectiveGain"]) for row in rows], dtype=float)
    mask_gain = np.asarray([float(row["MaskCollectiveGain"]) for row in rows], dtype=float)
    vc = np.asarray([float(row["local_VC"]) for row in rows], dtype=float)
    box_positive = box_gain > GAIN_THRESHOLD
    mask_positive = mask_gain > GAIN_THRESHOLD
    both = box_positive & mask_positive
    area_ratio = np.asarray(
        [float(row["geometry_to_coco_area_ratio"]) for row in rows], dtype=float
    )
    return {
        "n_gt": len(rows),
        "box_AC_mean": float(np.mean(box_ac)),
        "mask_AC_mean": float(np.mean(mask_ac)),
        "box_minus_mask_AC_mean": float(np.mean(box_ac - mask_ac)),
        "AC_mask_MAE": float(np.mean(np.abs(box_ac - mask_ac))),
        "AC_mask_RMSE": float(np.sqrt(np.mean((box_ac - mask_ac) ** 2))),
        "r_boxAC_maskAC": _pearson(box_ac, mask_ac),
        "rho_boxAC_maskAC": _spearman(box_ac, mask_ac),
        "box_gain_mean": float(np.mean(box_gain)),
        "mask_gain_mean": float(np.mean(mask_gain)),
        "box_minus_mask_gain_mean": float(np.mean(box_gain - mask_gain)),
        "gain_mask_MAE": float(np.mean(np.abs(box_gain - mask_gain))),
        "r_boxGain_maskGain": _pearson(box_gain, mask_gain),
        "rho_boxGain_maskGain": _spearman(box_gain, mask_gain),
        "P_box_AC050_mask_below050": _fraction(
            (box_ac >= AC_THRESHOLD) & (mask_ac < AC_THRESHOLD)
        ),
        "P_mask_AC050_box_below050": _fraction(
            (mask_ac >= AC_THRESHOLD) & (box_ac < AC_THRESHOLD)
        ),
        "P_abs_AC_gap_gt010": _fraction(np.abs(box_ac - mask_ac) > 0.10),
        "P_box_gain_gt005": _fraction(box_positive),
        "P_mask_gain_gt005": _fraction(mask_positive),
        "P_both_gain_gt005": _fraction(both),
        "P_box_gain_gt005_mask_not": _fraction(box_positive & ~mask_positive),
        "P_mask_gain_gt005_box_not": _fraction(mask_positive & ~box_positive),
        "P_mask_gain_gt005_given_box_gain_gt005": (
            _fraction(mask_positive[box_positive]) if np.any(box_positive) else 0.0
        ),
        "P_valid_collective": _fraction(both & (vc >= 0.50)),
        "P_VC_below050_given_both_gain_gt005": (
            _fraction(vc[both] < 0.50) if np.any(both) else 0.0
        ),
        "geometry_to_coco_area_ratio_mean": float(np.mean(area_ratio)),
        "geometry_to_coco_area_ratio_max_abs_error": float(
            np.max(np.abs(area_ratio - 1.0))
        ),
    }


def _run_complete_matches(path, source, annotation_sha256):
    if not Path(path).is_file():
        return False
    marker = _read_json(path)
    return (
        marker.get("status") == "complete"
        and marker.get("exact_version") == EXACT_VERSION
        and marker.get("source_run_fingerprint") == source.run_fingerprint
        and marker.get("annotation_sha256") == annotation_sha256
        and float(marker.get("theta", -1.0)) == PRIMARY_THETA
    )


def _evaluate_source(
        source, *, output_root, natural_root, annotation_sha256,
        images, annotations, categories, geometry_cache,
):
    output = _source_output(output_root, source)
    complete_path = output / "complete.json"
    per_gt_path = output / "per_gt_mask_validity.csv"
    if _run_complete_matches(complete_path, source, annotation_sha256):
        if not per_gt_path.is_file():
            raise FileNotFoundError(f"completed run is missing {per_gt_path}")
        return _read_json(output / "run_summary.json")

    output.mkdir(parents=True, exist_ok=True)
    natural_rows = _read_csv(
        _natural_source(natural_root, source) / "per_gt_collective_gain.csv"
    )
    natural_lookup = {
        (str(row["image_id"]), str(row["gt_id"])): row for row in natural_rows
    }
    if len(natural_lookup) != len(natural_rows):
        raise ValueError(f"duplicate natural GT rows: {source.model}/seed{source.seed}")
    document = _read_json(source.prediction_file)
    metadata = document.get("metadata", {})
    if metadata.get("run_fingerprint") != source.run_fingerprint:
        raise ValueError(f"prediction metadata mismatch: {source.prediction_file}")
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != 80:
        raise ValueError(f"invalid COCO class-name map: {source.prediction_file}")

    results = []
    for image_record in document["images"]:
        image_id = str(image_record["image_id"])
        image_width = int(image_record["width"])
        image_height = int(image_record["height"])
        predictions_by_class = defaultdict(list)
        for prediction in image_record["predictions"]:
            if float(prediction["score"]) >= PRIMARY_THETA:
                predictions_by_class[int(prediction["class_id"])].append(prediction)
        for gt in image_record["gts"]:
            gt_id = str(gt["id"])
            natural = natural_lookup.pop((image_id, gt_id), None)
            if natural is None:
                raise ValueError(f"natural analysis missed {image_id}/{gt_id}")
            annotation_id = _annotation_id(gt_id)
            annotation = annotations.get(annotation_id)
            if annotation is None:
                raise ValueError(f"COCO annotation not found: {annotation_id}")
            category = categories[int(annotation["category_id"])]
            cached_class_id = int(gt["class_id"])
            if str(category["name"]) != str(class_names[cached_class_id]):
                raise ValueError(
                    f"category mismatch for annotation {annotation_id}: "
                    f"COCO={category['name']}, cache={class_names[cached_class_id]}"
                )
            image_meta = images[int(annotation["image_id"])]
            if (int(image_meta["width"]) != image_width
                    or int(image_meta["height"]) != image_height):
                raise ValueError(f"image dimensions changed: {annotation_id}")
            geometry = geometry_cache.get(annotation_id)
            relative_error = abs(geometry.area - geometry.coco_area) / geometry.coco_area
            if relative_error > AREA_AUDIT_TOLERANCE:
                raise ValueError(
                    f"polygon area audit failed for {annotation_id}: "
                    f"{geometry.area} != {geometry.coco_area}"
                )
            foreground = foreground_metrics_exact(
                geometry, predictions_by_class[cached_class_id],
                image_width=image_width, image_height=image_height,
            )
            results.append({
                "image_id": image_id,
                "gt_id": gt_id,
                "annotation_id": annotation_id,
                "class_id": cached_class_id,
                "image_file": str(image_meta["file_name"]),
                "size_bin": str(natural["size_bin"]),
                "density_bin": str(natural["density_bin"]),
                "AC": float(natural["AC"]),
                "BestIoGT": float(natural["BestIoGT"]),
                "CollectiveGain": float(natural["CollectiveGain"]),
                "ABO": float(natural["ABO"]),
                "local_VC": float(natural["local_VC"]),
                "assigned_pred_count": int(float(natural["assigned_pred_count"])),
                "box_marginal_contributor_count": int(
                    float(natural["marginal_contributor_count"])
                ),
                "box_marginal_pred_ids": str(natural["marginal_pred_ids"]),
                "box_best_pred_id": str(natural["best_pred_id"]),
                "foreground_continuous_area": geometry.area,
                "coco_annotation_area": geometry.coco_area,
                "geometry_to_coco_area_ratio": geometry.area / geometry.coco_area,
                **foreground,
            })
    if natural_lookup:
        raise ValueError(
            f"natural analysis has {len(natural_lookup)} unmatched rows for "
            f"{source.model}/seed{source.seed}"
        )
    summary = {
        "dataset": "mscoco", "model": source.model, "seed": source.seed,
        "model_id": source.model_id,
        "source_run_fingerprint": source.run_fingerprint,
        **summarize_rows(results),
    }
    _atomic_csv(per_gt_path, results)
    _atomic_json(output / "run_summary.json", summary)
    _atomic_json(complete_path, {
        "status": "complete",
        "exact_version": EXACT_VERSION,
        "natural_analysis_version": NATURAL_ANALYSIS_VERSION,
        "source_run_fingerprint": source.run_fingerprint,
        "annotation_sha256": annotation_sha256,
        "theta": PRIMARY_THETA,
        "n_gt": len(results),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    del document, natural_rows, natural_lookup, results
    gc.collect()
    return summary


def _aggregate_five_runs(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    output = []
    for model in YOLO_MODELS:
        group = groups[model]
        if len(group) != 5 or {int(row["seed"]) for row in group} != set(range(5)):
            raise ValueError(f"incomplete five-run group: {model}")
        aggregate = {"dataset": "mscoco", "model": model, "n_seeds": 5}
        excluded = {
            "dataset", "model", "seed", "model_id", "source_run_fingerprint"
        }
        for field in group[0]:
            if field in excluded:
                continue
            values = [float(row[field]) for row in group]
            aggregate[f"{field}_mean"] = statistics.fmean(values)
            aggregate[f"{field}_std"] = statistics.stdev(values)
        output.append(aggregate)
    return output


def _threshold_summary(rows, threshold):
    box_gain = np.asarray([float(row["CollectiveGain"]) for row in rows])
    mask_gain = np.asarray([float(row["MaskCollectiveGain"]) for row in rows])
    vc = np.asarray([float(row["local_VC"]) for row in rows])
    box_positive = box_gain > threshold
    mask_positive = mask_gain > threshold
    both = box_positive & mask_positive
    box_only = box_positive & ~mask_positive
    return {
        "threshold": threshold,
        "n_gt": len(rows),
        "n_box_positive": int(np.sum(box_positive)),
        "n_mask_positive": int(np.sum(mask_positive)),
        "n_both_positive": int(np.sum(both)),
        "P_box_positive": _fraction(box_positive),
        "P_mask_positive": _fraction(mask_positive),
        "P_both_positive": _fraction(both),
        "P_box_only": _fraction(box_only),
        "P_mask_only": _fraction(mask_positive & ~box_positive),
        "confirmation_rate": (
            _fraction(mask_positive[box_positive]) if np.any(box_positive) else 0.0
        ),
        "P_VC_pass_given_both": (
            _fraction(vc[both] >= 0.50) if np.any(both) else 0.0
        ),
        "P_VC_pass_given_box_only": (
            _fraction(vc[box_only] >= 0.50) if np.any(box_only) else 0.0
        ),
    }


def _size_summary(rows):
    summary = summarize_rows(rows)
    return {
        "n_gt": summary["n_gt"],
        "AC_mask_MAE": summary["AC_mask_MAE"],
        "r_boxAC_maskAC": summary["r_boxAC_maskAC"],
        "P_box_gain_gt005": summary["P_box_gain_gt005"],
        "P_both_gain_gt005": summary["P_both_gain_gt005"],
        "confirmation_rate": summary[
            "P_mask_gain_gt005_given_box_gain_gt005"
        ],
    }


def _collect_secondary(output, sources):
    histograms = {
        model: {
            "AC": np.zeros((60, 60), dtype=np.int64),
            "gain": np.zeros((60, 60), dtype=np.int64),
        }
        for model in YOLO_MODELS
    }
    threshold_rows, size_rows, case_rows = [], [], []
    for source in sources:
        rows = _read_csv(_source_output(output, source) / "per_gt_mask_validity.csv")
        box_ac = np.asarray([float(row["AC"]) for row in rows])
        mask_ac = np.asarray([float(row["MaskAC"]) for row in rows])
        box_gain = np.clip(
            np.asarray([float(row["CollectiveGain"]) for row in rows]),
            0.0, GAIN_PLOT_MAX - 1e-10,
        )
        mask_gain = np.clip(
            np.asarray([float(row["MaskCollectiveGain"]) for row in rows]),
            0.0, GAIN_PLOT_MAX - 1e-10,
        )
        histograms[source.model]["AC"] += np.histogram2d(
            box_ac, mask_ac, bins=(AC_BINS, AC_BINS)
        )[0].astype(np.int64)
        histograms[source.model]["gain"] += np.histogram2d(
            box_gain, mask_gain, bins=(GAIN_BINS, GAIN_BINS)
        )[0].astype(np.int64)
        for threshold in GAIN_THRESHOLD_GRID:
            threshold_rows.append({
                "dataset": "mscoco", "model": source.model, "seed": source.seed,
                **_threshold_summary(rows, threshold),
            })
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row["size_bin"])].append(row)
        for size_bin in SIZE_ORDER:
            size_rows.append({
                "dataset": "mscoco", "model": source.model, "seed": source.seed,
                "size_bin": size_bin, **_size_summary(grouped[size_bin]),
            })
        if source.model == "yolo11s" and source.seed == 0:
            case_rows = rows
    return histograms, threshold_rows, size_rows, case_rows


def _plot_confirmation_curve(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    colors = ("#0072B2", "#009E73", "#D55E00")
    for model, color in zip(YOLO_MODELS, colors):
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: float(row["threshold"]),
        )
        x = np.asarray([float(row["threshold"]) for row in selected])
        confirmation = np.asarray(
            [float(row["confirmation_rate_mean"]) for row in selected]
        )
        confirmation_sd = np.asarray(
            [float(row["confirmation_rate_std"]) for row in selected]
        )
        prevalence = 100.0 * np.asarray(
            [float(row["P_both_positive_mean"]) for row in selected]
        )
        prevalence_sd = 100.0 * np.asarray(
            [float(row["P_both_positive_std"]) for row in selected]
        )
        axes[0].plot(x, confirmation, marker="o", markersize=3,
                     color=color, label=MODEL_LABELS[model])
        axes[0].fill_between(
            x, np.maximum(0.0, confirmation - confirmation_sd),
            np.minimum(1.0, confirmation + confirmation_sd),
            color=color, alpha=0.14,
        )
        axes[1].plot(x, prevalence, marker="o", markersize=3,
                     color=color, label=MODEL_LABELS[model])
        axes[1].fill_between(
            x, np.maximum(0.0, prevalence - prevalence_sd),
            prevalence + prevalence_sd, color=color, alpha=0.14,
        )
    for threshold in PAPER_GAIN_THRESHOLDS:
        for axis in axes:
            axis.axvline(threshold, color="0.65", linestyle=":", linewidth=0.9)
    axes[0].set(
        title="Conditional foreground confirmation",
        xlabel=r"Collective-gain threshold $\delta$",
        ylabel=r"$P(\Delta_m>\delta\mid\Delta_b>\delta)$",
        xlim=(GAIN_THRESHOLD_GRID[0], GAIN_THRESHOLD_GRID[-1]),
        ylim=(0.0, 1.0),
    )
    axes[1].set(
        title="Foreground-confirmed prevalence",
        xlabel=r"Collective-gain threshold $\delta$",
        ylabel="Ground-truth instances (%)",
        xlim=(GAIN_THRESHOLD_GRID[0], GAIN_THRESHOLD_GRID[-1]),
        ylim=(0.0, None),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(Path(output) / "collective_confirmation_curve.pdf", bbox_inches="tight")
    figure.savefig(Path(output) / "collective_confirmation_curve.png",
                   dpi=240, bbox_inches="tight")
    plt.close(figure)


def _plot_size_analysis(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    x = np.arange(len(SIZE_ORDER))
    width = 0.78 / len(YOLO_MODELS)
    for index, model in enumerate(YOLO_MODELS):
        lookup = {
            row["size_bin"]: row for row in rows if row["model"] == model
        }
        positions = x + (index - (len(YOLO_MODELS) - 1) / 2) * width
        mae = [100.0 * float(lookup[size]["AC_mask_MAE_mean"]) for size in SIZE_ORDER]
        mae_sd = [100.0 * float(lookup[size]["AC_mask_MAE_std"]) for size in SIZE_ORDER]
        confirmation = [
            100.0 * float(lookup[size]["confirmation_rate_mean"])
            for size in SIZE_ORDER
        ]
        confirmation_sd = [
            100.0 * float(lookup[size]["confirmation_rate_std"])
            for size in SIZE_ORDER
        ]
        axes[0].bar(positions, mae, width, yerr=mae_sd, capsize=2,
                    label=MODEL_LABELS[model])
        axes[1].bar(positions, confirmation, width,
                    yerr=confirmation_sd, capsize=2, label=MODEL_LABELS[model])
    axes[0].set(
        title="Box-to-foreground coverage discrepancy",
        xticks=x, xticklabels=SIZE_ORDER,
        ylabel="AC versus MaskAC MAE (percentage points)", ylim=(0.0, None),
    )
    axes[1].set(
        title=r"Foreground confirmation at $\delta=0.05$",
        xticks=x, xticklabels=SIZE_ORDER,
        ylabel="Conditional confirmation (%)", ylim=(0.0, 100.0),
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(Path(output) / "mask_validity_by_size.pdf", bbox_inches="tight")
    figure.savefig(Path(output) / "mask_validity_by_size.png",
                   dpi=240, bbox_inches="tight")
    plt.close(figure)


def _old_vs_exact(old_output, exact_rows, output):
    old_path = Path(old_output) / "model_summary.csv"
    if not old_path.is_file():
        return
    old_rows = {row["model"]: row for row in _read_csv(old_path)}
    fields = (
        "mask_AC_mean_mean", "AC_mask_MAE_mean", "r_boxAC_maskAC_mean",
        "mask_gain_mean_mean", "r_boxGain_maskGain_mean",
        "P_both_gain_gt005_mean",
        "P_mask_gain_gt005_given_box_gain_gt005_mean",
    )
    rows = []
    for exact in exact_rows:
        old = old_rows.get(exact["model"])
        if old is None:
            continue
        for field in fields:
            old_value, exact_value = float(old[field]), float(exact[field])
            rows.append({
                "model": exact["model"], "field": field,
                "opencv_raster_value": old_value,
                "continuous_polygon_value": exact_value,
                "exact_minus_raster": exact_value - old_value,
            })
    if rows:
        _atomic_csv(Path(output) / "opencv_vs_exact_summary.csv", rows)


def _report(model_rows, threshold_rows, output):
    lines = [
        "# Exact MS COCO Polygon-Grounded Validity Analysis", "",
        "This analysis uses continuous COCO polygons and saved full-validation "
        "predictions from three YOLO architectures and five configured seed runs. "
        "No training or model inference is performed.", "",
        "Continuous polygon areas are audited against COCO annotation areas "
        "before any coverage result is accepted.", "",
        "## Five-run results at delta = 0.05", "",
    ]
    for row in model_rows:
        lines.append(
            f"- {MODEL_LABELS[row['model']]}: "
            f"r(BoxAC, MaskAC)={float(row['r_boxAC_maskAC_mean']):.3f}; "
            f"P(box gain>0.05)={100*float(row['P_box_gain_gt005_mean']):.2f}%; "
            f"foreground-confirmed={100*float(row['P_both_gain_gt005_mean']):.2f}%; "
            f"conditional confirmation="
            f"{100*float(row['P_mask_gain_gt005_given_box_gain_gt005_mean']):.2f}%."
        )
    lines.extend(["", "## Paper thresholds", ""])
    for threshold in PAPER_GAIN_THRESHOLDS:
        lines.append(f"- delta = {threshold:.2f}")
        for model in YOLO_MODELS:
            row = next(
                value for value in threshold_rows
                if value["model"] == model
                and abs(float(value["threshold"]) - threshold) < 1e-12
            )
            lines.append(
                f"  - {MODEL_LABELS[model]}: confirmation="
                f"{100*float(row['confirmation_rate_mean']):.2f}%, "
                f"confirmed prevalence="
                f"{100*float(row['P_both_positive_mean']):.2f}%."
            )
    lines.extend([
        "", "## Outputs", "",
        "- box_vs_mask_relationships.pdf: paired box/polygon density plots.",
        "- collective_confirmation_curve.pdf: gain-threshold sensitivity.",
        "- mask_validity_by_size.pdf: small/medium/large analysis.",
        "- mask_validity_rates.pdf: delta=0.05 prevalence summary.",
        "- mask_grounded_cases.pdf: qualitative examples.",
        "- mask_grounded_validity_table.tex: paper-ready table.",
        "- opencv_vs_exact_summary.csv: change from the raster analysis.",
    ])
    _atomic_text(Path(output) / "README.md", "\n".join(lines) + "\n")


def preflight_exact_mask_grounded_validity(
        output=DEFAULT_OUTPUT, natural_output=NATURAL_OUTPUT,
        annotation_file=ANNOTATION_FILE,
):
    output = Path(output).resolve()
    natural_output = Path(natural_output).resolve()
    annotation_file = Path(annotation_file).resolve()
    if not annotation_file.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {annotation_file}")
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"COCO validation images not found: {IMAGE_ROOT}")
    natural_complete = natural_output / "analysis_complete.json"
    if not natural_complete.is_file():
        raise FileNotFoundError("natural collective-gain analysis is missing")
    marker = _read_json(natural_complete)
    if (marker.get("status") != "complete"
            or marker.get("analysis_version") != NATURAL_ANALYSIS_VERSION):
        raise ValueError("natural collective-gain analysis is incomplete or changed")

    sources = _mscoco_sources()
    images, annotations, categories = _load_coco(annotation_file)
    first_rows = _read_csv(
        _natural_source(natural_output, sources[0]) / "per_gt_collective_gain.csv"
    )
    cache = PolygonGeometryCache(annotations)
    errors, crowd = [], 0
    for row in first_rows:
        annotation_id = _annotation_id(row["gt_id"])
        annotation = annotations.get(annotation_id)
        if annotation is None:
            raise ValueError(f"COCO annotation not found: {annotation_id}")
        crowd += int(annotation.get("iscrowd", 0))
        geometry = cache.get(annotation_id)
        errors.append(abs(geometry.area - geometry.coco_area) / geometry.coco_area)
    maximum_error = max(errors, default=0.0)
    if crowd:
        raise ValueError(f"selected analysis contains {crowd} crowd annotations")
    if maximum_error > AREA_AUDIT_TOLERANCE:
        raise ValueError(f"polygon area audit failed: max error={maximum_error}")
    for source in sources:
        for path in (
            source.prediction_file,
            _natural_source(natural_output, source) / "per_gt_collective_gain.csv",
            _natural_source(natural_output, source) / "complete.json",
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"required input missing: {path}")
    return {
        "status": "passed", "exact_version": EXACT_VERSION,
        "natural_analysis_version": NATURAL_ANALYSIS_VERSION,
        "vc_version": VC_VERSION, "theta": PRIMARY_THETA,
        "n_runs": len(sources), "n_models": len(YOLO_MODELS),
        "n_configured_seed_runs_per_model": 5,
        "n_gt_per_run": len(first_rows),
        "n_coco_annotations": len(annotations),
        "n_coco_categories": len(categories),
        "selected_crowd_annotations": crowd,
        "geometry_method": (
            "continuous COCO polygon clipping against disjoint "
            "axis-aligned rectangle-union cells"
        ),
        "geometry_area_relative_error_mean": float(np.mean(errors)),
        "geometry_area_relative_error_max": float(maximum_error),
        "annotation_file": str(annotation_file),
        "annotation_sha256": _sha256(annotation_file),
        "output": str(output), "training": False, "inference": False,
    }


def run_exact_mask_grounded_validity(
        output=DEFAULT_OUTPUT, natural_output=NATURAL_OUTPUT,
        annotation_file=ANNOTATION_FILE, old_output=OLD_OUTPUT,
):
    output = Path(output).resolve()
    natural_output = Path(natural_output).resolve()
    annotation_file = Path(annotation_file).resolve()
    old_output = Path(old_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = preflight_exact_mask_grounded_validity(
        output=output, natural_output=natural_output,
        annotation_file=annotation_file,
    )
    _atomic_json(output / "preflight.json", preflight)
    annotation_sha256 = str(preflight["annotation_sha256"])
    images, annotations, categories = _load_coco(annotation_file)
    geometry_cache = PolygonGeometryCache(annotations)
    sources = _mscoco_sources()
    run_rows = []
    for index, source in enumerate(sources, 1):
        print(
            f"EXACT-MASK {index:02d}/{len(sources):02d} START: "
            f"mscoco:{source.model}:seed{source.seed}", flush=True,
        )
        run_rows.append(_evaluate_source(
            source, output_root=output, natural_root=natural_output,
            annotation_sha256=annotation_sha256, images=images,
            annotations=annotations, categories=categories,
            geometry_cache=geometry_cache,
        ))
        print(
            f"EXACT-MASK {index:02d}/{len(sources):02d} COMPLETE: "
            f"mscoco:{source.model}:seed{source.seed}; "
            f"geometry_cache={geometry_cache.size} GT", flush=True,
        )

    model_rows = _aggregate_five_runs(run_rows)
    _atomic_csv(output / "run_summary.csv", run_rows)
    _atomic_csv(output / "model_summary.csv", model_rows)
    histograms, threshold_rows, size_rows, case_rows = _collect_secondary(output, sources)
    threshold_model_rows = _aggregate_seed_rows(
        threshold_rows, ("dataset", "model", "threshold")
    )
    size_model_rows = _aggregate_seed_rows(
        size_rows, ("dataset", "model", "size_bin")
    )
    _atomic_csv(output / "threshold_by_run.csv", threshold_rows)
    _atomic_csv(output / "threshold_by_model.csv", threshold_model_rows)
    _atomic_csv(output / "size_by_run.csv", size_rows)
    _atomic_csv(output / "size_by_model.csv", size_model_rows)

    _plot_relationships(histograms, output)
    _plot_validity_rates(model_rows, output)
    _plot_confirmation_curve(threshold_model_rows, output)
    _plot_size_analysis(size_model_rows, output)
    case_source = next(
        source for source in sources
        if source.model == "yolo11s" and source.seed == 0
    )
    _plot_cases(
        case_rows, case_source, annotations, images,
        PolygonMaskCache(annotations, images), output,
    )
    _latex_table(model_rows, output)
    _old_vs_exact(old_output, model_rows, output)
    _report(model_rows, threshold_model_rows, output)

    complete = {
        "status": "complete", "exact_version": EXACT_VERSION,
        "natural_analysis_version": NATURAL_ANALYSIS_VERSION,
        "vc_version": VC_VERSION, "theta": PRIMARY_THETA,
        "annotation_sha256": annotation_sha256,
        "n_runs": len(run_rows), "n_models": len(model_rows),
        "n_configured_seed_runs_per_model": 5,
        "paper_gain_thresholds": list(PAPER_GAIN_THRESHOLDS),
        "threshold_grid": list(GAIN_THRESHOLD_GRID),
        "geometry_cache_count": geometry_cache.size,
        "geometry_area_relative_error_max": preflight[
            "geometry_area_relative_error_max"
        ],
        "training": False, "inference": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
    }
    _atomic_json(output / "analysis_complete.json", complete)
    return output
