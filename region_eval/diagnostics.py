"""Five diagnostic outputs, independent of Ultralytics, GPU and file I/O."""

from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral, Real

from .geometry import Rect, area, intersection, union_area

METRIC_VERSION = "region-diagnostics-0.1.0"


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identifier(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _class_id(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("class_id must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class Box:
    id: str
    xyxy: Rect
    class_id: int
    score: float | None = None


@dataclass(frozen=True)
class Config:
    theta: float = 0.25
    tau: float = 0.5
    cg_epsilon: float = 1e-9
    class_ids: tuple[int, ...] | None = None

    def __post_init__(self):
        for name in ("theta", "tau", "cg_epsilon"):
            value = _number(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if not 0 <= self.theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.cg_epsilon < 0:
            raise ValueError("cg_epsilon must be nonnegative")
        if self.class_ids is not None:
            ids = tuple(_class_id(c) for c in self.class_ids)
            if len(set(ids)) != len(ids):
                raise ValueError("class_ids must not contain duplicates")
            object.__setattr__(self, "class_ids", tuple(sorted(ids)))


@dataclass
class ImageDiagnostics:
    image_id: str
    width: float
    height: float
    config: Config
    per_gt: list[dict]
    per_class: list[dict]
    counts: dict[str, int]


def _prepare(boxes, width, height, config, *, prediction, counts):
    result = []
    seen = set()
    kind = "pred" if prediction else "gt"
    for box in boxes:
        if not isinstance(box, Box):
            raise ValueError("inputs must contain region_eval.Box objects")
        _identifier(box.id, f"{kind} id")
        if box.id in seen:
            raise ValueError(f"duplicate {kind} id: {box.id}")
        seen.add(box.id)
        cls = _class_id(box.class_id)
        if config.class_ids is not None and cls not in config.class_ids:
            raise ValueError(f"unknown class_id: {cls}")
        score = None
        if prediction:
            score = _number(box.score, "prediction score")
            if not 0 <= score <= 1:
                raise ValueError("prediction score must be in [0, 1]")
        if not isinstance(box.xyxy, (list, tuple)) or len(box.xyxy) != 4:
            raise ValueError(f"{kind} {box.id}: xyxy must contain four numbers")
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in box.xyxy):
            raise ValueError(f"{kind} {box.id}: coordinates must be real numbers")
        rect = tuple(float(v) for v in box.xyxy)
        if not all(isfinite(v) for v in rect):
            if prediction:
                counts["skipped_nonfinite_pred"] += 1
                continue
            raise ValueError(f"gt {box.id}: nonfinite coordinates")
        if rect[2] < rect[0] or rect[3] < rect[1]:
            raise ValueError(f"{kind} {box.id}: reversed coordinates are not repaired")
        clipped = (min(width, max(0.0, rect[0])), min(height, max(0.0, rect[1])),
                   min(width, max(0.0, rect[2])), min(height, max(0.0, rect[3])))
        if clipped != rect:
            counts[f"clipped_{kind}"] += 1
        if area(clipped) <= 0:
            if prediction:
                counts["skipped_degenerate_pred"] += 1
                continue
            raise ValueError(f"gt {box.id}: zero area after image clipping")
        if prediction and score < config.theta:
            counts["filtered_low_score_pred"] += 1
            continue
        result.append(Box(box.id, clipped, cls, score))
    return sorted(result, key=lambda b: b.id)


def image_region_diagnostics(gts, predictions, image_size, *, image_id,
                             config: Config | None = None) -> ImageDiagnostics:
    """Score one image. image_size is (width, height); input is original-image xyxy.

    No NMS, letterbox inversion, assignment, crowd handling or stateful accumulation
    is performed here. The caller must supply an already-defined evaluation set.
    """
    config = Config() if config is None else config
    if not isinstance(config, Config):
        raise ValueError("config must be a Config")
    _identifier(image_id, "image_id")
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width, height = (_number(v, "image dimension") for v in image_size)
    if width <= 0 or height <= 0 or not isfinite(width * height):
        raise ValueError("image dimensions and image area must be positive and finite")
    gts, predictions = list(gts), list(predictions)
    counts = dict(n_input_gt=len(gts), n_input_pred=len(predictions),
                  clipped_gt=0, clipped_pred=0, skipped_nonfinite_pred=0,
                  skipped_degenerate_pred=0, filtered_low_score_pred=0)
    gs = _prepare(gts, width, height, config, prediction=False, counts=counts)
    ps = _prepare(predictions, width, height, config, prediction=True, counts=counts)
    counts.update(n_gt=len(gs), n_pred=len(ps))
    gt_rows, class_rows = [], []
    for cls in sorted({b.class_id for b in gs + ps}):
        cg = [b for b in gs if b.class_id == cls]
        cp = [b for b in ps if b.class_id == cls]
        hits = {b.id: 0 for b in cp}
        all_intersections = []
        for gt in cg:
            clipped_preds = []
            best_area, best_id = 0.0, None
            for pred in cp:
                overlap = intersection(gt.xyxy, pred.xyxy)
                if overlap is None:
                    continue
                clipped_preds.append(overlap)
                hits[pred.id] += 1
                overlap_area = area(overlap)
                # cp is sorted by stable ID: equal-area ties are input-order invariant.
                if overlap_area > best_area:
                    best_area, best_id = overlap_area, pred.id
            gt_area = area(gt.xyxy)
            best = min(1.0, best_area / gt_area)
            ac = min(1.0, max(best, union_area(clipped_preds) / gt_area))
            all_intersections.extend(clipped_preds)
            gt_rows.append(dict(image_id=image_id, gt_id=gt.id, class_id=cls,
                                theta=config.theta, gt_area=gt_area, AC=ac,
                                BestIoGT_i=best, CollectiveGain_i=ac - best,
                                coverage_hit=ac >= config.tau,
                                intersecting_pred_count=len(clipped_preds),
                                best_pred_id=best_id))
        ga = union_area([b.xyxy for b in cg])
        pa = union_area([b.xyxy for b in cp])
        ia = min(ga, pa, union_area(all_intersections))
        # Both-empty class pairs are absent; no epsilon modifies valid denominators.
        scale = max(ga, pa)
        g_scaled, p_scaled, i_scaled = ga / scale, pa / scale, ia / scale
        dice = 2 * i_scaled / (g_scaled + p_scaled)
        iou = i_scaled / (g_scaled + p_scaled - i_scaled)
        orphan_boxes = [b.xyxy for b in cp if hits[b.id] == 0]
        class_rows.append(dict(image_id=image_id, class_id=cls, theta=config.theta,
                               n_gt=len(cg), n_pred=len(cp), gt_union_area=ga,
                               pred_union_area=pa, intersection_area=ia,
                               background_area=max(0.0, pa - ia),
                               SpatialPrecision=ia / pa if pa else None,
                               ClassUnionDice=dice, ClassUnionIoU=iou,
                               active_pair=True, orphan_pred_count=len(orphan_boxes),
                               orphan_union_area=union_area(orphan_boxes),
                               multigt_pred_count=sum(v >= 2 for v in hits.values())))
    return ImageDiagnostics(image_id, width, height, config, gt_rows, class_rows, counts)


def _mean(values):
    return fsum(values) / len(values) if values else None


def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (position - low) * (values[high] - values[low])


def aggregate(images, *, config: Config | None = None) -> dict:
    """GT-macro coverage and active-image-class-macro union metrics.

    Duplicate images and incompatible operating points fail instead of silently
    changing dataset weights. The function is stateless and may be repeated.
    """
    images = sorted(list(images), key=lambda d: d.image_id)
    config = (images[0].config if images else Config()) if config is None else config
    if not isinstance(config, Config):
        raise ValueError("config must be a Config")
    if len({d.image_id for d in images}) != len(images):
        raise ValueError("duplicate image_id in aggregation")
    if any(d.config != config for d in images):
        raise ValueError("cannot aggregate different metric configurations")
    gs = [g for d in images for g in d.per_gt]
    pairs = [p for d in images for p in d.per_class]
    gains = [g["CollectiveGain_i"] for g in gs]
    counts = {key: sum(d.counts[key] for d in images) for key in (
        "n_input_gt", "n_input_pred", "n_gt", "n_pred", "clipped_gt", "clipped_pred",
        "skipped_nonfinite_pred", "skipped_degenerate_pred", "filtered_low_score_pred")}
    return dict(metric_version=METRIC_VERSION, theta=config.theta, tau=config.tau,
                cg_epsilon=config.cg_epsilon, class_aware=True,
                coverage_aggregation="gt_macro",
                aggregation="active_image_class_macro", n_images=len(images), **counts,
                n_active_image_classes=len(pairs),
                mAC=_mean([g["AC"] for g in gs]),
                BestIoGT=_mean([g["BestIoGT_i"] for g in gs]),
                CollectiveGain=_mean(gains),
                RecallACtau=_mean([int(g["coverage_hit"]) for g in gs]),
                ClassUnionDice=_mean([p["ClassUnionDice"] for p in pairs]),
                ClassUnionIoU=_mean([p["ClassUnionIoU"] for p in pairs]),
                avg_preds=counts["n_pred"] / len(images) if images else None,
                cg_positive_fraction=_mean([int(g > config.cg_epsilon) for g in gains]),
                cg_p50=_quantile(gains, 0.5), cg_p90=_quantile(gains, 0.9),
                cg_p95=_quantile(gains, 0.95),
                orphan_pred_count=sum(p["orphan_pred_count"] for p in pairs),
                multigt_pred_count=sum(p["multigt_pred_count"] for p in pairs))
