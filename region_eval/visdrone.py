"""VisDrone regional diagnostic preprocessing, using continuous box geometry.

This is a declared research protocol, not a port of official raster AP/AR.
Only category 0 defines spatial ignore regions, as in dropObjectsInIgr.m.
"""

from hashlib import sha256
from math import isfinite
from pathlib import Path

from .geometry import area, intersection, union_area


CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]
IGNORE_POLICY = "visdrone_category0_union_ioa_ge_0.5_continuous_v1"
IGNORE_REFERENCE = "https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/dropObjectsInIgr.m"


def select_images(image_dir, annotation_dir, count=50, seed=0):
    """Hash-rank filenames for deterministic selection independent of predictions."""
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("count must be a positive integer; seed must be an integer")
    image_dir, annotation_dir = Path(image_dir), Path(annotation_dir)
    images = sorted(image_dir.glob("*.jpg"))
    annotations = sorted(annotation_dir.glob("*.txt"))
    if not images or {p.stem for p in images} != {p.stem for p in annotations}:
        raise ValueError("image JPG and annotation TXT stems must match exactly")
    if count > len(images):
        raise ValueError(f"requested {count} images but only {len(images)} are available")
    ranked = sorted(images, key=lambda p: (sha256(f"{seed}:{p.name}".encode()).hexdigest(), p.name))
    return sorted(ranked[:count]), [p.name for p in images]


def parse_annotations(text):
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.strip().rstrip(",").split(",")
        if len(fields) != 8:
            raise ValueError(f"annotation line {line_number}: expected eight fields")
        values = [float(v) for v in fields]
        if not all(isfinite(v) for v in values):
            raise ValueError(f"annotation line {line_number}: nonfinite value")
        x, y, w, h, score, category, truncation, occlusion = values
        if w <= 0 or h <= 0 or not isfinite(x + w) or not isfinite(y + h):
            raise ValueError(f"annotation line {line_number}: invalid rectangle")
        if score not in (0, 1) or not category.is_integer() or not 0 <= category <= 11:
            raise ValueError(f"annotation line {line_number}: invalid score/category")
        if truncation not in (0, 1) or occlusion not in (0, 1, 2):
            raise ValueError(f"annotation line {line_number}: invalid truncation/occlusion")
        # This validation split has no score=0 annotations for classes 1..10.
        # Do not invent category-specific ignore matching for a different split.
        if 1 <= category <= 10 and score == 0:
            raise ValueError("score=0 target-class GT needs a separate ignore-matching protocol")
        rows.append(dict(id=f"gt_{line_number:06d}", xyxy=[x, y, x + w, y + h],
                         score=int(score), category=int(category),
                         truncation=int(truncation), occlusion=int(occlusion)))
    return rows


def _clip(rect, width, height):
    return (max(0.0, min(width, rect[0])), max(0.0, min(height, rect[1])),
            max(0.0, min(width, rect[2])), max(0.0, min(height, rect[3])))


def ignore_fraction(rect, ignore_rects):
    size = area(rect)
    if size <= 0:
        raise ValueError("ignore overlap needs a positive-area rectangle")
    overlaps = [r for other in ignore_rects if (r := intersection(rect, other)) is not None]
    return min(1.0, union_area(overlaps) / size)


def prepare_ground_truth(rows, width, height):
    if not all(isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("positive finite image dimensions required")
    clipped = []
    clipped_ids = []
    for row in rows:
        rect = _clip(row["xyxy"], width, height)
        if area(rect) <= 0:
            raise ValueError(f"{row['id']}: annotation clipped to zero area")
        if list(rect) != row["xyxy"]:
            clipped_ids.append(row["id"])
        clipped.append((row, rect))
    ignore_rects = [rect for row, rect in clipped if row["category"] == 0]
    gts, dropped = [], []
    excluded = [row["id"] for row, _ in clipped if row["category"] in (0, 11)]
    for row, rect in clipped:
        if row["category"] in (0, 11):
            continue
        overlap = ignore_fraction(rect, ignore_rects)
        if overlap >= 0.5:
            dropped.append(dict(id=row["id"], ignore_fraction=overlap))
        else:
            gts.append(dict(id=row["id"], xyxy=list(rect), class_id=row["category"] - 1))
    audit = dict(n_raw_annotations=len(rows), n_ignore_regions=len(ignore_rects),
                 n_others=sum(row["category"] == 11 for row in rows),
                 clipped_annotation_ids=clipped_ids, excluded_category_ids=excluded,
                 dropped_gt_in_ignore=dropped, n_gt=len(gts))
    return gts, ignore_rects, audit


def filter_predictions(predictions, ignore_rects, width, height):
    """Retain full boxes below 50% ignore overlap; never trim into fragments."""
    kept, dropped, invalid, clipped_ids = [], [], [], []
    identifiers = set()
    for pred in predictions:
        identifier = pred["id"]
        if identifier in identifiers:
            raise ValueError("duplicate prediction id")
        identifiers.add(identifier)
        rect = pred["xyxy"]
        if len(rect) != 4 or not all(isfinite(v) for v in rect):
            invalid.append(dict(id=identifier, reason="nonfinite_coordinates"))
            continue
        if rect[2] < rect[0] or rect[3] < rect[1]:
            raise ValueError("reversed prediction rectangle")
        if (type(pred["class_id"]) is not int or pred["class_id"] not in range(10)
                or not isfinite(pred["score"]) or not 0 <= pred["score"] <= 1):
            raise ValueError("invalid prediction class/score")
        clipped = _clip(rect, width, height)
        if area(clipped) <= 0:
            invalid.append(dict(id=identifier, reason="degenerate_after_clipping"))
            continue
        if list(clipped) != list(rect):
            clipped_ids.append(identifier)
        overlap = ignore_fraction(clipped, ignore_rects)
        if overlap >= 0.5:
            dropped.append(dict(id=identifier, ignore_fraction=overlap))
        else:
            kept.append(dict(pred, xyxy=list(clipped)))
    return kept, dict(n_raw_predictions=len(predictions), n_kept_predictions=len(kept),
                      dropped_pred_in_ignore=dropped, invalid_predictions=invalid,
                      clipped_prediction_ids=clipped_ids)
