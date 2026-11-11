"""MS COCO validation adapter for the VC post-NMS cache protocol."""

from collections import defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from .geometry import area, intersection, union_area


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
COCO_CROWD_POLICY = "coco_iscrowd_same_class_bbox_union_ioa_ge_0.5_continuous_v1"


def _clip(rect, width, height):
    return (
        max(0.0, min(width, rect[0])),
        max(0.0, min(height, rect[1])),
        max(0.0, min(width, rect[2])),
        max(0.0, min(height, rect[3])),
    )


def _bbox(annotation, width, height):
    raw = annotation.get("bbox")
    if (not isinstance(raw, list) or len(raw) != 4
            or not all(isinstance(value, (int, float))
                       and not isinstance(value, bool) and isfinite(value)
                       for value in raw)):
        raise ValueError("COCO annotations require a finite xywh bbox")
    x, y, w, h = raw
    if w <= 0 or h <= 0:
        raise ValueError("COCO annotation bbox must have positive width and height")
    rect = _clip((x, y, x + w, y + h), width, height)
    if area(rect) <= 0:
        raise ValueError("COCO annotation bbox disappears after clipping")
    return rect


def _read_instances(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    path = dataset_root / "annotations" / "instances_val2017.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("COCO instances file must contain an object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"COCO instances file is missing {key}")
    categories = sorted(value["categories"], key=lambda row: row["id"])
    names = [row.get("name") for row in categories]
    if names != COCO_NAMES:
        raise ValueError("COCO category order/names do not match the 80-class model mapping")
    category_map = {row["id"]: index for index, row in enumerate(categories)}
    if len(category_map) != len(COCO_NAMES):
        raise ValueError("COCO category IDs must be unique")
    return path, value, category_map


def select_coco_images(dataset_root, count=20, seed=0):
    """Return a deterministic image subset plus its annotations and class map."""
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("count must be a positive integer; seed must be an integer")
    dataset_root = Path(dataset_root).resolve()
    annotation_path, value, category_map = _read_instances(dataset_root)
    image_dir = dataset_root / "images" / "val"
    image_rows = sorted(value["images"], key=lambda row: row["file_name"])
    if count > len(image_rows):
        raise ValueError(f"requested {count} images but only {len(image_rows)} are available")
    names = [row["file_name"] for row in image_rows]
    if len(names) != len(set(names)):
        raise ValueError("COCO image file names must be unique")
    ranked = sorted(
        image_rows,
        key=lambda row: (
            sha256(f"{seed}:{row['file_name']}".encode()).hexdigest(),
            row["file_name"],
        ),
    )
    selected = sorted(ranked[:count], key=lambda row: row["file_name"])
    selected_ids = {row["id"] for row in selected}
    annotations = defaultdict(list)
    for annotation in value["annotations"]:
        if annotation.get("image_id") in selected_ids:
            annotations[annotation["image_id"]].append(annotation)
    output = []
    for row in selected:
        path = image_dir / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"COCO validation image is missing: {path}")
        output.append((row, sorted(annotations[row["id"]], key=lambda item: item["id"])))
    return output, names, category_map, annotation_path


def prepare_coco_image(image_row, annotations, category_map):
    width, height = image_row.get("width"), image_row.get("height")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("COCO image dimensions must be positive and finite")
    gts = []
    crowd_by_class = defaultdict(list)
    clipped_ids = []
    ignored_ids = []
    seen = set()
    for annotation in annotations:
        identifier = annotation.get("id")
        if identifier in seen:
            raise ValueError("duplicate COCO annotation id")
        seen.add(identifier)
        category_id = annotation.get("category_id")
        if category_id not in category_map:
            raise ValueError("COCO annotation uses an unknown category id")
        class_id = category_map[category_id]
        rect = _bbox(annotation, width, height)
        x, y, w, h = annotation["bbox"]
        if list(rect) != [x, y, x + w, y + h]:
            clipped_ids.append(identifier)
        ignored = bool(annotation.get("iscrowd", 0) or annotation.get("ignore", 0))
        if ignored:
            ignored_ids.append(identifier)
            crowd_by_class[class_id].append(rect)
        else:
            gts.append(dict(
                id=f"gt_{identifier}",
                class_id=class_id,
                xyxy=list(rect),
            ))
    audit = dict(
        n_raw_annotations=len(annotations),
        n_gt=len(gts),
        n_ignored_annotations=len(ignored_ids),
        ignored_annotation_ids=ignored_ids,
        clipped_annotation_ids=clipped_ids,
    )
    return gts, dict(crowd_by_class), audit


def _ignore_fraction(rect, ignore_rects):
    overlaps = [
        overlap for other in ignore_rects
        if (overlap := intersection(tuple(rect), tuple(other))) is not None
    ]
    return min(1.0, union_area(overlaps) / area(tuple(rect)))


def filter_coco_predictions(predictions, crowd_by_class, width, height):
    """Drop predictions mostly contained in a same-class crowd/ignore bbox."""
    kept, dropped, invalid, clipped_ids = [], [], [], []
    seen = set()
    for prediction in predictions:
        identifier = prediction.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("prediction IDs must be unique nonempty strings")
        seen.add(identifier)
        rect = prediction.get("xyxy")
        if (not isinstance(rect, list) or len(rect) != 4
                or not all(isinstance(value, (int, float))
                           and not isinstance(value, bool) and isfinite(value)
                           for value in rect)):
            invalid.append(dict(id=identifier, reason="nonfinite_coordinates"))
            continue
        if rect[2] < rect[0] or rect[3] < rect[1]:
            raise ValueError("reversed prediction rectangle")
        class_id = prediction.get("class_id")
        score = prediction.get("score")
        if (type(class_id) is not int or class_id not in range(len(COCO_NAMES))
                or not isinstance(score, (int, float)) or isinstance(score, bool)
                or not isfinite(score) or not 0 <= score <= 1):
            raise ValueError("invalid COCO prediction class/score")
        clipped = _clip(rect, width, height)
        if area(clipped) <= 0:
            invalid.append(dict(id=identifier, reason="degenerate_after_clipping"))
            continue
        if list(clipped) != rect:
            clipped_ids.append(identifier)
        fraction = _ignore_fraction(clipped, crowd_by_class.get(class_id, []))
        if fraction >= 0.5:
            dropped.append(dict(id=identifier, ignore_fraction=fraction))
        else:
            kept.append(dict(prediction, xyxy=list(clipped)))
    return kept, dict(
        n_raw_predictions=len(predictions),
        n_kept_predictions=len(kept),
        dropped_pred_in_crowd=dropped,
        invalid_predictions=invalid,
        clipped_prediction_ids=clipped_ids,
    )
