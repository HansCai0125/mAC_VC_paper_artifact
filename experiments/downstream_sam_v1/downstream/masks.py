"""Box-prompted masks and held-out COCO foreground evaluation.

Ground truth enters only the evaluation functions. ``SamBoxPredictor`` accepts
images and boxes and has no annotation, class, or metric-selection interface.
All mask encodings returned by this module are JSON-serializable COCO RLEs.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import redirect_stdout
import copy
import importlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def _mask_api():
    try:
        return importlib.import_module("pycocotools.mask")
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for COCO masks. Run prepare_environment.py "
            "using the same Python interpreter as run_experiment.py."
        ) from exc


def _shape(height, width):
    if (isinstance(height, bool) or isinstance(width, bool)
            or int(height) != height or int(width) != width
            or height <= 0 or width <= 0):
        raise ValueError("Mask height and width must be positive integers")
    return int(height), int(width)


def serializable_rle(rle):
    """Return a defensive copy of one compressed COCO RLE."""
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not isinstance(counts, str):
        raise ValueError("Expected compressed COCO RLE counts")
    return {"size": [int(x) for x in rle["size"]], "counts": counts}


def encode_binary_mask(mask):
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("A binary mask must have shape (height, width)")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("Mask contains non-binary values")
    return serializable_rle(
        _mask_api().encode(np.asfortranarray(value, dtype=np.uint8)))


def _native_rle(rle, height, width):
    height, width = _shape(height, width)
    if not isinstance(rle, dict) or rle.get("size") != [height, width]:
        raise ValueError("RLE size does not match the original image")
    counts = rle.get("counts")
    if isinstance(counts, str):
        return {"size": [height, width], "counts": counts.encode("ascii")}
    if isinstance(counts, bytes):
        return {"size": [height, width], "counts": counts}
    if isinstance(counts, list):
        if (any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                for x in counts) or sum(counts) != height * width):
            raise ValueError("Invalid uncompressed RLE counts")
        return _mask_api().frPyObjects(
            {"size": [height, width], "counts": counts}, height, width)
    raise ValueError("Unsupported COCO RLE counts")


def decode_rle(rle, height=None, width=None):
    if height is None or width is None:
        height, width = rle["size"]
    value = _mask_api().decode(_native_rle(rle, height, width))
    if value.shape != (height, width):
        raise ValueError("Expected one RLE mask in original-image coordinates")
    return value.astype(bool, copy=False)


def _empty_rle(height, width):
    return encode_binary_mask(np.zeros((height, width), dtype=np.uint8))


def _merge_rles(rles, height, width):
    if not rles:
        return _empty_rle(height, width)
    return serializable_rle(_mask_api().merge(
        [_native_rle(rle, height, width) for rle in rles], intersect=False))


def _segmentation_rle(segmentation, height, width):
    if isinstance(segmentation, dict):
        result = _native_rle(segmentation, height, width)
    elif isinstance(segmentation, list) and segmentation:
        for polygon in segmentation:
            if (not isinstance(polygon, list) or len(polygon) < 6
                    or len(polygon) % 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in polygon)):
                raise ValueError("Invalid polygon segmentation")
        result = _mask_api().merge(
            _mask_api().frPyObjects(segmentation, height, width), intersect=False)
    else:
        raise ValueError("Missing or empty segmentation")
    # Decode once to reject malformed compressed encodings at the boundary.
    decoded = _mask_api().decode(result)
    if decoded.shape != (height, width):
        raise ValueError("Segmentation dimensions do not match image dimensions")
    return serializable_rle(result)


class CocoMasks:
    """COCO annotations with a class-specific crowd/ignore pixel policy.

    Valid non-crowd foreground takes precedence where it overlaps an ignored
    region. Invalid non-crowd segmentations fail visibly. An ignored annotation
    with no decodable mask may use its bounding box; every such annotation is
    counted in ``audit``. A small LRU retains compressed RLEs, not pixel arrays.
    """

    def __init__(self, annotation_file):
        self.annotation_file = Path(annotation_file).resolve()
        data = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self.images = {int(row["id"]): row for row in data["images"]}
        if len(self.images) != len(data["images"]):
            raise ValueError("Duplicate COCO image IDs")
        categories = sorted(data["categories"], key=lambda row: row["id"])
        self.category_ids = [int(row["id"]) for row in categories]
        self.category_names = [row["name"] for row in categories]
        if len(set(self.category_ids)) != len(self.category_ids):
            raise ValueError("Duplicate COCO category IDs")
        self.n_classes = len(self.category_ids)
        self.category_to_class = {
            category: index for index, category in enumerate(self.category_ids)}
        self.annotations = defaultdict(list)
        seen = set()
        for annotation in data["annotations"]:
            ann_id = annotation["id"]
            if ann_id in seen:
                raise ValueError(f"Duplicate COCO annotation ID: {ann_id}")
            seen.add(ann_id)
            image_id = int(annotation["image_id"])
            if image_id not in self.images:
                raise ValueError(f"Unknown annotation image ID: {image_id}")
            if annotation["category_id"] not in self.category_to_class:
                raise ValueError("Unknown annotation category")
            self.annotations[image_id].append(annotation)
        self._cache = OrderedDict()
        self._fallback_ids = set()
        self._zero_mask_ids = set()

    @property
    def audit(self):
        return {
            "ignore_policy": "same_class_ignore_pixels_except_positive_target",
            "ignored_bbox_fallback_count": len(self._fallback_ids),
            "ignored_bbox_fallback_annotation_ids": sorted(self._fallback_ids),
            "zero_rasterized_positive_count": len(self._zero_mask_ids),
            "zero_rasterized_positive_annotation_ids": sorted(self._zero_mask_ids),
        }

    def ground_truth(self, image_id):
        image_id = int(image_id)
        if image_id in self._cache:
            self._cache.move_to_end(image_id)
            return self._cache[image_id]
        image = self.images[image_id]
        height, width = _shape(image["height"], image["width"])
        grouped = defaultdict(lambda: {"target": [], "ignore": []})
        for annotation in self.annotations.get(image_id, []):
            class_id = self.category_to_class[annotation["category_id"]]
            ignored = bool(annotation.get("iscrowd", 0) or annotation.get("ignore", 0))
            try:
                rle = _segmentation_rle(
                    annotation.get("segmentation"), height, width)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                if not ignored:
                    raise ValueError(
                        f"Invalid positive segmentation for annotation "
                        f"{annotation['id']} in image {image_id}") from exc
                box = annotation.get("bbox", [])
                if (len(box) != 4 or any(not isinstance(v, (int, float))
                                        or not math.isfinite(v) for v in box)
                        or box[2] <= 0 or box[3] <= 0):
                    raise ValueError(
                        f"Invalid ignored mask and bbox for {annotation['id']}") from exc
                x, y, box_width, box_height = box
                rle = _segmentation_rle([[x, y, x + box_width, y,
                                         x + box_width, y + box_height,
                                         x, y + box_height]], height, width)
                self._fallback_ids.add(annotation["id"])
            if not ignored and float(_mask_api().area(
                    _native_rle(rle, height, width))) == 0:
                # A valid subpixel polygon can rasterize to zero pixels. It is
                # retained and explicitly counted; no bbox replacement is made.
                self._zero_mask_ids.add(annotation["id"])
            grouped[class_id]["ignore" if ignored else "target"].append(rle)
        output = {}
        for class_id, masks in grouped.items():
            target = _merge_rles(masks["target"], height, width)
            ignore = _merge_rles(masks["ignore"], height, width)
            if masks["target"] and masks["ignore"]:
                ignore = encode_binary_mask(
                    decode_rle(ignore) & ~decode_rle(target))
            output[class_id] = {"target": target, "ignore": ignore}
        self._cache[image_id] = output
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return output


def evaluate_class_unions(predictions, mask_rles, gt_by_class,
                          height, width, n_classes):
    """Return int64 ``[class, TP/FP/FN]`` sufficient statistics.

    Different predicted classes are evaluated independently (multi-label
    foreground regions); wrong-class masks incur FP in their predicted class
    and do not rescue FN in the true class. Duplicate same-class masks do not
    multiply pixel counts.
    """
    height, width = _shape(height, width)
    if len(predictions) != len(mask_rles):
        raise ValueError("Each prediction requires exactly one mask")
    if isinstance(n_classes, bool) or int(n_classes) != n_classes or n_classes < 1:
        raise ValueError("n_classes must be positive")
    by_class = defaultdict(list)
    for prediction, rle in zip(predictions, mask_rles):
        class_id = prediction.get("class_id")
        if (isinstance(class_id, bool) or not isinstance(class_id, (int, np.integer))
                or not 0 <= class_id < n_classes):
            raise ValueError("Predicted class ID outside the declared class map")
        _native_rle(rle, height, width)
        by_class[int(class_id)].append(rle)
    counts = np.zeros((int(n_classes), 3), dtype=np.int64)
    for class_id in sorted(set(by_class) | set(gt_by_class)):
        if not isinstance(class_id, (int, np.integer)) or not 0 <= class_id < n_classes:
            raise ValueError("Ground-truth class ID outside the declared class map")
        predicted = decode_rle(_merge_rles(by_class[class_id], height, width))
        truth = gt_by_class.get(class_id)
        if truth is None:
            target = np.zeros((height, width), dtype=bool)
            ignore = target
        else:
            target = decode_rle(truth["target"], height, width)
            ignore = decode_rle(truth["ignore"], height, width) & ~target
        predicted = predicted & ~ignore
        counts[class_id, 0] = np.count_nonzero(predicted & target)
        counts[class_id, 1] = np.count_nonzero(predicted & ~target)
        counts[class_id, 2] = np.count_nonzero(target & ~predicted)
    return counts


def _find_vendor_root(vendor_dir):
    vendor_dir = Path(vendor_dir).resolve()
    if (vendor_dir / "segment_anything" / "__init__.py").is_file():
        return vendor_dir
    candidates = sorted(path for path in vendor_dir.glob("*")
                        if (path / "segment_anything" / "__init__.py").is_file())
    if len(candidates) > 1:
        raise ValueError("Multiple vendored SAM sources found; choose one pinned source")
    return candidates[0] if candidates else None


class SamBoxPredictor:
    """Frozen official SAM box predictor, with image features reused by callers."""

    def __init__(self, checkpoint, vendor_dir, model_type="vit_b",
                 device="auto", batch_size=16):
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint is missing: {self.checkpoint}. "
                "Run prepare_environment.py first.")
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("SAM batch_size must be a positive integer")
        self.batch_size = int(batch_size)
        self.model_type = model_type
        vendor_root = _find_vendor_root(vendor_dir)
        if vendor_root is not None:
            loaded = sys.modules.get("segment_anything")
            if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(vendor_root):
                raise RuntimeError("A different SAM package is already imported; restart Python")
            sys.path.insert(0, str(vendor_root))
        try:
            self.torch = importlib.import_module("torch")
            sam_module = importlib.import_module("segment_anything")
        except ImportError as exc:
            raise RuntimeError(
                "SAM dependencies are unavailable. Run prepare_environment.py "
                "using this Python interpreter.") from exc
        self.package_path = str(Path(sam_module.__file__).resolve())
        selected_device = ("cuda:0" if self.torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device = self.torch.device(selected_device)
        if self.device.type == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if model_type not in sam_module.sam_model_registry:
            raise ValueError(f"Unknown SAM model type: {model_type}")
        model = sam_module.sam_model_registry[model_type](checkpoint=str(self.checkpoint))
        model = model.float().to(device=self.device).eval()
        self.predictor = sam_module.SamPredictor(model)
        self.image_shape = None
        self.n_image_encodes = 0

    def _synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def set_image(self, rgb):
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("SAM requires an RGB uint8 image of shape (H,W,3)")
        self._synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            self.predictor.set_image(image, image_format="RGB")
        self._synchronize()
        self.image_shape = image.shape[:2]
        self.n_image_encodes += 1
        return time.perf_counter() - started

    def predict(self, boxes):
        if self.image_shape is None:
            raise RuntimeError("Call set_image before predict")
        values = np.asarray(boxes, dtype=np.float64)
        if values.size == 0:
            return [], 0.0
        if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
            raise ValueError("Boxes must be finite Nx4 xyxy coordinates")
        height, width = self.image_shape
        if (np.any(values[:, 0] < 0) or np.any(values[:, 1] < 0)
                or np.any(values[:, 2] > width) or np.any(values[:, 3] > height)
                or np.any(values[:, 0] >= values[:, 2])
                or np.any(values[:, 1] >= values[:, 3])):
            raise ValueError("Boxes must be positive-area and within the original image")
        self._synchronize()
        started = time.perf_counter()
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = self.torch.as_tensor(
                    values[start:start + self.batch_size],
                    dtype=self.torch.float32, device=self.device)
                transformed = self.predictor.transform.apply_boxes_torch(
                    batch, self.image_shape)
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None, point_labels=None, boxes=transformed,
                    multimask_output=False, return_logits=False)
                array = masks[:, 0].detach().cpu().numpy()
                if array.shape != (len(batch), height, width):
                    raise RuntimeError("SAM returned unexpected mask dimensions")
                outputs.extend(encode_binary_mask(mask) for mask in array)
        self._synchronize()
        return outputs, time.perf_counter() - started


def evaluate_mask_ap(annotation_file, image_ids, predictions, max_dets=100):
    """COCO segmentation AP for the supplied retained prediction sets.

    This is a secondary downstream instance diagnostic, not detector AP used
    to select thresholds. ``max_dets`` is applied per image/category by the
    COCO evaluator, after any upstream per-image prompt budget. The complete
    standard COCO summary is returned; console chatter is suppressed.
    """
    if isinstance(max_dets, bool) or not isinstance(max_dets, int) or max_dets < 1:
        raise ValueError("max_dets must be positive")
    image_ids = sorted(set(int(value) for value in image_ids))
    if not image_ids:
        raise ValueError("Mask AP requires at least one image")
    _mask_api()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_file))
        if not set(image_ids).issubset(coco.imgs):
            raise ValueError("Mask AP image IDs are not in the annotation file")
        selected = set(image_ids)
        rows = []
        for raw in predictions:
            row = copy.deepcopy(raw)
            image_id = int(row["image_id"])
            if image_id not in selected:
                raise ValueError("Prediction belongs to an image outside this AP split")
            if row["category_id"] not in coco.cats:
                raise ValueError("Unknown COCO category ID in mask predictions")
            score = row["score"]
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Mask AP scores must be finite")
            image = coco.imgs[image_id]
            row["segmentation"] = _native_rle(
                row["segmentation"], image["height"], image["width"])
            # ``loadRes`` must derive area from the actual mask, not a bbox.
            row.pop("bbox", None)
            rows.append(row)
        if rows:
            # COCO packages differ in whether ``info`` is assumed to exist.
            coco.dataset.setdefault("info", {})
            result = coco.loadRes(rows)
        else:
            result = COCO()
            result.dataset = {
                "images": [coco.imgs[image_id] for image_id in image_ids],
                "categories": list(coco.cats.values()), "annotations": [],
            }
            result.createIndex()
        evaluator = COCOeval(coco, result, "segm")
        evaluator.params.imgIds = image_ids
        evaluator.params.maxDets = [1, min(10, max_dets), max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]

    def ap_at(iou=None):
        # Shape T,R,K,A,M; area=all and the explicitly requested maxDets.
        selected_precision = precision[:, :, :, 0, -1]
        if iou is not None:
            selected_precision = selected_precision[
                np.isclose(evaluator.params.iouThrs, iou)]
        values = selected_precision[selected_precision > -1]
        return float(values.mean()) if values.size else None

    return {
        "mask_AP50_95": ap_at(), "mask_AP50": ap_at(0.50),
        "mask_AP75": ap_at(0.75), "max_dets": max_dets,
        "n_images": len(image_ids), "n_predictions": len(rows),
        "standard_coco_stats": [float(value) for value in evaluator.stats],
        "standard_coco_stats_maxdets_100_only": max_dets == 100,
        "protocol": "COCO segmentation AP on retained box-prompted masks",
        "iou_thresholds": [float(x) for x in evaluator.params.iouThrs],
    }
