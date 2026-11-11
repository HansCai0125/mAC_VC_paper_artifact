"""Analytical mask evaluation tests; SAM inference is deliberately not run."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downstream.masks import (CocoMasks, decode_rle, encode_binary_mask,
                              evaluate_class_unions, evaluate_mask_ap,
                              SamBoxPredictor)


HAS_COCO = importlib.util.find_spec("pycocotools") is not None


def rectangle(x1, y1, x2, y2, height=6, width=8):
    result = np.zeros((height, width), dtype=np.uint8)
    result[y1:y2, x1:x2] = 1
    return result


@unittest.skipUnless(HAS_COCO, "pycocotools is not installed")
class MaskTests(unittest.TestCase):
    def setUp(self):
        self.target = rectangle(1, 1, 4, 4)
        self.empty = np.zeros_like(self.target)
        self.gt = {0: {"target": encode_binary_mask(self.target),
                       "ignore": encode_binary_mask(self.empty)}}

    def evaluate(self, classes, masks, gt=None):
        return evaluate_class_unions(
            [{"class_id": c} for c in classes],
            [encode_binary_mask(mask) for mask in masks],
            self.gt if gt is None else gt, 6, 8, 2)

    def test_perfect_and_missing(self):
        np.testing.assert_array_equal(self.evaluate([0], [self.target]),
                                      [[9, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(self.evaluate([], []),
                                      [[0, 0, 9], [0, 0, 0]])

    def test_wrong_class_creates_fp_and_fn(self):
        np.testing.assert_array_equal(self.evaluate([1], [self.target]),
                                      [[0, 0, 9], [0, 9, 0]])

    def test_duplicate_union_does_not_count_twice(self):
        single = self.evaluate([0], [self.target])
        duplicate = self.evaluate([0, 0], [self.target, self.target])
        np.testing.assert_array_equal(single, duplicate)

    def test_overreach_and_partial_coverage(self):
        predicted = rectangle(2, 1, 6, 4)
        np.testing.assert_array_equal(self.evaluate([0], [predicted]),
                                      [[6, 6, 3], [0, 0, 0]])

    def test_crowd_ignore_cannot_remove_positive_target(self):
        ignored = rectangle(2, 0, 7, 5)
        gt = {0: {"target": encode_binary_mask(self.target),
                  "ignore": encode_binary_mask(ignored)}}
        predicted = np.maximum(self.target, ignored)
        np.testing.assert_array_equal(self.evaluate([0], [predicted], gt),
                                      [[9, 0, 0], [0, 0, 0]])
        # A wrong-class prediction does not inherit class 0's ignore region.
        np.testing.assert_array_equal(self.evaluate([1], [predicted], gt),
                                      [[0, 0, 9], [0, 28, 0]])

    def test_rle_round_trip_and_shape_validation(self):
        rle = encode_binary_mask(self.target)
        json.dumps(rle)
        np.testing.assert_array_equal(decode_rle(rle), self.target)
        with self.assertRaises(ValueError):
            evaluate_class_unions([{"class_id": 0}], [rle], self.gt, 8, 6, 2)
        with self.assertRaises(ValueError):
            evaluate_class_unions([], [rle], self.gt, 6, 8, 2)

    def fixture(self, directory, annotations):
        path = Path(directory) / "annotations.json"
        path.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "one.jpg", "height": 6, "width": 8}],
            "categories": [{"id": 7, "name": "object"}],
            "annotations": annotations,
        }), encoding="utf-8")
        return path

    def annotation(self, ann_id, segmentation, crowd=0):
        return {"id": ann_id, "image_id": 1, "category_id": 7,
                "bbox": [1, 1, 3, 3], "area": 9, "iscrowd": crowd,
                "segmentation": segmentation}

    def test_coco_polygon_union_and_ignore_precedence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            positive = self.annotation(1, [[1, 1, 4, 1, 4, 4, 1, 4]])
            ignored = self.annotation(2, encode_binary_mask(rectangle(0, 0, 5, 5)), 1)
            dataset = CocoMasks(self.fixture(directory, [positive, ignored]))
            gt = dataset.ground_truth(1)
            np.testing.assert_array_equal(decode_rle(gt[0]["target"]), self.target)
            self.assertFalse(np.any(decode_rle(gt[0]["target"]) &
                                    decode_rle(gt[0]["ignore"])))
            self.assertEqual(dataset.category_ids, [7])
            self.assertEqual(dataset.audit["ignored_bbox_fallback_count"], 0)

    def test_bad_positive_fails_but_ignored_fallback_is_counted_once(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            dataset = CocoMasks(self.fixture(directory, [self.annotation(1, [])]))
            with self.assertRaises(ValueError):
                dataset.ground_truth(1)
            dataset = CocoMasks(self.fixture(directory, [self.annotation(1, [], 1)]))
            dataset.ground_truth(1)
            dataset.ground_truth(1)
            self.assertEqual(dataset.audit["ignored_bbox_fallback_count"], 1)

    def test_mask_ap_perfect_empty_and_explicit_budget(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = self.fixture(directory, [self.annotation(
                1, [[1, 1, 4, 1, 4, 4, 1, 4]])])
            prediction = {"image_id": 1, "category_id": 7,
                          "segmentation": encode_binary_mask(self.target), "score": 0.9}
            perfect = evaluate_mask_ap(path, [1], [prediction])
            self.assertAlmostEqual(perfect["mask_AP50_95"], 1.0)
            self.assertAlmostEqual(perfect["mask_AP50"], 1.0)
            self.assertAlmostEqual(evaluate_mask_ap(path, [1], [], 100)["mask_AP50_95"], 0.0)
            self.assertAlmostEqual(evaluate_mask_ap(path, [1], [prediction], 20)["mask_AP50_95"], 1.0)


class SamAdapterTests(unittest.TestCase):
    """Test API isolation and batching without importing torch or SAM."""

    def test_box_only_batched_adapter_keeps_full_image_masks(self):
        calls = []

        class Tensor:
            def __init__(self, array):
                self.array = np.asarray(array)

            def __len__(self):
                return len(self.array)

            def __getitem__(self, key):
                return Tensor(self.array[key])

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.array

        class Model:
            def float(self):
                calls.append("float32")
                return self

            def to(self, device):
                calls.append(device.type)
                return self

            def eval(self):
                calls.append("eval")
                return self

        class Predictor:
            def __init__(self, model):
                self.transform = SimpleNamespace(
                    apply_boxes_torch=lambda boxes, shape: boxes)

            def set_image(self, image, image_format):
                self.shape = image.shape[:2]
                calls.append(("set_image", image_format))

            def predict_torch(self, **kwargs):
                calls.append(kwargs)
                batch = len(kwargs["boxes"])
                return Tensor(np.ones((batch, 1, *self.shape), dtype=bool)), None, None

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            device=lambda name: SimpleNamespace(type=name.split(":")[0]),
            inference_mode=nullcontext,
            float32=np.float32,
            as_tensor=lambda values, dtype, device: Tensor(np.asarray(values, dtype=dtype)),
        )
        fake_sam = SimpleNamespace(
            __file__=str(Path(__file__).resolve()),
            sam_model_registry={"vit_b": lambda checkpoint: Model()},
            SamPredictor=Predictor,
        )
        modules = {"torch": fake_torch, "segment_anything": fake_sam}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            checkpoint = Path(directory) / "fake.pth"
            checkpoint.write_bytes(b"unit-test fixture")
            with mock.patch("downstream.masks.encode_binary_mask",
                            side_effect=lambda mask: {"size": list(mask.shape),
                                                      "pixel_count": int(mask.sum())}), mock.patch(
                    "downstream.masks.importlib.import_module",
                    side_effect=lambda name: modules[name]):
                predictor = SamBoxPredictor(checkpoint, Path(directory) / "absent", batch_size=2)
                with self.assertRaises(RuntimeError):
                    predictor.predict([[0, 0, 1, 1]])
                with self.assertRaises(ValueError):
                    predictor.set_image(np.zeros((4, 6, 3), dtype=float))
                predictor.set_image(np.zeros((4, 6, 3), dtype=np.uint8))
                masks, elapsed = predictor.predict([[1, 1, 2, 2]] * 5)
                self.assertEqual([mask["pixel_count"] for mask in masks], [24] * 5)
                self.assertEqual(predictor.n_image_encodes, 1)
                self.assertGreaterEqual(elapsed, 0)
                self.assertEqual(predictor.predict([]), ([], 0.0))
                with self.assertRaises(ValueError):
                    predictor.predict([[0, 0, 7, 2]])
        prediction_calls = [call for call in calls if isinstance(call, dict)]
        self.assertEqual([len(call["boxes"]) for call in prediction_calls], [2, 2, 1])
        self.assertEqual(sum(isinstance(call, tuple) for call in calls), 1)
        for call in prediction_calls:
            self.assertIsNone(call["point_coords"])
            self.assertIsNone(call["point_labels"])
            self.assertFalse(call["multimask_output"])
            self.assertFalse(call["return_logits"])
        self.assertIn("float32", calls)
        self.assertIn("cpu", calls)
        self.assertIn("eval", calls)


if __name__ == "__main__":
    unittest.main()
