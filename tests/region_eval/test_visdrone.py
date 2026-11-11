from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from region_eval.visdrone import (filter_predictions, ignore_fraction, parse_annotations,
                                  prepare_ground_truth, select_images)


class VisDroneProtocolTests(unittest.TestCase):
    def test_cache_bytes_match_recorded_source_on_windows(self):
        from region_eval.visdrone_run import _save_json
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            data = _save_json(path, {"image_id": "test", "rows": [1, 2]})
            self.assertEqual(path.read_bytes(), data)
            with self.assertRaises(FileExistsError):
                _save_json(path, {})

    def test_mapping_and_ignored_prediction_not_false_background(self):
        rows = parse_annotations("0,0,10,10,0,0,0,0\n20,0,10,10,1,4,0,2\n40,0,10,10,0,11,0,0\n")
        gts, ignores, audit = prepare_ground_truth(rows, 100, 100)
        self.assertEqual([g["class_id"] for g in gts], [3])
        self.assertEqual(audit["n_others"], 1)
        preds = [dict(id="ignored", xyxy=[0, 0, 10, 10], class_id=3, score=.9),
                 dict(id="valid", xyxy=[20, 0, 30, 10], class_id=3, score=.9),
                 dict(id="others_location", xyxy=[40, 0, 50, 10], class_id=3, score=.9)]
        kept, audit = filter_predictions(preds, ignores, 100, 100)
        self.assertEqual([p["id"] for p in kept], ["valid", "others_location"])
        self.assertEqual(audit["dropped_pred_in_ignore"][0]["id"], "ignored")

    def test_union_overlap_threshold_equality_and_retained_whole_box(self):
        rect = [0, 0, 10, 10]
        ignores = [(0, 0, 3, 10), (2, 0, 5, 10)]
        self.assertEqual(ignore_fraction(rect, ignores), .5)
        self.assertEqual(ignore_fraction(rect, ignores * 10), .5)
        pred = dict(id="p", xyxy=rect, class_id=0, score=.9)
        kept, _ = filter_predictions([pred], ignores, 100, 100)
        self.assertEqual(kept, [])
        kept, _ = filter_predictions([pred], [(0, 0, 4.9, 10)], 100, 100)
        self.assertEqual(kept[0]["xyxy"], rect)
        # GT obeys the same >= threshold as predictions.
        rows = parse_annotations("0,0,5,10,0,0,0,0\n0,0,10,10,1,1,0,0")
        gts, _, audit = prepare_ground_truth(rows, 100, 100)
        self.assertEqual(gts, [])
        self.assertEqual(len(audit["dropped_gt_in_ignore"]), 1)

    def test_clipped_denominator_and_invalid_records(self):
        pred = dict(id="p", xyxy=[-10, 0, 10, 10], class_id=0, score=.9)
        kept, audit = filter_predictions([pred], [(0, 0, 5, 10)], 100, 100)
        self.assertEqual(kept, [])
        self.assertEqual(audit["clipped_prediction_ids"], ["p"])
        kept, audit = filter_predictions([dict(pred, xyxy=[-10, 0, -1, 10])], [], 100, 100)
        self.assertEqual(kept, [])
        self.assertEqual(audit["invalid_predictions"][0]["reason"], "degenerate_after_clipping")
        with self.assertRaises(ValueError):
            filter_predictions([dict(pred, xyxy=[10, 0, 0, 10])], [], 100, 100)
        with self.assertRaises(ValueError):
            filter_predictions([pred, pred], [], 100, 100)

    def test_raw_annotation_validation(self):
        for text in ["0,0,10,10,0,1,0,0", "0,0,0,10,1,1,0,0",
                     "0,0,10,10,1,12,0,0", "nan,0,10,10,1,1,0,0", "0,0,10"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_annotations(text)
        self.assertEqual(parse_annotations("\n 0,0,1,1,1,10,1,2, \n")[0]["category"], 10)

    def test_selection_is_reproducible_nested_and_complete_pairs_required(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images, annotations = root / "images", root / "annotations"
            images.mkdir()
            annotations.mkdir()
            for i in reversed(range(20)):
                (images / f"{i:03d}.jpg").touch()
                (annotations / f"{i:03d}.txt").touch()
            subset, names = select_images(images, annotations, 5, 0)
            larger, _ = select_images(images, annotations, 10, 0)
            self.assertEqual(subset, select_images(images, annotations, 5, 0)[0])
            self.assertEqual(len(names), 20)
            self.assertTrue(set(subset) <= set(larger))
            self.assertNotEqual(subset, select_images(images, annotations, 5, 1)[0])
            with self.assertRaises(ValueError):
                select_images(images, annotations, 21, 0)
            (annotations / "missing_image.txt").touch()
            with self.assertRaises(ValueError):
                select_images(images, annotations, 5, 0)


if __name__ == "__main__":
    unittest.main()
