import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from region_eval.coco import (
    COCO_NAMES,
    filter_coco_predictions,
    prepare_coco_image,
    select_coco_images,
)


class CocoProtocolTests(unittest.TestCase):
    def test_selection_mapping_and_same_class_crowd_filter(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            (root / "images" / "val").mkdir(parents=True)
            images = [
                dict(id=1, file_name="000001.jpg", width=100, height=100),
                dict(id=2, file_name="000002.jpg", width=100, height=100),
            ]
            for row in images:
                (root / "images" / "val" / row["file_name"]).touch()
            categories = [
                dict(id=index + 1, name=name) for index, name in enumerate(COCO_NAMES)
            ]
            annotations = [
                dict(id=10, image_id=1, category_id=1,
                     bbox=[20, 0, 10, 10], area=100, iscrowd=0),
                dict(id=11, image_id=1, category_id=1,
                     bbox=[0, 0, 10, 10], area=100, iscrowd=1),
            ]
            (root / "annotations" / "instances_val2017.json").write_text(
                json.dumps(dict(
                    images=images,
                    annotations=annotations,
                    categories=categories,
                )),
                encoding="utf-8",
            )
            selected, names, category_map, _ = select_coco_images(
                root, count=2, seed=0)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(names), 2)
            image_row, rows = next(row for row in selected if row[0]["id"] == 1)
            gts, crowd, audit = prepare_coco_image(
                image_row, rows, category_map)
            self.assertEqual(gts[0]["class_id"], 0)
            self.assertEqual(audit["n_ignored_annotations"], 1)
            predictions = [
                dict(id="ignored", class_id=0, score=.9, xyxy=[0, 0, 10, 10]),
                dict(id="kept", class_id=0, score=.9, xyxy=[20, 0, 30, 10]),
                dict(id="other_class", class_id=1, score=.9, xyxy=[0, 0, 10, 10]),
            ]
            kept, pred_audit = filter_coco_predictions(
                predictions, crowd, 100, 100)
            self.assertEqual(
                [row["id"] for row in kept], ["kept", "other_class"])
            self.assertEqual(
                pred_audit["dropped_pred_in_crowd"][0]["id"], "ignored")

    def test_invalid_category_order_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            (root / "images" / "val").mkdir(parents=True)
            (root / "images" / "val" / "a.jpg").touch()
            value = dict(
                images=[dict(id=1, file_name="a.jpg", width=10, height=10)],
                annotations=[],
                categories=[dict(id=1, name="wrong")],
            )
            (root / "annotations" / "instances_val2017.json").write_text(
                json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                select_coco_images(root, 1, 0)


if __name__ == "__main__":
    unittest.main()
