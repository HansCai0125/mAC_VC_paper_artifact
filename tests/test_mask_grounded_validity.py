import unittest

from region_eval.mask_grounded_validity import (
    foreground_metrics,
    rasterize_coco_polygon,
)


class MaskGroundedValidityTests(unittest.TestCase):
    def setUp(self):
        self.image = {"id": 1, "width": 10, "height": 10}
        self.annotation = {
            "id": 7,
            "image_id": 1,
            "category_id": 1,
            "area": 64.0,
            "bbox": [1.0, 1.0, 7.0, 7.0],
            "iscrowd": 0,
            "segmentation": [[1, 1, 8, 1, 8, 8, 1, 8]],
        }

    @staticmethod
    def prediction(identifier, xyxy):
        return {
            "id": identifier,
            "class_id": 1,
            "score": 0.9,
            "xyxy": xyxy,
        }

    def test_two_complementary_boxes_create_collective_gain(self):
        mask = rasterize_coco_polygon(self.annotation, self.image)
        metrics = foreground_metrics(
            mask,
            [
                self.prediction("left", [1, 1, 5, 9]),
                self.prediction("right", [5, 1, 9, 9]),
            ],
            image_width=10,
            image_height=10,
        )
        self.assertAlmostEqual(metrics["MaskAC"], 1.0)
        self.assertAlmostEqual(metrics["MaskBestCoverage"], 0.5)
        self.assertAlmostEqual(metrics["MaskCollectiveGain"], 0.5)
        self.assertEqual(metrics["mask_marginal_contributor_count"], 2)

    def test_single_box_has_zero_collective_gain(self):
        mask = rasterize_coco_polygon(self.annotation, self.image)
        metrics = foreground_metrics(
            mask,
            [self.prediction("left", [1, 1, 5, 9])],
            image_width=10,
            image_height=10,
        )
        self.assertAlmostEqual(metrics["MaskAC"], 0.5)
        self.assertAlmostEqual(metrics["MaskBestCoverage"], 0.5)
        self.assertAlmostEqual(metrics["MaskCollectiveGain"], 0.0)


if __name__ == "__main__":
    unittest.main()
