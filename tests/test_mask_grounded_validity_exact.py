import unittest

import numpy as np

from region_eval.mask_grounded_validity_exact import (
    foreground_metrics_exact,
    geometry_from_annotation,
    polygon_rectangle_area,
    polygon_rectangle_union_area,
    rectangle_union_cells,
)


class ExactMaskGroundedValidityTests(unittest.TestCase):
    @staticmethod
    def geometry(segmentation, area):
        return geometry_from_annotation({
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "area": area,
            "bbox": [0, 0, 10, 10],
            "iscrowd": 0,
            "segmentation": segmentation,
        })

    @staticmethod
    def prediction(identifier, rectangle):
        return {
            "id": identifier,
            "class_id": 1,
            "score": 0.9,
            "xyxy": rectangle,
        }

    def test_square_clips_to_exact_half(self):
        geometry = self.geometry([[0, 0, 8, 0, 8, 8, 0, 8]], 64.0)
        self.assertAlmostEqual(
            polygon_rectangle_area(geometry, (0, 0, 4, 8)), 32.0
        )

    def test_concave_polygon_split_by_clip_is_exact(self):
        # U shape: outer area 16 minus a 2-by-3 notch = 10.
        geometry = self.geometry([
            [0, 0, 4, 0, 4, 4, 3, 4, 3, 1, 1, 1, 1, 4, 0, 4]
        ], 10.0)
        # The upper clip contains two disjoint 1-by-2 arms.
        self.assertAlmostEqual(
            polygon_rectangle_area(geometry, (0, 2, 4, 4)), 4.0
        )

    def test_overlapping_rectangle_union_is_not_double_counted(self):
        geometry = self.geometry([[0, 0, 8, 0, 8, 8, 0, 8]], 64.0)
        area = polygon_rectangle_union_area(
            geometry, [(0, 0, 5, 8), (3, 0, 8, 8)]
        )
        self.assertAlmostEqual(area, 64.0)
        cells = rectangle_union_cells([(0, 0, 5, 8), (3, 0, 8, 8)])
        self.assertAlmostEqual(
            sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in cells),
            64.0,
        )

    def test_two_complementary_predictions_create_exact_gain(self):
        geometry = self.geometry([[0, 0, 8, 0, 8, 8, 0, 8]], 64.0)
        metrics = foreground_metrics_exact(
            geometry,
            [
                self.prediction("left", [0, 0, 4, 8]),
                self.prediction("right", [4, 0, 8, 8]),
            ],
            image_width=10,
            image_height=10,
        )
        self.assertAlmostEqual(metrics["MaskAC"], 1.0)
        self.assertAlmostEqual(metrics["MaskBestCoverage"], 0.5)
        self.assertAlmostEqual(metrics["MaskCollectiveGain"], 0.5)
        self.assertEqual(metrics["mask_marginal_contributor_count"], 2)

    def test_annotation_area_audit_is_continuous(self):
        geometry = self.geometry([[0.2, 0.3, 5.7, 0.3, 5.7, 2.3, 0.2, 2.3]], 11.0)
        self.assertTrue(np.isclose(geometry.area, geometry.coco_area))


if __name__ == "__main__":
    unittest.main()
