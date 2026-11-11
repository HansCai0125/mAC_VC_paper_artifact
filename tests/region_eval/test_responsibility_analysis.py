import unittest

from region_eval.responsibility_analysis import (
    assignment_weights,
    score_document,
    weighted_union_area,
)


class ResponsibilityAnalysisTests(unittest.TestCase):
    def test_weighted_union_matches_hard_union_and_duplicate_invariance(self):
        first = (0, 0, 10, 10)
        second = (5, 0, 15, 10)
        self.assertEqual(
            weighted_union_area([(first, 1.0), (second, 1.0)]),
            150.0,
        )
        self.assertAlmostEqual(
            weighted_union_area([(first, .5), (first, .5)]),
            50.0,
        )
        self.assertEqual(
            weighted_union_area([(first, 1.0)], clip=(2, 2, 8, 8)),
            36.0,
        )

    def test_hard_iou_has_one_deterministic_owner(self):
        gts = [
            dict(id="a", xyxy=[0, 0, 10, 10]),
            dict(id="b", xyxy=[10, 0, 20, 10]),
        ]
        predictions = [dict(id="p", xyxy=[0, 0, 20, 10])]
        weights = assignment_weights(gts, predictions, "hard_iou")
        self.assertEqual(weights, {"p": {"a": 1.0}})

    def test_center_rejects_an_off_center_intersection(self):
        gts = [dict(id="g", xyxy=[0, 0, 10, 10])]
        predictions = [dict(id="p", xyxy=[9, 0, 29, 10])]
        self.assertEqual(
            assignment_weights(gts, predictions, "center"),
            {"p": {}},
        )
        self.assertEqual(
            assignment_weights(gts, predictions, "hard_iou"),
            {"p": {"g": 1.0}},
        )

    def test_soft_overlap_weights_sum_to_one(self):
        gts = [
            dict(id="a", xyxy=[0, 0, 10, 10]),
            dict(id="b", xyxy=[10, 0, 30, 10]),
        ]
        predictions = [dict(id="p", xyxy=[0, 0, 30, 10])]
        weights = assignment_weights(gts, predictions, "soft_overlap")
        self.assertAlmostEqual(weights["p"]["a"], 1 / 3)
        self.assertAlmostEqual(weights["p"]["b"], 2 / 3)
        self.assertAlmostEqual(sum(weights["p"].values()), 1.0)

    def test_fragments_are_preserved_and_cross_gt_box_is_constrained(self):
        fragments = dict(
            metadata=dict(class_ids=[0]),
            images=[dict(
                image_id="fragments",
                width=10,
                height=10,
                gts=[dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
                predictions=[
                    dict(id="left", class_id=0, score=.9,
                         xyxy=[0, 0, 5, 10]),
                    dict(id="right", class_id=0, score=.9,
                         xyxy=[5, 0, 10, 10]),
                ],
            )],
        )
        fragment_rows, _ = score_document(fragments, .25, "hard_iou")
        self.assertEqual(fragment_rows[0]["responsibility_coverage"], 1.0)
        self.assertEqual(fragment_rows[0]["responsibility_precision"], 1.0)
        self.assertEqual(fragment_rows[0]["responsibility_gain"], .5)

        base = dict(
            metadata=dict(class_ids=[0]),
            images=[dict(
                image_id="image",
                width=30,
                height=10,
                gts=[
                    dict(id="a", class_id=0, xyxy=[0, 0, 10, 10]),
                    dict(id="b", class_id=0, xyxy=[10, 0, 20, 10]),
                ],
                predictions=[
                    dict(id="left", class_id=0, score=.9,
                         xyxy=[0, 0, 5, 10]),
                    dict(id="right", class_id=0, score=.9,
                         xyxy=[5, 0, 10, 10]),
                    dict(id="wide", class_id=0, score=.9,
                         xyxy=[0, 0, 20, 10]),
                ],
            )],
        )
        hard_rows, hard_pairs = score_document(base, .25, "hard_iou")
        by_id = {row["gt_id"]: row for row in hard_rows}
        self.assertEqual(by_id["a"]["responsibility_coverage"], 1.0)
        self.assertEqual(by_id["a"]["responsibility_gain"], 0.0)
        self.assertEqual(by_id["b"]["responsibility_coverage"], 0.0)
        self.assertEqual(hard_pairs[0]["multigt_source_pred_count"], 1)
        self.assertEqual(hard_pairs[0]["unassigned_intersecting_pred_count"], 0)

        soft_rows, _ = score_document(base, .25, "soft_overlap")
        soft_by_id = {row["gt_id"]: row for row in soft_rows}
        self.assertEqual(soft_by_id["a"]["responsibility_coverage"], 1.0)
        self.assertAlmostEqual(soft_by_id["b"]["responsibility_coverage"], .5)

    def test_invalid_method_fails(self):
        with self.assertRaises(ValueError):
            assignment_weights([], [], "unknown")


if __name__ == "__main__":
    unittest.main()
