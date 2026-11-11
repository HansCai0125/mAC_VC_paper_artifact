import unittest

from region_eval.gaming import (
    baseline_tp_ids,
    jitter_xyxy,
    manipulate_duplicates,
    manipulate_scale,
    run_gaming_grid,
    scale_xyxy,
    score_document,
)


def document(gts, predictions):
    return dict(
        schema_version=1,
        metadata=dict(
            dataset="synthetic",
            split="test",
            model_id="synthetic",
            checkpoint_id="none",
            coordinate_frame="original_image_xyxy",
            ignore_crowd_policy="already_filtered",
            class_ids=[0],
            class_names=["object"],
            postprocessing=dict(
                stage="synthetic",
                confidence_floor=0.0,
                nms_iou=None,
                max_det=100,
            ),
        ),
        images=[dict(
            image_id="image",
            width=100,
            height=100,
            gts=gts,
            predictions=predictions,
        )],
    )


class GamingTests(unittest.TestCase):
    def test_centre_scaling_and_clipping(self):
        self.assertEqual(
            scale_xyxy(
                [10, 10, 30, 30], .5, 100, 100),
            [15, 15, 25, 25],
        )
        self.assertEqual(
            scale_xyxy(
                [0, 0, 20, 20], 3, 100, 100),
            [0.0, 0.0, 40.0, 40.0],
        )

    def test_only_baseline_true_positive_is_scaled(self):
        value = document(
            [dict(
                id="g", class_id=0,
                xyxy=[10, 10, 30, 30])],
            [
                dict(
                    id="tp", class_id=0, score=.9,
                    xyxy=[10, 10, 30, 30]),
                dict(
                    id="fp", class_id=0, score=.8,
                    xyxy=[60, 60, 70, 70]),
            ],
        )
        targets = baseline_tp_ids(value)
        self.assertEqual(
            targets, {("image", "tp")})
        changed, audit = manipulate_scale(
            value, targets, .5)
        boxes = {
            row["id"]: row["xyxy"]
            for row in changed["images"][0]["predictions"]
        }
        self.assertEqual(
            boxes["tp"], [15, 15, 25, 25])
        self.assertEqual(
            boxes["fp"], [60, 60, 70, 70])
        self.assertEqual(
            audit["modified_prediction_count"], 1)

    def test_exact_duplicate_keeps_vc_and_reduces_f1(self):
        value = document(
            [dict(
                id="g", class_id=0,
                xyxy=[10, 10, 30, 30])],
            [dict(
                id="tp", class_id=0, score=.9,
                xyxy=[10, 10, 30, 30])],
        )
        targets = baseline_tp_ids(value)
        baseline = score_document(value)
        changed, audit = manipulate_duplicates(
            value, targets, 3, jitter=False)
        duplicated = score_document(changed)
        self.assertEqual(
            duplicated["VC"], baseline["VC"])
        self.assertLess(
            duplicated["F1"], baseline["F1"])
        self.assertEqual(
            audit["added_prediction_count"], 2)
        scores = [
            row["score"]
            for row in changed["images"][0]["predictions"]
        ]
        self.assertEqual(scores, [.9, .9, .9])

    def test_sparse_diagnostic_ap_normalization_accepts_blank_rows(self):
        value = document(
            [dict(
                id="g", class_id=0,
                xyxy=[10, 10, 30, 30])],
            [dict(
                id="tp", class_id=0, score=.9,
                xyxy=[10, 10, 30, 30])],
        )
        rows, _ = run_gaming_grid(
            value,
            shrink_scales=(1.0, .71),
            expand_scales=(),
            duplicate_factors=(),
            threshold_grid=(),
            diagnostic_ap_levels=(("baseline", "unmodified", 1.0),),
        )
        self.assertIsNotNone(rows[0]["AP50_95"])
        self.assertIsNone(rows[1]["AP50_95"])
        self.assertIsNone(rows[1]["normalized_AP50_95"])

    def test_jitter_is_deterministic_and_inside_image(self):
        first = jitter_xyxy(
            [90, 90, 100, 100], 1, 100, 100)
        second = jitter_xyxy(
            [90, 90, 100, 100], 1, 100, 100)
        self.assertEqual(first, second)
        self.assertTrue(0.0 <= first[0] < first[2] <= 100.0)
        self.assertTrue(0.0 <= first[1] < first[3] <= 100.0)
        self.assertAlmostEqual(first[2] - first[0], 10.1)
        self.assertAlmostEqual(first[3] - first[1], 9.9)


if __name__ == "__main__":
    unittest.main()
