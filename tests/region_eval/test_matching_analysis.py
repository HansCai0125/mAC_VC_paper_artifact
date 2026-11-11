import unittest

from region_eval.matching_analysis import (
    _counterfactual_documents,
    average_precision,
    counterfactual_metrics,
    harmonic3,
    match_class,
    optimal_lrp,
    summarize_events,
)


def document(gts, predictions):
    return dict(
        metadata=dict(class_ids=[0]),
        images=[dict(
            image_id="image",
            width=100,
            height=100,
            gts=gts,
            predictions=predictions,
        )],
    )


class MatchingAnalysisTests(unittest.TestCase):
    def test_one_to_one_matching_turns_duplicate_into_fp(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            [
                dict(id="p1", class_id=0, score=.9,
                     xyxy=[0, 0, 10, 10]),
                dict(id="p2", class_id=0, score=.8,
                     xyxy=[0, 0, 10, 10]),
            ],
        )
        events, n_gt = match_class(value, 0, .5, .25)
        metrics = summarize_events(events, n_gt, .5)
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]),
                         (1, 1, 0))
        self.assertAlmostEqual(metrics["F1"], 2 / 3)
        self.assertEqual(events[0]["matched_gt_id"], "g")
        self.assertIsNone(events[1]["matched_gt_id"])

    def test_best_unmatched_gt_and_tie_are_deterministic(self):
        value = document(
            [
                dict(id="a", class_id=0, xyxy=[0, 0, 10, 10]),
                dict(id="b", class_id=0, xyxy=[10, 0, 20, 10]),
            ],
            [dict(id="p", class_id=0, score=.9,
                  xyxy=[0, 0, 20, 10])],
        )
        events, _ = match_class(value, 0, .5)
        self.assertEqual(events[0]["matched_gt_id"], "a")

    def test_coco_style_ap_perfect_and_missed(self):
        perfect = document(
            [
                dict(id="g1", class_id=0, xyxy=[0, 0, 10, 10]),
                dict(id="g2", class_id=0, xyxy=[20, 0, 30, 10]),
            ],
            [
                dict(id="p1", class_id=0, score=.9,
                     xyxy=[0, 0, 10, 10]),
                dict(id="p2", class_id=0, score=.8,
                     xyxy=[20, 0, 30, 10]),
            ],
        )
        events, n_gt = match_class(perfect, 0, .5)
        self.assertEqual(average_precision(events, n_gt), 1.0)

        missed = document(
            [
                dict(id="g1", class_id=0, xyxy=[0, 0, 10, 10]),
                dict(id="g2", class_id=0, xyxy=[20, 0, 30, 10]),
            ],
            [dict(id="p1", class_id=0, score=.9,
                  xyxy=[0, 0, 10, 10])],
        )
        events, n_gt = match_class(missed, 0, .5)
        self.assertAlmostEqual(
            average_precision(events, n_gt),
            51 / 101,
        )

    def test_lrp_analytic_components_and_optimum(self):
        events = [
            dict(score=.9, is_tp=True, matched_iou=.75),
            dict(score=.8, is_tp=False, matched_iou=None),
        ]
        metrics = summarize_events(events, 2, .5)
        self.assertAlmostEqual(metrics["LRP"], 2.5 / 3)
        self.assertAlmostEqual(metrics["LRP_localization_error"], .5)
        self.assertAlmostEqual(metrics["LRP_FP_error"], .5)
        self.assertAlmostEqual(metrics["LRP_FN_error"], .5)
        optimum = optimal_lrp(events, 2, .5)
        self.assertEqual(optimum["optimal_LRP_threshold"], .9)
        self.assertAlmostEqual(optimum["oLRP"], .75)
        self.assertEqual(optimum["optimal_F1_threshold"], .9)

    def test_counterfactuals_separate_area_and_count_behavior(self):
        documents = _counterfactual_documents()
        self.assertEqual(
            set(documents),
            {
                "perfect_base", "exact_duplicate", "small_orphan",
                "oversized_boxes", "cross_gt_box",
                "complementary_fragments",
            },
        )
        rows = {row["scenario"]: row for row in counterfactual_metrics()}
        self.assertEqual(rows["perfect_base"]["ResponsibilityDice"], 1.0)
        self.assertEqual(rows["perfect_base"]["ClassUnionDice"], 1.0)
        self.assertEqual(rows["perfect_base"]["F1"], 1.0)
        self.assertEqual(
            rows["exact_duplicate"]["ResponsibilityDice"], 1.0)
        self.assertEqual(rows["exact_duplicate"]["ClassUnionDice"], 1.0)
        self.assertLess(rows["exact_duplicate"]["F1"], 1.0)
        self.assertGreater(rows["small_orphan"]["ClassUnionDice"], .99)
        self.assertLess(rows["small_orphan"]["F1"], 1.0)
        self.assertEqual(
            rows["complementary_fragments"]["ResponsibilityDice"], 1.0)
        self.assertEqual(
            rows["complementary_fragments"]["ClassUnionDice"], 1.0)
        self.assertLess(rows["complementary_fragments"]["F1"], 1.0)
        self.assertLess(rows["cross_gt_box"]["ResponsibilityDice"], .5)

    def test_harmonic3_requires_all_layers(self):
        self.assertEqual(harmonic3(1, 1, 1), 1.0)
        self.assertEqual(harmonic3(1, 1, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
