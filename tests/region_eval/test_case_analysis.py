import unittest

from region_eval.case_analysis import (detail_crop, enrich_case_geometry, harmonic,
                                  local_support_summary, pair_vc,
                                  rank_single_gt_low_overlap_cases, select_diverse)
from region_eval.diagnostics import Config, ImageDiagnostics


class CaseAnalysisTests(unittest.TestCase):
    def test_exploratory_pair_score_includes_prediction_only_zero(self):
        config = Config(theta=.25, class_ids=(0, 1))
        image = ImageDiagnostics(
            "image", 100, 100, config,
            per_gt=[dict(image_id="image", gt_id="g", class_id=0, AC=.8)],
            per_class=[dict(image_id="image", class_id=0, SpatialPrecision=.5),
                       dict(image_id="image", class_id=1, SpatialPrecision=0.0)],
            counts={})
        self.assertAlmostEqual(harmonic(.8, .5), 8 / 13)
        self.assertAlmostEqual(pair_vc([image]), (8 / 13) / 2)

    def test_diverse_selection_is_deterministic_and_pair_unique(self):
        candidates = [
            dict(selection_score=10, image_id="a", class_id=0, gt_id="1"),
            dict(selection_score=9, image_id="a", class_id=0, gt_id="2"),
            dict(selection_score=8, image_id="b", class_id=0, gt_id="1"),
            dict(selection_score=7, image_id="c", class_id=1, gt_id="1"),
            dict(selection_score=6, image_id="d", class_id=2, gt_id="1"),
        ]
        selected = select_diverse(candidates, 4)
        self.assertEqual([(row["image_id"], row["class_id"]) for row in selected],
                         [("a", 0), ("c", 1), ("d", 2), ("b", 0)])
        self.assertEqual(selected, select_diverse(list(reversed(candidates)), 4))

    def test_low_overlap_ranking_excludes_multigt_and_high_overlap(self):
        config = Config(theta=.25, class_ids=(0,))
        pair = dict(image_id="image", class_id=0, SpatialPrecision=.8, ClassUnionDice=.8,
                    ClassUnionIoU=2/3, n_gt=3, n_pred=6, orphan_pred_count=0,
                    multigt_pred_count=1)
        gt_rows = []
        local_rows = []
        for index, (multi, overlap, gain) in enumerate(((0, .1, .3), (1, .1, .4), (0, .6, .5))):
            row = dict(image_id="image", gt_id=f"g{index}", class_id=0, gt_area=100,
                       AC=.9, BestIoGT_i=.9-gain, CollectiveGain_i=gain,
                       coverage_hit=True, intersecting_pred_count=2, best_pred_id="p")
            gt_rows.append(row)
            local_rows.append(dict(row, contributing_pred_count=2,
                                   contributing_multigt_count=multi,
                                   max_contributing_pairwise_iou=overlap,
                                   selected_gt_local_dice=.8))
        image = ImageDiagnostics("image", 100, 100, config, gt_rows, [pair], counts={})
        selected = rank_single_gt_low_overlap_cases([image], local_rows, ["class"], 2)
        self.assertEqual([row["gt_id"] for row in selected], ["g0"])

    def test_detail_crop_contains_gt_and_intersecting_predictions(self):
        gt = dict(xyxy=[100, 100, 120, 120])
        predictions = [dict(xyxy=[90, 90, 115, 115]),
                       dict(xyxy=[900, 900, 950, 950])]
        crop = detail_crop(gt, predictions, 1000, 1000)
        self.assertLessEqual(crop[0], 90)
        self.assertLessEqual(crop[1], 90)
        self.assertGreaterEqual(crop[2], 120)
        self.assertGreaterEqual(crop[3], 120)
        self.assertLess(crop[2], 900)

    def test_local_and_class_precision_expose_neighbor_credit(self):
        case = dict(class_id=0, gt_id="a", AC=1.0)
        image = dict(
            gts=[dict(id="a", class_id=0, xyxy=[0, 0, 10, 10]),
                 dict(id="b", class_id=0, xyxy=[10, 0, 20, 10])],
            predictions=[dict(id="p", class_id=0, score=.9, xyxy=[0, 0, 20, 10])])
        enrich_case_geometry(case, image, .25)
        self.assertEqual(case["contributing_multigt_count"], 1)
        self.assertEqual(case["selected_gt_local_precision"], .5)
        self.assertEqual(case["contributing_class_precision"], 1.0)
        self.assertEqual(case["neighboring_class_credit"], .5)
        self.assertAlmostEqual(case["selected_gt_local_dice"], 2 / 3)
        case["CollectiveGain_i"] = .5
        summary = local_support_summary([case])
        self.assertEqual(summary["positive_CG_any_multigt_contributor"], 1)
        self.assertEqual(summary["positive_CG_neighbor_credit_gt_01"], 1)


if __name__ == "__main__":
    unittest.main()
