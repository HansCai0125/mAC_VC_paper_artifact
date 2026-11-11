import unittest

from region_eval.direct_metric_comparison import summarize_paired_rows


class DirectMetricComparisonTests(unittest.TestCase):
    def test_union_only_threshold_crossing_is_counted(self):
        rows = [{
            "AC": 0.80,
            "BestIoGT": 0.40,
            "ABO": 0.35,
            "BestPredIoA": 0.90,
            "local_VC": 0.75,
        }]
        summary = summarize_paired_rows(rows)
        self.assertAlmostEqual(summary["gain_mean"], 0.40)
        self.assertAlmostEqual(summary["P_gain_gt_005"], 1.0)
        self.assertAlmostEqual(summary["P_union050_single_below050"], 1.0)
        self.assertAlmostEqual(summary["P_union075_single_below050"], 1.0)

    def test_oversized_single_box_separates_coverage_from_iou(self):
        rows = [{
            "AC": 1.0,
            "BestIoGT": 1.0,
            "ABO": 0.25,
            "BestPredIoA": 0.25,
            "local_VC": 0.40,
        }]
        summary = summarize_paired_rows(rows)
        self.assertAlmostEqual(summary["P_union050_single_below050"], 0.0)
        self.assertAlmostEqual(summary["P_AC050_ABO_below050"], 1.0)
        self.assertAlmostEqual(summary["P_BestIoGT050_ABO_below050"], 1.0)


if __name__ == "__main__":
    unittest.main()

