import unittest

import numpy as np

from region_eval.fixed_checkpoint_analysis import (
    BOOTSTRAP_METRICS,
    CONTRIBUTION_FIELDS,
    _matching_contributions,
    _new_contributions,
    bootstrap_models,
    metrics_from_totals,
)


class FixedCheckpointAnalysisTests(unittest.TestCase):
    def test_metric_ratios_from_additive_totals(self):
        values = {field: 0.0 for field in CONTRIBUTION_FIELDS}
        values.update(
            n_gt=2,
            ac_sum=1.5,
            responsibility_coverage_sum=1.4,
            responsibility_precision_sum=1.2,
            responsibility_dice_sum=1.3,
            coverage_hit_050=1,
            n_pred=3,
            n_orphan=1,
            standard_tp=1,
            standard_fp=2,
            standard_fn=1,
            lrp_numerator=2.5,
        )
        totals = np.asarray([values[field] for field in CONTRIBUTION_FIELDS])
        metrics = metrics_from_totals(totals)
        self.assertAlmostEqual(float(metrics["VC"]), 1.3 / 3.0)
        self.assertAlmostEqual(float(metrics["mAC"]), 0.75)
        self.assertAlmostEqual(float(metrics["standard_F1"]), 0.4)
        self.assertAlmostEqual(float(metrics["standard_LRP"]), 2.5 / 4.0)

    def test_matching_contributions_preserve_image_unit(self):
        contributions = _new_contributions(["a", "b"])
        contributions["a"]["n_gt"] = 1
        contributions["b"]["n_gt"] = 1
        document = {
            "images": [
                {
                    "image_id": "a",
                    "gts": [
                        {"id": "g1", "class_id": 0,
                         "xyxy": [0.0, 0.0, 10.0, 10.0]},
                    ],
                    "predictions": [
                        {"id": "p1", "class_id": 0, "score": 0.9,
                         "xyxy": [0.0, 0.0, 10.0, 10.0]},
                        {"id": "p2", "class_id": 0, "score": 0.8,
                         "xyxy": [20.0, 20.0, 30.0, 30.0]},
                    ],
                },
                {
                    "image_id": "b",
                    "gts": [
                        {"id": "g2", "class_id": 1,
                         "xyxy": [0.0, 0.0, 5.0, 5.0]},
                    ],
                    "predictions": [],
                },
            ],
        }
        _matching_contributions(contributions, document)
        self.assertEqual(contributions["a"]["standard_tp"], 1)
        self.assertEqual(contributions["a"]["standard_fp"], 1)
        self.assertEqual(contributions["a"]["standard_fn"], 0)
        self.assertAlmostEqual(contributions["a"]["lrp_numerator"], 1.0)
        self.assertEqual(contributions["b"]["standard_fn"], 1)
        self.assertAlmostEqual(contributions["b"]["lrp_numerator"], 1.0)

    def test_paired_bootstrap_reuses_draws_across_models(self):
        rng = np.random.default_rng(7)
        matrix = rng.uniform(size=(12, len(CONTRIBUTION_FIELDS)))
        matrix[:, 0] += 1.0
        matrix[:, 6] += 1.0
        matrix[:, 8:11] += 1.0
        samples = bootstrap_models(
            {"first": matrix, "second": matrix.copy()},
            replicates=200,
            seed=19,
            batch=17,
        )
        for metric in BOOTSTRAP_METRICS:
            np.testing.assert_array_equal(
                samples["first"][metric], samples["second"][metric])

    def test_bootstrap_is_reproducible(self):
        matrix = np.ones((5, len(CONTRIBUTION_FIELDS)), dtype=np.float64)
        first = bootstrap_models(
            {"model": matrix}, replicates=100, seed=5, batch=13)
        second = bootstrap_models(
            {"model": matrix}, replicates=100, seed=5, batch=29)
        for metric in BOOTSTRAP_METRICS:
            np.testing.assert_array_equal(
                first["model"][metric], second["model"][metric])


if __name__ == "__main__":
    unittest.main()