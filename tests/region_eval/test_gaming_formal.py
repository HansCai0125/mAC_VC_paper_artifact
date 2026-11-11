import math
import unittest

from region_eval.gaming_formal import (
    DIAGNOSTIC_AP_LEVELS,
    DUPLICATE_FACTORS,
    EXPAND_SCALES,
    METRIC_FIELDS,
    SHRINK_SCALES,
    _aggregate,
)


def result(dataset, seed, value):
    row = dict(
        dataset=dataset,
        model="yolov8n",
        seed=seed,
        model_id=f"model_{seed}",
        uncertainty_unit=(
            "fixed checkpoint" if dataset == "mscoco" else "training seed"),
        manipulation="shrink",
        variant="center_scale",
        level=.71,
        level_label="side scale 0.71",
    )
    row.update({field: value for field in METRIC_FIELDS})
    return row


class FormalGamingTests(unittest.TestCase):
    def test_protocol_keeps_interpretable_legacy_anchors(self):
        self.assertIn(.71, SHRINK_SCALES)
        self.assertIn(1.41, EXPAND_SCALES)
        self.assertEqual(DUPLICATE_FACTORS, (1, 2, 4, 8, 16))
        self.assertIn(("baseline", "unmodified", 1.0), DIAGNOSTIC_AP_LEVELS)

    def test_aggregate_uses_visdrone_sample_standard_deviation(self):
        rows = [result("visdrone", seed, float(seed + 1)) for seed in range(5)]
        aggregate = _aggregate(rows)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["n_runs"], 5)
        self.assertEqual(aggregate[0]["VC_mean"], 3.0)
        self.assertAlmostEqual(aggregate[0]["VC_std"], math.sqrt(2.5))

    def test_aggregate_keeps_coco_fixed_checkpoint_point(self):
        aggregate = _aggregate([result("mscoco", 0, .4)])
        self.assertEqual(aggregate[0]["n_runs"], 1)
        self.assertEqual(aggregate[0]["VC_mean"], .4)
        self.assertEqual(aggregate[0]["VC_std"], 0.0)
        self.assertEqual(
            aggregate[0]["uncertainty"], "fixed checkpoint point estimate")

    def test_aggregate_rejects_incomplete_visdrone_seed_group(self):
        with self.assertRaises(ValueError):
            _aggregate([result("visdrone", seed, .4) for seed in range(4)])


if __name__ == "__main__":
    unittest.main()
