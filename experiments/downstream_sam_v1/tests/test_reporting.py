"""Synthetic tests; they do not constitute results of the real experiment."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downstream.reporting import POLICIES, bootstrap_miou, build_reports, scores_from_counts


class ReportingTests(unittest.TestCase):
    def test_pooled_macro_iou_and_micro_recall(self):
        result = scores_from_counts(np.array([[9, 1, 0], [1, 0, 9], [0, 0, 0]]))
        self.assertAlmostEqual(result["mIoU"], 0.5)
        self.assertAlmostEqual(result["foreground_precision"], 10 / 11)
        self.assertAlmostEqual(result["foreground_recall"], 10 / 19)

    def test_false_positive_only_class_is_included(self):
        result = scores_from_counts(np.array([[3, 0, 0], [0, 5, 0]]))
        self.assertEqual(result["mIoU"], 0.5)
        self.assertEqual(scores_from_counts(np.zeros((2, 3), dtype=int))["mIoU"], None)

    def test_bootstrap_aggregates_pixels_before_iou(self):
        counts = np.array([[[[9, 1, 0]]], [[[0, 0, 1]]]], dtype=np.int64)
        weights = np.array([[1, 1], [2, 0], [0, 2]])
        result = bootstrap_miou(counts, weights)
        np.testing.assert_allclose(result[:, 0], [9 / 11, 0.9, 0.0])

    @staticmethod
    def _write_run(directory, seed, image_ids=(10, 20), scale=1, score=1.0, vc_offset=0, false_positives=0):
        configs = [dict(run_id=f"test_s{seed}", model="test", seed=seed, budget=10,
                        policy=policy, theta=0.25) for policy in POLICIES]
        counts = np.zeros((len(image_ids), len(configs), 2, 3), dtype=np.int64)
        total = 100 * scale
        counts[:, :, 0, 0] = round(total * score)
        counts[:, :, 0, 2] = total - round(total * score)
        counts[:, :, 0, 1] = false_positives
        if vc_offset:
            vc = POLICIES.index("VC")
            counts[:, vc, 0, 0] += vc_offset
            counts[:, vc, 0, 2] -= vc_offset
        path = directory / f"run_{seed}.npz"
        np.savez_compressed(path, image_ids=np.array(image_ids, dtype=np.int64), counts=counts,
                            prompt_counts=np.ones(counts.shape[:2], dtype=np.int64),
                            config_json=np.array(json.dumps(configs)))
        return path

    @staticmethod
    def _read_rows(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_identical_policies_have_zero_paired_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs = [self._write_run(directory, seed) for seed in range(2)]
            with patch("downstream.reporting._plots", return_value={"status": "unavailable", "files": []}):
                result = build_reports(runs, directory / "reports", bootstrap_reps=30)
            self.assertEqual(result["status"], "complete")
            for row in self._read_rows(result["files"]["paired_differences"]):
                self.assertEqual(float(row["delta_mIoU"]), 0)
                self.assertEqual(float(row["ci95_low"]), 0)
                self.assertEqual(float(row["ci95_high"]), 0)

    def test_seed_metrics_are_averaged_not_pooled(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs = [self._write_run(directory, 0, score=1.0),
                    self._write_run(directory, 1, score=1.0, false_positives=900)]
            with patch("downstream.reporting._plots", return_value={"status": "unavailable", "files": []}):
                result = build_reports(runs, directory / "reports", bootstrap_reps=30)
            rows = self._read_rows(result["files"]["model_summary"])
            self.assertAlmostEqual(float(rows[0]["mIoU_mean"]), 0.55)
            self.assertAlmostEqual(float(rows[0]["mIoU_std"]), np.sqrt(0.405))
            self.assertFalse(result["all_model_groups_have_five_seeds"])

    def test_image_pairing_must_match_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs = [self._write_run(directory, 0), self._write_run(directory, 1, image_ids=(20, 10))]
            with self.assertRaisesRegex(ValueError, "image IDs/order"):
                build_reports(runs, directory / "reports", bootstrap_reps=10)
            self.assertFalse((directory / "reports" / "report_complete.json").exists())

    def test_paired_difference_and_percentage_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs = [self._write_run(directory, 0, score=0.5, vc_offset=10),
                    self._write_run(directory, 1, score=0.5, vc_offset=10)]
            with patch("downstream.reporting._plots", return_value={"status": "unavailable", "files": []}):
                result = build_reports(runs, directory / "reports", bootstrap_reps=40)
            rows = self._read_rows(result["files"]["paired_differences"])
            for row in rows:
                self.assertAlmostEqual(float(row["delta_mIoU"]), 0.1)
                self.assertAlmostEqual(float(row["delta_mIoU_pp"]), 10)
                self.assertAlmostEqual(float(row["ci95_low"]), 0.1)
                self.assertAlmostEqual(float(row["ci95_high"]), 0.1)

    def test_mask_ap_merge_rejects_unknown_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run = self._write_run(directory, 0)
            with self.assertRaisesRegex(ValueError, "do not match"):
                build_reports([run], directory / "reports", bootstrap_reps=10,
                              mask_ap_rows=[dict(run_id="unknown", budget=10, policy="VC",
                                                 mask_AP50_95=0.2, mask_AP50=0.3)])


if __name__ == "__main__":
    unittest.main()
