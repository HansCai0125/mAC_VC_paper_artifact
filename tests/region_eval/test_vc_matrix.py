import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from region_eval.vc_matrix import (
    _atomic_json_with_retry,
    _replace_with_retry,
    aggregate_rows,
    summarize_primary_gt,
)


class VCMatrixTests(unittest.TestCase):
    def test_atomic_json_retries_transient_windows_file_lock(self):
        with (
            patch(
                "region_eval.vc_matrix._original_atomic_json",
                side_effect=[PermissionError(13, "locked"), None],
            ) as writer,
            patch("region_eval.vc_matrix.sleep") as wait,
        ):
            _atomic_json_with_retry(Path("progress.json"), {"done": 750})
        self.assertEqual(writer.call_count, 2)
        wait.assert_called_once_with(0.05)

    def test_atomic_json_raises_after_retry_limit(self):
        with (
            patch(
                "region_eval.vc_matrix._original_atomic_json",
                side_effect=PermissionError(13, "locked"),
            ) as writer,
            patch("region_eval.vc_matrix.sleep") as wait,
        ):
            with self.assertRaises(PermissionError):
                _atomic_json_with_retry(
                    Path("progress.json"), {"done": 750}, attempts=3)
        self.assertEqual(writer.call_count, 3)
        self.assertEqual(wait.call_count, 2)

    def test_atomic_replace_retries_without_rewriting_temporary_file(self):
        with (
            patch(
                "region_eval.vc_matrix.os.replace",
                side_effect=[PermissionError(13, "locked"), None],
            ) as replace,
            patch("region_eval.vc_matrix.sleep") as wait,
        ):
            _replace_with_retry(Path("report.tmp"), Path("report.csv"))
        self.assertEqual(replace.call_count, 2)
        wait.assert_called_once_with(0.05)

    def test_primary_gt_summary_counts_collective_cases(self):
        fields = [
            "assigned_pred_count", "responsibility_gain",
            "assigned_multigt_source_count",
        ]
        rows = [
            {"assigned_pred_count": 0, "responsibility_gain": 0,
             "assigned_multigt_source_count": 0},
            {"assigned_pred_count": 1, "responsibility_gain": 0,
             "assigned_multigt_source_count": 1},
            {"assigned_pred_count": 2, "responsibility_gain": .2,
             "assigned_multigt_source_count": 0},
            {"assigned_pred_count": 3, "responsibility_gain": .1,
             "assigned_multigt_source_count": 2},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gt.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            result = summarize_primary_gt(path)
        self.assertEqual(result["n_gt"], 4)
        self.assertEqual(result["missed_gt_count"], 1)
        self.assertEqual(result["multi_assigned_gt_count"], 2)
        self.assertEqual(result["positive_coverage_gain_gt_count"], 2)
        self.assertAlmostEqual(result["positive_coverage_gain_sum"], .3)
        self.assertEqual(result["multigt_source_gt_count"], 2)
        self.assertEqual(result["assigned_prediction_sum"], 6)
        self.assertAlmostEqual(result["multi_assigned_detected_gt_rate"], 2 / 3)

    def test_aggregate_rows_uses_sample_standard_deviation(self):
        rows = [
            {"dataset": "a", "model": "m", "value": 1},
            {"dataset": "a", "model": "m", "value": 2},
            {"dataset": "a", "model": "m", "value": 3},
        ]
        result = aggregate_rows(rows, ("dataset", "model"), ("value",))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["n_seeds"], 3)
        self.assertEqual(result[0]["value_mean"], 2)
        self.assertEqual(result[0]["value_std"], 1)


if __name__ == "__main__":
    unittest.main()
