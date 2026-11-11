import math
import unittest

from region_eval.vc_incremental_value import (
    equal_mac_expansion_rows,
    orphan_injection_rows,
    responsibility_competition_rows,
)


class VCIncrementalValueTests(unittest.TestCase):
    def test_expansion_holds_mac_constant_and_penalizes_excess(self):
        rows = equal_mac_expansion_rows()
        expanded = [row for row in rows if row["scale"] >= 1.0]
        self.assertTrue(all(math.isclose(row["mAC"], 1.0) for row in expanded))
        self.assertTrue(all(
            first["VC"] > second["VC"]
            for first, second in zip(expanded, expanded[1:])
        ))

    def test_orphan_penalty_is_dataset_level(self):
        rows = orphan_injection_rows()
        self.assertTrue(all(math.isclose(row["mAC"], 1.0) for row in rows))
        self.assertTrue(all(math.isclose(row["mean_local_VC"], 1.0) for row in rows))
        for row in rows:
            count = row["added_orphan_count"]
            self.assertAlmostEqual(row["VC"], 1.0 / (1.0 + count))

    def test_shared_prediction_receives_one_owner(self):
        rows = responsibility_competition_rows()
        shared = [row for row in rows if row["variant"] == "single_shared_owner"]
        separate = [row for row in rows if row["variant"] == "separate_owners"]
        self.assertTrue(all(math.isclose(row["mAC"], 1.0) for row in rows))
        self.assertTrue(all(math.isclose(row["VC"], 1.0) for row in separate))
        for row in shared:
            self.assertAlmostEqual(row["VC"], 1.0 / (3.0 + row["gap_ratio"]))


if __name__ == "__main__":
    unittest.main()

