"""Parity and leakage-boundary tests; only tiny synthetic fixtures are used."""

import copy
from pathlib import Path
import sys
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT), str(PROJECT)]

from downstream.selection import calibrate, choose_policies, retain_predictions
from region_eval.vc_v1 import evaluate_vc
from region_eval.offline import evaluate_document


def fixture():
    metadata = dict(
        dataset="synthetic", split="calibration", model_id="fixture",
        checkpoint_id="none", ignore_crowd_policy="already_filtered",
        coordinate_frame="original_image_xyxy", class_ids=[0, 1, 2],
        class_names=["a", "b", "c"],
        postprocessing=dict(stage="synthetic", confidence_floor=0.0, nms_iou=None, max_det=100),
    )

    def gt(identifier, cls, rect):
        return dict(id=identifier, class_id=cls, xyxy=rect)

    def pred(identifier, cls, rect, score):
        return dict(id=identifier, class_id=cls, xyxy=rect, score=score)

    images = [dict(image_id="img1", width=100, height=100,
        gts=[gt("g2", 0, [5, 0, 15, 10]), gt("g1", 0, [0, 0, 10, 10]),
             gt("g3", 1, [60, 60, 80, 80])],
        predictions=[
            pred("p02", 0, [0, 0, 15, 10], .8),
            pred("p01", 0, [0, 0, 5, 10], .9),
            pred("p03", 0, [0, 0, 15, 10], .8),
            pred("p04", 1, [61, 61, 81, 81], .6),
            pred("p05", 2, [60, 60, 80, 80], .5),
            pred("p06", 0, [90, 90, 100, 100], .3),
            pred("p07", 0, [5, 0, 10, 10], .2),
        ]),
        dict(image_id="img2", width=100, height=100,
             gts=[gt("x1", 0, [20, 20, 30, 30])], predictions=[]),
        dict(image_id="img3", width=100, height=100, gts=[],
             predictions=[pred("z", 1, [1, 1, 3, 3], .75)]),
    ]
    return dict(schema_version=1, metadata=metadata, images=images)


class CalibrationTests(unittest.TestCase):
    def test_parity_with_existing_paper_evaluator(self):
        original = fixture()
        before = copy.deepcopy(original)
        results = calibrate(original["images"], original["metadata"],
                            [.05, .25, .6, .8, .95], [1, 2, 4, 20])
        for row in results:
            capped = copy.deepcopy(original)
            for image in capped["images"]:
                image["predictions"] = retain_predictions(image["predictions"], row["theta"], row["budget"])
            vc, _, _ = evaluate_vc(capped, theta=row["theta"])
            _, _, regional = evaluate_document(capped, theta=row["theta"])
            for field, expected in (("F1", vc["standard_F1"]), ("LRP", vc["standard_LRP"]),
                                    ("VC", vc["VC"]), ("mAC", regional["mAC"]),
                                    ("BestIoGT", regional["BestIoGT"])):
                self.assertAlmostEqual(row[field], expected, places=12, msg=f"{field}: {row}")
            self.assertEqual(row["n_pred"], regional["n_pred"])
            self.assertEqual(row["orphan_pred_count"], vc["orphan_pred_count"])
        self.assertEqual(original, before)

    def test_score_tie_and_class_agnostic_cap(self):
        preds = fixture()["images"][0]["predictions"]
        selected = retain_predictions(list(reversed(preds)), .8, 2)
        self.assertEqual([row["id"] for row in selected], ["p01", "p02"])

    def test_policy_selection_minimizes_lrp_and_prefers_highest_tied_theta(self):
        doc = fixture()
        rows = calibrate(doc["images"], doc["metadata"], [.05, .25, .95], [20])
        for row in rows:
            row.update(F1=.5, LRP=.4, BestIoGT=.6, mAC=.7, VC=.5)
        rows[0]["LRP"] = .2
        rows[1]["F1"] = .9
        chosen = {r["policy"]: r for r in choose_policies(rows)}
        self.assertEqual(chosen["fixed_025"]["theta"], .25)
        self.assertEqual(chosen["F1"]["theta"], .25)
        self.assertEqual(chosen["LRP"]["theta"], .05)
        self.assertEqual(chosen["mAC"]["theta"], .95)
        self.assertEqual(chosen["VC"]["theta"], .95)

    def test_empty_predictions_have_misses(self):
        doc = fixture()
        images = [doc["images"][1]]
        row = calibrate(images, doc["metadata"], [.25], [20])[0]
        self.assertEqual((row["mAC"], row["VC"], row["BestIoGT"], row["F1"]), (0, 0, 0, 0))
        self.assertEqual(row["LRP"], 1)

    def test_empty_ground_truth_not_silently_selected(self):
        doc = fixture()
        row = calibrate([doc["images"][2]], doc["metadata"], [.25], [20])[0]
        self.assertIsNone(row["mAC"])
        self.assertEqual(row["VC"], 0)
        with self.assertRaisesRegex(ValueError, "BestIoGT undefined"):
            choose_policies([row])

    def test_scope_is_exactly_the_images_supplied(self):
        doc = fixture()
        cal_images = copy.deepcopy(doc["images"][:2])
        baseline = calibrate(cal_images, doc["metadata"], [.25], [20])
        doc["images"][2]["predictions"] = []  # A held-out image is not an input.
        self.assertEqual(baseline, calibrate(cal_images, doc["metadata"], [.25], [20]))

    def test_progress_and_validation(self):
        doc = fixture()
        calls = []
        calibrate(doc["images"], doc["metadata"], [.25], [20], progress=lambda a, b: calls.append((a, b)))
        self.assertEqual(calls[-1], (3, 3))
        with self.assertRaisesRegex(ValueError, "unique"):
            calibrate(doc["images"] * 2, doc["metadata"], [.25], [20])
        with self.assertRaises(ValueError):
            retain_predictions(doc["images"][0]["predictions"], .25, 0)
        with self.assertRaisesRegex(ValueError, "fixed-threshold"):
            choose_policies(calibrate(doc["images"], doc["metadata"], [.2], [20]))


if __name__ == "__main__":
    unittest.main()
