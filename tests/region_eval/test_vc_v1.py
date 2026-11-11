import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from region_eval.vc_v1 import (
    analyze_vc_document, evaluate_vc, write_vc_analysis,
)


def document(gts, predictions, class_names=None):
    class_names = class_names or ["object"]
    return dict(
        schema_version=1,
        metadata=dict(
            dataset="synthetic",
            split="test",
            model_id="synthetic",
            checkpoint_id="none",
            coordinate_frame="original_image_xyxy",
            ignore_crowd_policy="already_filtered",
            class_ids=list(range(len(class_names))),
            class_names=class_names,
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


class VCV1Tests(unittest.TestCase):
    def test_complementary_fragments_recover_collective_region(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            [dict(id="left", class_id=0, score=.9, xyxy=[0, 0, 5, 10]),
             dict(id="right", class_id=0, score=.8, xyxy=[5, 0, 10, 10])],
        )
        result, rows, _ = evaluate_vc(value)
        self.assertEqual(rows[0]["responsibility_coverage"], 1.0)
        self.assertEqual(rows[0]["responsibility_precision"], 1.0)
        self.assertEqual(result["VC"], 1.0)
        self.assertAlmostEqual(result["standard_F1"], 2 / 3)

    def test_large_cross_gt_box_cannot_rescue_both_targets(self):
        value = document(
            [dict(id="a", class_id=0, xyxy=[0, 0, 10, 10]),
             dict(id="b", class_id=0, xyxy=[10, 0, 20, 10])],
            [dict(id="wide", class_id=0, score=.9, xyxy=[0, 0, 20, 10])],
        )
        result, rows, _ = evaluate_vc(value)
        by_id = {row["gt_id"]: row for row in rows}
        self.assertAlmostEqual(by_id["a"]["responsibility_dice"], 2 / 3)
        self.assertEqual(by_id["b"]["responsibility_dice"], 0.0)
        self.assertAlmostEqual(result["VC"], 1 / 3)

    def test_orphan_prediction_enters_dataset_denominator(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            [dict(id="perfect", class_id=0, score=.9, xyxy=[0, 0, 10, 10]),
             dict(id="orphan", class_id=0, score=.9, xyxy=[20, 20, 21, 21])],
        )
        result, _, _ = evaluate_vc(value)
        self.assertEqual(result["mean_local_VC"], 1.0)
        self.assertEqual(result["orphan_pred_count"], 1)
        self.assertEqual(result["VC"], .5)

    def test_duplicate_is_region_invariant_but_standard_fp_remains(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            [dict(id="first", class_id=0, score=.9, xyxy=[0, 0, 10, 10]),
             dict(id="duplicate", class_id=0, score=.8, xyxy=[0, 0, 10, 10])],
        )
        result, _, _ = evaluate_vc(value)
        self.assertEqual(result["VC"], 1.0)
        self.assertEqual(result["standard_FP"], 1)
        self.assertAlmostEqual(result["standard_F1"], 2 / 3)

    def test_wrong_class_is_orphan_and_miss(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            [dict(id="wrong", class_id=1, score=.9, xyxy=[0, 0, 10, 10])],
            class_names=["first", "second"],
        )
        result, _, _ = evaluate_vc(value)
        self.assertEqual(result["VC"], 0.0)
        self.assertEqual(result["orphan_pred_count"], 1)
        self.assertEqual(result["standard_F1"], 0.0)

    def test_threshold_grid_reuses_same_cache(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            [dict(id="p", class_id=0, score=.2, xyxy=[0, 0, 10, 10])],
        )
        low, _, _ = evaluate_vc(value, theta=.1)
        high, _, _ = evaluate_vc(value, theta=.25)
        self.assertEqual(low["VC"], 1.0)
        self.assertEqual(high["VC"], 0.0)
        analysis = analyze_vc_document(
            value,
            confidence_thresholds=(.1, .25),
            primary_theta=.25,
            coverage_thresholds=(.5,),
        )
        self.assertEqual(len(analysis["threshold_rows"]), 2)

    def test_diagnostic_ap_skips_classes_absent_from_smoke_subset(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            [dict(id="p", class_id=0, score=.9, xyxy=[0, 0, 10, 10])],
            class_names=["present", "absent"],
        )
        analysis = analyze_vc_document(
            value,
            confidence_thresholds=(.25,),
            primary_theta=.25,
            coverage_thresholds=(.5,),
        )
        self.assertEqual(analysis["diagnostic_ap_summary"]["AP50"], 1.0)
        self.assertIsNone(
            analysis["diagnostic_ap_class_rows"][1]["AP50_95"])
    def test_analysis_writer_creates_complete_reusable_report(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            [dict(id="p", class_id=0, score=.9, xyxy=[0, 0, 10, 10])],
        )
        analysis = analyze_vc_document(
            value,
            confidence_thresholds=(.25,),
            primary_theta=.25,
            coverage_thresholds=(.5,),
        )
        with TemporaryDirectory() as directory:
            source = Path(directory) / "predictions.json"
            source.write_text(json.dumps(value), encoding="utf-8")
            output = write_vc_analysis(
                Path(directory) / "analysis",
                document=value,
                source_file=source,
                analysis=analysis,
            )
            self.assertTrue((output / "vc_thresholds.csv").is_file())
            complete = json.loads(
                (output / "analysis_complete.json").read_text(encoding="utf-8"))
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["EPI_role"], "excluded")
    def test_noncanonical_boxes_are_rejected_before_scoring(self):
        value = document(
            [dict(id="g", class_id=0, xyxy=[-1, 0, 10, 10])], [])
        with self.assertRaises(ValueError):
            evaluate_vc(value)


if __name__ == "__main__":
    unittest.main()
