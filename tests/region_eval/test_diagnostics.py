import copy
import csv
import importlib.util
import json
from math import isclose
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from region_eval import Box, Config, aggregate, image_region_diagnostics
from region_eval.geometry import area, intersection, union_area
from region_eval.offline import PROJECT_ROOT, evaluate_document, write_report

TOL = 1e-9


def gt(rect=(0, 0, 10, 10), id="g", cls=0):
    return Box(id, rect, cls)


def pred(rect=(0, 0, 10, 10), id="p", cls=0, score=0.9):
    return Box(id, rect, cls, score)


def score(gs, ps, image_id="image", config=None, size=(100, 100)):
    return image_region_diagnostics(gs, ps, size, image_id=image_id, config=config)


def oracle_area(rects, other=None):
    """Independent coordinate-cell enumeration, not the production x-slab merge."""
    all_rects = rects + (other or [])
    xs = sorted({v for r in all_rects for v in (r[0], r[2])})
    ys = sorted({v for r in all_rects for v in (r[1], r[3])})
    total = 0.0
    for a, b in zip(xs, xs[1:]):
        for c, d in zip(ys, ys[1:]):
            x, y = (a+b)/2, (c+d)/2
            in_first = any(r[0] < x < r[2] and r[1] < y < r[3] for r in rects)
            in_second = other is None or any(r[0] < x < r[2] and r[1] < y < r[3] for r in other)
            if in_first and in_second:
                total += (b-a)*(d-c)
    return total


class GeometryTests(unittest.TestCase):
    def near(self, actual, expected):
        self.assertTrue(isclose(actual, expected, rel_tol=TOL, abs_tol=TOL), (actual, expected))

    def test_analytic_cases(self):
        cases = [
            ("perfect", [gt()], [pred()], (1, 1, 0, 1, 1)),
            ("fragment", [gt()], [pred((0,0,5,10),"a"),pred((5,0,10,10),"b")], (1,.5,.5,1,1)),
            ("overlapping_fragments", [gt()], [pred((0,0,6,10),"a"),pred((4,0,10,10),"b")], (1,.6,.4,1,1)),
            ("expand", [gt()], [pred((0,0,20,10))], (1,1,0,2/3,.5)),
            ("full_image", [gt()], [pred((0,0,100,100))], (1,1,0,2/101,.01)),
            ("translate", [gt()], [pred((5,0,15,10))], (.5,.5,0,.5,1/3)),
            ("undercoverage", [gt()], [pred((0,0,5,10))], (.5,.5,0,2/3,.5)),
            ("orphan", [gt()], [pred(),pred((20,0,30,10),"orphan")], (1,1,0,2/3,.5)),
            ("spanning", [gt((0,0,1,1),"a"),gt((1,0,2,1),"b")], [pred((0,0,2,1))], (1,1,0,1,1)),
            ("spanning_gap", [gt((0,0,1,1),"a"),gt((2,0,3,1),"b")], [pred((0,0,3,1))], (1,1,0,.8,2/3)),
            ("miss_tiny", [gt(id="large"),gt((20,0,21,1),"tiny")], [pred()], (.5,.5,0,200/201,100/101)),
            ("wrong_class", [gt()], [pred(cls=1)], (0,0,0,0,0)),
            ("no_overlap", [gt()], [pred((20,0,30,10))], (0,0,0,0,0)),
        ]
        for name, gs, ps, expected in cases:
            with self.subTest(name=name):
                summary = aggregate([score(gs, ps)])
                for key, value in zip(("mAC","BestIoGT","CollectiveGain","ClassUnionDice","ClassUnionIoU"), expected):
                    self.near(summary[key], value)

    def test_duplicate_and_repartition_invariance(self):
        base = score([gt()], [pred()])
        copies = score([gt()], [pred(id=str(i)) for i in range(100)])
        tiled = score([gt()], [pred((i,0,i+1,10),str(i)) for i in range(10)])
        for other in (copies, tiled):
            for key in ("mAC", "ClassUnionDice", "ClassUnionIoU"):
                self.near(aggregate([base])[key], aggregate([other])[key])
        self.near(aggregate([tiled])["CollectiveGain"], .9)
        self.assertEqual(copies.counts["n_pred"], 100)

    def test_overlapping_gt_and_orphans_count_area_once(self):
        d = score([gt(id="a"),gt((2,2,8,8),"b")],
                  [pred(),pred((20,0,30,10),"o1"),pred((20,0,30,10),"o2")])
        row = d.per_class[0]
        self.near(row["intersection_area"],100)
        self.near(row["gt_union_area"],100)
        self.near(row["orphan_union_area"],100)
        self.assertEqual(row["orphan_pred_count"],2)
        self.assertEqual(row["multigt_pred_count"],1)

    def test_order_and_repeated_calls(self):
        gs = [gt(id="a"), gt((2,2,8,8),"b")]
        ps = [pred(id="z"),pred(id="a")]
        a = score(gs, ps)
        self.assertEqual(a, score(gs[::-1], ps[::-1]))
        self.assertEqual(a, score(gs, ps))
        self.assertEqual(a.per_gt[0]["best_pred_id"], "a")

    def test_empty_cases(self):
        present = aggregate([score([gt()], [])])
        self.assertEqual(present["mAC"],0)
        self.assertEqual(present["ClassUnionDice"],0)
        absent = aggregate([score([], [pred()])])
        self.assertIsNone(absent["mAC"])
        self.assertEqual(absent["ClassUnionDice"],0)
        empty = aggregate([score([], [])])
        self.assertIsNone(empty["mAC"])
        self.assertIsNone(empty["ClassUnionDice"])
        self.assertEqual(empty["avg_preds"],0)
        self.assertIsNone(aggregate([])["avg_preds"])

    def test_threshold_equality_and_tau_separation(self):
        ps = [pred((0,0,3,10),"low",score=.249),
              pred((3,0,6,10),"equal",score=.25),pred((6,0,10,10),"high",score=.9)]
        a = aggregate([score([gt()],ps,config=Config(tau=.5))])
        b = aggregate([score([gt()],ps,config=Config(tau=.8))])
        self.near(a["mAC"],.7)
        self.assertEqual(a["mAC"], b["mAC"])
        self.assertEqual(a["RecallACtau"],1)
        self.assertEqual(b["RecallACtau"],0)
        self.assertEqual(a["filtered_low_score_pred"],1)

    def test_clipping_and_invalid_prediction_counts(self):
        ps = [pred((-5,-5,15,15)), pred((0,0,0,1),"zero"),
              pred((20,20,30,30),"outside"),pred((float("nan"),0,1,1),"nan")]
        d = score([gt()],ps,size=(10,10))
        self.near(aggregate([d])["mAC"],1)
        self.assertEqual(d.counts["n_pred"],1)
        self.assertEqual(d.counts["skipped_nonfinite_pred"],1)
        self.assertEqual(d.counts["skipped_degenerate_pred"],2)
        self.assertEqual(d.counts["clipped_pred"],2)
        clipped_gt = score([gt((-5,0,5,10))],[pred((0,0,5,10))],size=(10,10))
        self.assertEqual(clipped_gt.counts["clipped_gt"],1)
        self.assertEqual(clipped_gt.per_gt[0]["gt_area"],50)

    def test_fail_loudly_on_malformed_input(self):
        for gs, ps in [
            ([gt((0,0,0,10))],[]), ([gt((float("inf"),0,1,1))],[]),
            ([gt((20,20,30,30))],[]), ([gt((5,0,1,1))],[]),
            ([gt()],[pred((5,0,1,1))]), ([gt()],[pred(score=float("nan"))]),
            ([gt()],[pred(score=None)]), ([gt(),gt()],[]), ([gt()],[pred(),pred()]),
            ([gt(cls=-1)],[]), ([gt(cls=True)],[]), ([gt()], [pred(cls=2)]),
            ([gt()],[pred((0,0,"x",1))]),
        ]:
            with self.subTest(gs=gs,ps=ps), self.assertRaises(ValueError):
                score(gs,ps,size=(10,10),config=Config(class_ids=(0,1)))
        for kwargs in ({"theta":2},{"theta":float("nan")},{"tau":0},{"cg_epsilon":-1},{"class_ids":(0,0)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)

    def test_fractional_touching_and_numeric_scale(self):
        self.near(union_area([(0,.1,.25,.6),(.25,.1,.5,.6)]),.25)
        self.assertIsNone(intersection((0,0,1,1),(1,0,2,1)))
        huge = score([gt((0,0,1e154,1e154))],[pred((0,0,1e154,1e154))],size=(1e154,1e154))
        self.near(huge.per_class[0]["ClassUnionDice"],1)

    def test_random_independent_geometry_oracle(self):
        rng = random.Random(1701)
        for _ in range(80):
            def rect():
                x,y = rng.randrange(0,20)/4, rng.randrange(0,20)/4
                return (x,y,x+rng.randrange(1,12)/4,y+rng.randrange(1,12)/4)
            gs, ps = [rect() for _ in range(3)], [rect() for _ in range(6)]
            self.near(union_area(ps), oracle_area(ps))
            d = score([gt(r,str(i)) for i,r in enumerate(gs)],
                      [pred(r,str(i)) for i,r in enumerate(ps)])
            row = d.per_class[0]
            self.near(row["intersection_area"],oracle_area(gs,ps))
            self.near(row["ClassUnionDice"],2*row["ClassUnionIoU"]/(1+row["ClassUnionIoU"]))
            for item,g in zip(d.per_gt,gs):
                self.near(item["AC"],oracle_area([g],ps)/area(g))

    def test_addition_monotonicity_and_gain_can_decrease(self):
        parts = [pred((0,0,5,10),"a"),pred((5,0,10,10),"b")]
        a = aggregate([score([gt()],parts)])
        b = aggregate([score([gt()],parts+[pred()])])
        self.assertGreaterEqual(b["mAC"],a["mAC"])
        self.assertGreaterEqual(b["BestIoGT"],a["BestIoGT"])
        self.assertGreater(a["CollectiveGain"],b["CollectiveGain"])


class AggregationTests(unittest.TestCase):
    def test_gt_weighting_and_prediction_only_pairs(self):
        a = score([gt()],[pred()],image_id="a")
        b = score([gt(id=str(i)) for i in range(3)],[],image_id="b")
        c = score([],[pred()],image_id="c")
        summary = aggregate([a,b,c])
        self.assertAlmostEqual(summary["mAC"],.25)
        self.assertAlmostEqual(summary["ClassUnionDice"],1/3)
        self.assertEqual(summary["n_active_image_classes"],3)
        self.assertEqual(summary,aggregate([c]+[b,a]))

    def test_duplicate_images_and_mixed_configs_fail(self):
        a = score([gt()],[pred()],image_id="a")
        with self.assertRaises(ValueError):
            aggregate([a,a])
        with self.assertRaises(ValueError):
            aggregate([a,score([gt()],[pred()],image_id="b",config=Config(theta=.5))])

    def test_quantiles_and_mean_difference(self):
        d = score([gt()], [pred((0,0,5,10),"a"),pred((5,0,10,10),"b")])
        s = aggregate([d])
        self.assertEqual(s["cg_positive_fraction"],1)
        self.assertEqual(s["cg_p50"],.5)
        self.assertAlmostEqual(s["CollectiveGain"],s["mAC"]-s["BestIoGT"])


class LegacyParityTests(unittest.TestCase):
    def test_100_seeded_scenes_against_working_and_archived_mac(self):
        if (importlib.util.find_spec("mean_AC") is None
                or not (PROJECT_ROOT / "bin" / "old" / "mean_AC.py").is_file()):
            self.skipTest("Historical working and archived mean_AC sources are not bundled")
        import mean_AC as legacy
        spec = importlib.util.spec_from_file_location("archived_mean_AC", PROJECT_ROOT/"bin"/"old"/"mean_AC.py")
        archived = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = archived
        spec.loader.exec_module(archived)
        rng = random.Random(409)
        max_error = 0.0
        for case in range(100):
            def rect():
                x,y = rng.uniform(0,70),rng.uniform(0,70)
                return (x,y,x+rng.uniform(.1,25),y+rng.uniform(.1,25))
            gs = [gt(rect(),str(i),rng.randrange(2)) for i in range(1+case%5)]
            ps = [pred(rect(),str(i),rng.randrange(2),rng.random()) for i in range(case%13)]
            actual = score(gs,ps)
            for module in (legacy,archived):
                old_g = [module.Box(*b.xyxy,cls=b.class_id) for b in gs]
                old_p = [module.Box(*b.xyxy,cls=b.class_id,conf=b.score) for b in ps]
                m,covs,hits = module.image_ac(old_g,old_p,tau=.5,score_thresh=.25,class_aware=True)
                result = aggregate([actual])
                by_id = {row["gt_id"]: row for row in actual.per_gt}
                aligned = [by_id[box.id] for box in gs]
                errors = [abs(m-result["mAC"])] + [abs(a-b["AC"]) for a,b in zip(covs,aligned)]
                max_error = max(max_error,*errors)
                self.assertTrue(all(e<=TOL for e in errors))
                self.assertEqual(hits,[row["coverage_hit"] for row in aligned])
        print(f"Legacy parity: 100 scenes x 2 source versions, max abs error={max_error:.3g}")


class OfflineTests(unittest.TestCase):
    def document(self):
        return json.loads((PROJECT_ROOT/"examples"/"region_eval"/"synthetic_predictions.json").read_text(encoding="utf-8"))

    def test_input_contract_rejects_ambiguous_protocols(self):
        doc = self.document()
        changes = [
            ("coordinate_frame","normalized_xywh"),("class_ids",[99]),
            ("ignore_crowd_policy","unspecified"),
            ("postprocessing",dict(stage="post_nms",confidence_floor=.5,nms_iou=.7,max_det=500)),
            ("postprocessing",dict(stage="post_nms",confidence_floor=0,nms_iou=.7,max_det=1)),
        ]
        for key,value in changes:
            modified=copy.deepcopy(doc); modified["metadata"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                evaluate_document(modified)
        modified=copy.deepcopy(doc); modified["images"][0]["gts"][0]["ignore"]=True
        with self.assertRaises(ValueError):
            evaluate_document(modified)
        modified=copy.deepcopy(doc); modified["images"].append(modified["images"][0])
        with self.assertRaises(ValueError):
            evaluate_document(modified)

    def test_csv_metadata_roundtrip_and_no_overwrite(self):
        doc=self.document()
        config,images,summary=evaluate_document(doc)
        source=PROJECT_ROOT/"examples"/"region_eval"/"synthetic_predictions.json"
        with tempfile.TemporaryDirectory(prefix="region_eval_test_") as tmp:
            output=Path(tmp)/"new_run"
            kwargs=dict(source=source,source_bytes=source.read_bytes(),document=doc,
                        config=config,images=images,summary=summary)
            write_report(output,**kwargs)
            with (output/"region_gt.csv").open(encoding="utf-8-sig",newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),summary["n_gt"])
            self.assertTrue(any(row["best_pred_id"]=="NA" for row in rows))
            meta=json.loads((output/"metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"],"complete")
            self.assertEqual(len(meta["image_manifest"]),summary["n_images"])
            self.assertIn("region_eval/diagnostics.py",meta["code_sha256"])
            with self.assertRaises(ValueError):
                write_report(output,**kwargs)
            with self.assertRaises(ValueError):
                write_report(PROJECT_ROOT/"bin"/"old"/"forbidden_new_output",**kwargs)

    def test_cli_smoke_and_refuse_existing_output(self):
        source=PROJECT_ROOT/"examples"/"region_eval"/"synthetic_predictions.json"
        with tempfile.TemporaryDirectory(prefix="region_eval_cli_") as tmp:
            out=Path(tmp)/"cli"
            cmd=[sys.executable,"-B","-m","region_eval","--input",str(source),"--output",str(out)]
            result=subprocess.run(cmd,cwd=PROJECT_ROOT,capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)["n_images"],len(self.document()["images"]))
            duplicate=subprocess.run(cmd,cwd=PROJECT_ROOT,capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(duplicate.returncode,2)
            self.assertIn("already exists",duplicate.stderr)


if __name__=="__main__":
    unittest.main()

