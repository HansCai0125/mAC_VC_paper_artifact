import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pandas as pd
from diagnostics.common import Settings, Results
from diagnostics.geometry import matching, owners, local_dice, unit_features, enrichment, components
from diagnostics.statistics import ratio, grouped_ols, holm, residual_contrast
from diagnostics.masks import coarsening
from region_eval.mask_grounded_validity_exact import geometry_from_annotation

def gt(i,box,c=0):return dict(id=f'gt_{i}',xyxy=box,class_id=c)
def pred(i,box,score=.9,c=0):return dict(id=f'p{i}',xyxy=box,score=score,class_id=c)
def annotation(i,box):
    x,y,X,Y=box
    return dict(id=i,area=(X-x)*(Y-y),segmentation=[[x,y,X,y,X,Y,x,Y]])

class GeometryTests(unittest.TestCase):
    def test_perfect_duplicate_and_orphan(self):
        g=[gt(1,[0,0,10,10])];p=[pred(1,[0,0,10,10]),pred(2,[0,0,10,10]),pred(3,[15,15,20,20])]
        f=unit_features(g,p,30,30)
        self.assertEqual(f['mAC'],1);self.assertEqual(f['VC'],.5);self.assertEqual(f['TP'],1);self.assertEqual(f['FP'],2)
        self.assertAlmostEqual(f['LRP'],2/3)
    def test_empty(self):
        self.assertEqual(unit_features([],[],10,10)['VC'],0)
        f=unit_features([], [pred(1,[0,0,1,1])],10,10)
        self.assertEqual(f['VC'],0);self.assertEqual(f['n_orphan'],1)
    def test_ownership_exclusive_and_alternative(self):
        g=[gt(1,[0,0,10,10]),gt(2,[0,0,100,100])];p=[pred(1,[0,0,15,15])]
        own,_=owners(g,p);alt,_=owners(g,p,True)
        self.assertEqual(len(own['gt_1']),1);self.assertEqual(len(alt['gt_2']),1)
        self.assertEqual(len(matching(g,p)),0)
    def test_fragments_cover_without_match(self):
        g=gt(1,[0,0,10,10]);p=[pred(i,[i*2.5,0,(i+1)*2.5,10]) for i in range(4)]
        f=unit_features([g],p,10,10)
        self.assertEqual(f['mAC'],1);self.assertEqual(f['VC'],1);self.assertEqual(f['F1'],0)
        self.assertAlmostEqual(local_dice(g,p),1)
    def test_enrichment_analytic_and_repeatable(self):
        g=gt(1,[0,0,10,10]);p=[pred(1,[0,0,5,10]),pred(2,[5,0,10,10],.8)]
        geo=geometry_from_annotation(annotation(1,[0,0,10,5]));s=replace(Settings(),translations=20)
        r=enrichment(g,p,geo,s,'run','image');rr=enrichment(g,p,geo,s,'run','image')
        self.assertEqual(r,rr);self.assertAlmostEqual(r['added_area'],50);self.assertAlmostEqual(r['observed_fg'],25)
        self.assertAlmostEqual(r['area_expected_fg'],25);self.assertAlmostEqual(r['observed_density'],.5)
        self.assertAlmostEqual(r['shift_expected_fg'],r['shift_expected_area']/2)
    def test_exclusion_and_zero_reference(self):
        g=gt(1,[0,0,10,10]);geo=geometry_from_annotation(annotation(1,[0,0,5,10]));s=replace(Settings(),translations=2)
        p=pred(1,[0,0,5,10]);self.assertEqual(enrichment(g,[p],geo,s,'r','i')['status'],'K_less_than_2')
        self.assertEqual(enrichment(g,[p,p],geo,s,'r','i')['status'],'no_added_area')
        r=enrichment(g,[p,pred(2,[5,0,10,10])],geo,s,'r','i');self.assertTrue(r['area_reference_zero'])
    def test_graph_is_transitive_and_class_specific(self):
        g=[gt(1,[0,0,3,3]),gt(2,[2,0,5,3]),gt(3,[4,0,7,3]),gt(4,[0,0,7,3],1)]
        self.assertEqual(sorted(map(len,components(g,0,10,10))),[1,3])
    def test_coarsening_exposes_gap_background(self):
        g=[gt(1,[0,0,2,2]),gt(2,[3,0,5,2])]
        im=dict(width=10,height=10,gts=g,predictions=[pred(1,[0,0,5,2])])
        rows=coarsening(im,{1:annotation(1,[0,0,2,2]),2:annotation(2,[3,0,5,2])},replace(Settings(),coarsening_deltas=(.25,)))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['group_size'],2)
        self.assertAlmostEqual(rows[0]['foreground_occupancy'],.8)

class StatsTests(unittest.TestCase):
    def test_constant_cluster_ratio(self):
        f=pd.DataFrame(dict(image_id=['a','a','b'],n=[2,4,6],d=[1,2,3]))
        r=ratio(f,'n','d',['a','b'],replace(Settings(),bootstrap=50))
        self.assertEqual(r['estimate'],2);self.assertEqual(r['ci_low'],2);self.assertEqual(r['ci_high'],2)
    def test_holm_includes_missing_test(self):
        r=holm([dict(p_one_sided=.01),dict(p_one_sided=.04),dict(p_one_sided=None)])
        self.assertAlmostEqual(r[0]['p_holm'],.03);self.assertAlmostEqual(r[1]['p_holm'],.08);self.assertEqual(r[2]['p_holm'],1)
    def test_same_image_seeds_never_leak(self):
        rng=np.random.default_rng(11);rows=[]
        for i in range(60):
            x,z=rng.normal(size=2)
            for seed in range(5):rows.append(dict(image_id=str(i),x=x,z=z,y=.2*x+.8*z,seed=seed))
        f=pd.DataFrame(rows);s=replace(Settings(),cv_repeats=2)
        base,_,_=grouped_ols(f,['x'],s,list(map(str,range(60))),'y')
        add,_,audit=grouped_ols(f,['x','z'],s,list(map(str,range(60))),'y')
        self.assertTrue(all(a['shared_images']==0 for a in audit));self.assertLess(add.mean(),1e-20);self.assertGreater(base.mean(),.1)
    def test_strata_do_not_invent_contrast_with_ties(self):
        f=pd.DataFrame([dict(image_id=str(i),gt_id='g',is_fn=1,intersecting_pred_count=2,BestIoGT=.4,MaskCollectiveGain=.2,CollectiveGain=0,class_id=0,size_bin='small') for i in range(70)])
        r,a=residual_contrast(f,list(map(str,range(70))),replace(Settings(),bootstrap=50))
        self.assertIsNone(r['estimate']);self.assertEqual(a[0]['status'],'tied_terciles')
    def test_prespecified_contrast_estimates_known_difference(self):
        rows=[]
        for i in range(90):
            for seed in range(5):
                gain=(i%3)*.1
                rows.append(dict(image_id=str(i),gt_id='g',seed=seed,is_fn=1,intersecting_pred_count=2,BestIoGT=.4,MaskCollectiveGain=gain/2,CollectiveGain=gain,class_id=0,size_bin='small'))
        f=pd.DataFrame(rows);r,a=residual_contrast(f,list(map(str,range(90))),replace(Settings(),bootstrap=50))
        self.assertAlmostEqual(r['estimate'],.1);self.assertAlmostEqual(r['ci_low'],.1)
        self.assertAlmostEqual(a[0]['seed_4_contrast'],.1);self.assertEqual(a[0]['unique_targets'],90)
    def test_result_resume_and_no_overwrite(self):
        parent=Path(__file__).resolve().parents[1]/'validation'
        parent.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=parent) as d:
            assert Path(d).resolve().is_relative_to(parent.resolve())
            r=Results(Path(d));r.save('A','run',1,dict(x=2));r.close()
            r=Results(Path(d));self.assertEqual(r.done('A','run'),{1});self.assertEqual(list(r.payloads('A','run'))[0][1],dict(x=2))
            with self.assertRaises(Exception):r.save('A','run',1,dict(x=3))
            r.close()

if __name__=='__main__':unittest.main()
