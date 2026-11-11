"""Existing continuous evaluator for A/B; COCO RLE geometry for grouped masks."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from region_eval.geometry import area, intersection, union_area
from region_eval.natural_collective_gain import _gt_metrics
from region_eval.mask_grounded_validity_exact import polygon_rectangle_union_area, polygon_rectangle_area
from downstream.selection import _prefix_statistics
from .common import seed_for

def predicted(image, theta=.25, budget=None):
    p=sorted((p for p in image['predictions'] if p['score']>=theta),key=lambda p:(-p['score'],p['id']))
    return p if budget is None else p[:budget]

def matching(gts, predictions):
    unmatched={g['id']:g for g in gts};result={}
    for p in sorted(predictions,key=lambda p:(-p['score'],p['id'])):
        choices=[]
        for gid,g in unmatched.items():
            if p['class_id']!=g['class_id']:continue
            rect=intersection(tuple(p['xyxy']),tuple(g['xyxy']))
            a=area(rect) if rect else 0.
            iou=a/(area(p['xyxy'])+area(g['xyxy'])-a) if a else 0.
            if iou>=.5:choices.append((-iou,-a,gid))
        if choices:
            ni,_,gid=min(choices);result[gid]={'pred_id':p['id'],'iou':-ni};unmatched.pop(gid)
    return result

def owners(gts,predictions,intersection_first=False):
    owned=defaultdict(list);orphan=[]
    for p in predictions:
        choices=[]
        for g in gts:
            if g['class_id']!=p['class_id']:continue
            rect=intersection(tuple(g['xyxy']),tuple(p['xyxy']))
            a=area(rect) if rect else 0.
            if not a:continue
            iou=a/(area(g['xyxy'])+area(p['xyxy'])-a)
            choices.append((-a,-iou,g['id']) if intersection_first else (-iou,-a,g['id']))
        if choices:owned[min(choices)[2]].append(p)
        else:orphan.append(p)
    return owned,orphan

def local_dice(gt,ps):
    ints=[r for p in ps if (r:=intersection(tuple(gt['xyxy']),tuple(p['xyxy']))) is not None]
    a=union_area(ints);den=area(gt['xyxy'])+union_area([tuple(p['xyxy']) for p in ps])
    return 2*a/den if den else 0.

def unit_features(gts,ps,width,height):
    im={'gts':gts,'predictions':ps,'width':width,'height':height}
    st=_prefix_statistics(im, sorted(ps,key=lambda p:(-p['score'],p['id'])), {len(ps)})[len(ps)]
    ga=union_area([tuple(g['xyxy']) for g in gts]);pa=union_area([tuple(p['xyxy']) for p in ps])
    clips=[r for g in gts for p in ps if (r:=intersection(tuple(g['xyxy']),tuple(p['xyxy']))) is not None]
    ua=union_area(clips)
    tp,fp,fn=st['TP'],st['FP'],st['FN'];z=tp+fp+fn
    score=np.array([p['score'] for p in ps])
    return {'union_dice':2*ua/(ga+pa) if ga+pa else 0.,'n_prompts':len(ps),
            'mAC':st['ac_sum']/len(gts) if gts else 0.,
            'BestIoGT':st['best_iogt_sum']/len(gts) if gts else 0.,
            'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'LRP':(st['localization_error_sum']+fp+fn)/z if z else 0.,
            'VC':st['local_vc_sum']/(len(gts)+st['orphan_pred_count']) if len(gts)+st['orphan_pred_count'] else 0.,
            'score_mean':float(score.mean()) if len(score) else 0.,
            'score_max':float(score.max()) if len(score) else 0.,
            'score_std':float(score.std()) if len(score) else 0.,
            'n_gt':len(gts),'n_orphan':st['orphan_pred_count'],'TP':tp,'FP':fp,'FN':fn}

def enrichment(gt,ps,geometry,settings,run_id,image_id):
    """Randomized clipped rectangles retain size; E-prime area may change.

    R_shift compares total observed foreground area to expected translated
    foreground area. Additional area and foreground-density diagnostics expose
    the change in E-prime area. This is not an equal-new-area null.
    """
    g=tuple(gt['xyxy']);ga=area(g);eps=max(1.,ga)*settings.area_relative_epsilon
    candidates=[]
    for p in ps:
        r=intersection(g,tuple(p['xyxy']))
        if r is not None:candidates.append((p,r))
    candidates.sort(key=lambda t:(-area(t[1]),-t[0]['score'],t[0]['id']))
    if len(candidates)<2:return {'status':'K_less_than_2'}
    best=candidates[0][1];rects=[r for _,r in candidates]
    rest=ga-area(best);ea=max(0.,union_area(rects)-area(best))
    if rest<=eps:return {'status':'no_remaining_area','added_area':ea}
    if ea<=eps:return {'status':'no_added_area','added_area':ea}
    bfg=polygon_rectangle_area(geometry,best)
    # Match the historical continuous polygon convention, including GT-box clipping.
    in_g=polygon_rectangle_area(geometry,g)
    fg=max(0.,polygon_rectangle_union_area(geometry,rects)-bfg)
    if in_g>ga+eps or bfg>area(best)+eps or fg>ea+eps:
        return {'status':'polygon_area_bound_violation','added_area':ea,'observed_fg':fg}
    ref=max(0.,in_g-bfg)/rest
    rng=np.random.default_rng(seed_for(settings.random_seed,run_id,image_id,gt['id'],'translation'))
    fg_random=[];area_random=[];density_random=[]
    for _ in range(settings.translations):
        shifted=[best]
        for _,r in candidates[1:]:
            w,h=r[2]-r[0],r[3]-r[1]
            x=g[0]+rng.random()*max(0.,g[2]-g[0]-w)
            y=g[1]+rng.random()*max(0.,g[3]-g[1]-h)
            shifted.append((x,y,x+w,y+h))
        ar=max(0.,union_area(shifted)-area(best))
        fr=max(0.,polygon_rectangle_union_area(geometry,shifted)-bfg)
        if fr>ar+eps:return {'status':'translated_polygon_area_bound_violation','added_area':ea,'observed_fg':fg}
        area_random.append(ar);fg_random.append(fr)
        if ar>eps:density_random.append(fr/ar)
    expected=float(np.mean(fg_random));random_area=float(np.mean(area_random))
    return {'status':'eligible','added_area':ea,'observed_fg':fg,'area_expected_fg':ea*ref,
            'shift_expected_fg':expected,'shift_fg_mc_se':float(np.std(fg_random,ddof=1)/np.sqrt(len(fg_random))) if len(fg_random)>1 else 0.,
            'shift_expected_area':random_area,'observed_density':fg/ea,'area_reference_density':ref,
            'shift_reference_density':float(np.mean(density_random)) if density_random else None,
            'shift_zero_area_draws':len(area_random)-len(density_random),
            'area_reference_zero':ea*ref<=eps,'shift_reference_zero':expected<=eps}

def components(gts,delta,width,height):
    """Same-class connected components; boundary contact counts as intersection."""
    parent=list(range(len(gts)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    boxes=[]
    for g in gts:
        r=g['xyxy'];pad=delta*np.sqrt(area(r))
        boxes.append((max(0,r[0]-pad),max(0,r[1]-pad),min(width,r[2]+pad),min(height,r[3]+pad)))
    for i,a in enumerate(boxes):
        for j in range(i):
            b=boxes[j]
            if gts[i]['class_id']==gts[j]['class_id'] and max(a[0],b[0])<=min(a[2],b[2]) and max(a[1],b[1])<=min(a[3],b[3]):
                parent[find(i)]=find(j)
    groups=defaultdict(list)
    for i,g in enumerate(gts):groups[find(i)].append(g)
    return list(groups.values())
