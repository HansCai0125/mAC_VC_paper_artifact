"""RLE union audit for controlled coarsening and SAM mechanism diagnostics."""
from collections import defaultdict
import numpy as np
from pycocotools import mask as api
from downstream.masks import _native_rle, _merge_rles, _segmentation_rle
from .geometry import components, predicted, owners, local_dice, _gt_metrics, area
from .common import digest_json

def rarea(r):return float(api.area(_native_rle(r,*r['size'])))

def overlap(a,b):
    return float(api.area(api.merge([_native_rle(a,*a['size']),_native_rle(b,*b['size'])],intersect=True)))

def rectangle(r,h,w):
    x,y,X,Y=r
    return _segmentation_rle([[x,y,X,y,X,Y,x,Y]],h,w)

def coarsening(image,annotations,settings):
    h,w=image['height'],image['width'];ps=predicted(image,settings.theta)
    raster={g['id']:_segmentation_rle(annotations[int(g['id'].split('_')[-1])]['segmentation'],h,w) for g in image['gts']}
    byclass=defaultdict(list)
    for p in ps:byclass[p['class_id']].append(p)
    predunion={c:_merge_rles([rectangle(p['xyxy'],h,w) for p in pp],h,w) for c,pp in byclass.items()}
    rows=[]
    for delta in settings.coarsening_deltas:
        groups=components(image['gts'],delta,w,h);targets=[]
        for j,members in enumerate(groups):
            a=np.array([g['xyxy'] for g in members]);box=[float(a[:,0].min()),float(a[:,1].min()),float(a[:,2].max()),float(a[:,3].max())]
            targets.append(dict(id=f'group_{j:06d}',class_id=members[0]['class_id'],xyxy=box,members=members))
        own,orp=owners(targets,ps)
        for g in targets:
            members=g['members'];c=g['class_id'];pp=byclass[c]
            mask=_merge_rles([raster[m['id']] for m in members],h,w);boxmask=rectangle(g['xyxy'],h,w)
            union=predunion.get(c);ma,ba=rarea(mask),rarea(boxmask)
            maskac=overlap(mask,union)/ma if ma>0 and union is not None else (0. if ma>0 else None)
            boxac=overlap(boxmask,union)/ba if ba>0 and union is not None else (0. if ba>0 else None)
            metrics=_gt_metrics(g,pp)
            # Foreground outside the enclosing annotation bbox is reported separately.
            inside=overlap(mask,boxmask)
            edges=0
            if len(members)>1:
                for i,m in enumerate(members):
                    for n in members[:i]:
                        if len(components([m,n],delta,w,h))==1:edges+=1
            rows.append(dict(delta=delta,group_id=g['id'],class_id=c,member_ids=';'.join(m['id'] for m in members),group_size=len(members),
                is_transitive_chain=int(len(members)>2 and edges<len(members)*(len(members)-1)/2),
                bbox_area=area(g['xyxy']),mask_pixels=ma,bbox_pixels=ba,
                mask_outside_bbox_pixels=max(0.,ma-inside),foreground_occupancy=inside/ba if ba else None,
                box_AC_continuous=metrics['AC'],box_AC_raster=boxac,mask_AC_raster=maskac,
                box_minus_mask=boxac-maskac if boxac is not None and maskac is not None else None,
                collective_gain=metrics['CollectiveGain'],local_VC=local_dice(g,own[g['id']]),
                n_groups=len(targets),n_orphans=len(orp)))
    return rows

def sam_mechanism(image,ps,mask_map,annotations):
    h,w=image['height'],image['width'];byclass=defaultdict(list)
    for p in ps:byclass[p['class_id']].append(p)
    own,_=owners(image['gts'],ps);alt,_=owners(image['gts'],ps,True)
    def union(pp):return _merge_rles([mask_map[digest_json([float(x) for x in p['xyxy']])] for p in pp],h,w)
    classmask={c:union(pp) for c,pp in byclass.items()};rows=[]
    for g in image['gts']:
        target=_segmentation_rle(annotations[int(g['id'].split('_')[-1])]['segmentation'],h,w);a=rarea(target)
        if not a:continue
        pp=byclass[g['class_id']];m=_gt_metrics(g,pp);cm=classmask.get(g['class_id'])
        row=dict(gt_id=g['id'],class_id=g['class_id'],AC=m['AC'],ABO=m['ABO'],BestIoGT=m['BestIoGT'],
                 foreground_recall=overlap(target,cm)/a if cm is not None else 0.,local_VC=local_dice(g,own[g['id']]),local_VC_intersection=local_dice(g,alt[g['id']]))
        for name,allocation in [('iou',own),('intersection',alt)]:
            r=union(allocation[g['id']]);inter=overlap(target,r);den=a+rarea(r)-inter
            row['owner_mask_iou_'+name]=inter/den if den else 0.
        rows.append(row)
    return rows
