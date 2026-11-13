"""Read saved predictions, polygon audit and frozen SAM outputs; never infer."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json
import importlib.metadata
import sys
import time
import numpy as np
import pandas as pd
from region_eval.cache_registry import paired_yolo_sources
from region_eval.mask_grounded_validity_exact import PolygonGeometryCache
from downstream.masks import CocoMasks, evaluate_class_unions
from .common import *
from .geometry import predicted, matching, enrichment, unit_features
from .masks import coarsening, sam_mechanism

def run_id(s):return f'{s.dataset}_{s.model}_seed{s.seed}'

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def read_csv(path):return pd.read_csv(path,dtype={'image_id':str,'gt_id':str})

def preflight(root,settings):
    sources=paired_yolo_sources();files=[ANNOTATION,SAM/'protocol.json',SAM/'split.json',SAM/'complete.json',SAM/'cache.sqlite']
    if read_json(SAM/'complete.json').get('status')!='complete':raise ValueError('SAM run is incomplete')
    sam_protocol=read_json(SAM/'protocol.json');sam_sources={s['run_id']:s for s in sam_protocol['sources']}
    if sha256_file(ANNOTATION)!=sam_protocol['annotation_sha256']:raise ValueError('SAM annotation checksum mismatch')
    for s in sources:
        files.extend([s.prediction_file,NATURAL/s.dataset/s.model/f'seed_{s.seed}'/'per_gt_collective_gain.csv'])
        if s.dataset=='mscoco':
            files.extend([EXACT/s.dataset/s.model/f'seed_{s.seed}'/'per_gt_mask_validity.csv',SAM/'per_image'/f'{s.model}_seed{s.seed}.npz'])
            saved=sam_sources[f'{s.model}_seed{s.seed}']
            if saved['sha256']!=s.cache_sha256:raise ValueError(f'SAM/prediction source mismatch: {run_id(s)}')
    identities=[file_identity(p) for p in files]
    ann=read_json(ANNOTATION);crowd=[a for a in ann['annotations'] if a.get('iscrowd',0)]
    with read_only_db(SAM/'cache.sqlite') as db:
        tables={name:db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0] for name in ('detections','inferred','evaluated')}
    report=dict(status='passed',sources=len(sources),coco_runs=15,visdrone_runs=15,settings=asdict(settings),
        source_files=len(identities),sam_cache_tables=tables,crowd_annotations=len(crowd),
        crowd_images=len({a['image_id'] for a in crowd}),
        D1_status='not_run: registered prediction caches already apply crowd filtering; D2 is the prespecified substitute',
        sam_protocol_fingerprint=read_json(SAM/'complete.json')['protocol_fingerprint'])
    atomic_json(root/'validation/preflight.json',report)
    log(f"PREFLIGHT passed: 30 prediction runs, 15 SAM result files; offline CPU analysis only")
    return sources,identities,ann

def ab_run(source,settings,results,annotations,smoke_ids):
    rid=run_id(source);done=results.done('AB_D2',rid)
    log(f'{rid}: checking saved prediction checksum')
    if sha256_file(source.prediction_file)!=source.cache_sha256:raise ValueError(f'Prediction checksum mismatch: {rid}')
    document=read_json(source.prediction_file)
    images=document['images']
    if len(images)!=source.n_images or len({str(im['image_id']) for im in images})!=len(images):raise ValueError('Invalid image count or duplicate image IDs')
    natural=read_csv(NATURAL/source.dataset/source.model/f'seed_{source.seed}'/'per_gt_collective_gain.csv')
    natural=natural.set_index(['image_id','gt_id'])
    if not natural.index.is_unique:raise ValueError('Duplicate natural-audit target keys')
    exact=None;geo=PolygonGeometryCache(annotations)
    if source.dataset=='mscoco':
        exact=read_csv(EXACT/source.dataset/source.model/f'seed_{source.seed}'/'per_gt_mask_validity.csv').set_index(['image_id','gt_id'])
        if not exact.index.is_unique:raise ValueError('Duplicate polygon-audit target keys')
    selected=[(j,im) for j,im in enumerate(images) if not settings.smoke or (image_number(im['image_id']) in smoke_ids if source.dataset=='mscoco' else j<settings.smoke_images)]
    previous=time.monotonic()
    for n,(j,im) in enumerate(selected,1):
        if j in done:continue
        image_id=str(im['image_id']);ps=predicted(im,settings.theta);matched=matching(im['gts'],ps);rows=[];brows=[]
        for g in im['gts']:
            key=(image_id,g['id'])
            if key not in natural.index:raise ValueError(f'Missing natural target {rid}/{key}')
            row=natural.loc[key].to_dict();row.update(image_id=image_id,gt_id=g['id'],is_fn=int(g['id'] not in matched))
            pp=[p for p in ps if p['class_id']==g['class_id']]
            scores=[p['score'] for p in pp];row.update(confidence_mean=float(np.mean(scores)) if scores else 0.,confidence_max=max(scores,default=0.))
            row['mask_eligible']=0
            if exact is not None and key in exact.index:
                erow=exact.loc[key].to_dict()
                for field in ('AC','BestIoGT','CollectiveGain','ABO'):
                    if not np.isclose(row[field],erow[field],atol=1e-9,rtol=0):raise ValueError(f'Historical geometry mismatch: {key}/{field}')
                row.update({k:erow[k] for k in ('MaskAC','MaskBestCoverage','MaskCollectiveGain','foreground_continuous_area')})
                row['mask_eligible']=1;row['mask_occupancy']=erow['foreground_continuous_area']/row['gt_area']
                b=enrichment(g,pp,geo.get(int(erow['annotation_id'])),settings,rid,image_id)
                brows.append(dict(image_id=image_id,gt_id=g['id'],size_bin=row['size_bin'],is_fn=row['is_fn'],**b))
            rows.append(clean(row))
        drows=coarsening(im,annotations,settings) if source.dataset=='mscoco' else []
        results.save('AB_D2',rid,j,dict(image_id=image_id,A=rows,B=clean(brows),D2=clean(drows)))
        if time.monotonic()-previous>15 or n==len(selected):
            log(f'{rid}: A/B/D2 {n}/{len(selected)} images saved');previous=time.monotonic()
    return [str(im['image_id']) for _,im in selected]

def c_run(source,settings,results,annotations,smoke_ids):
    rid=run_id(source);samid=f'{source.model}_seed{source.seed}';done=results.done('C',rid)
    coco=CocoMasks(ANNOTATION)
    with np.load(SAM/'per_image'/f'{samid}.npz',allow_pickle=False) as z:
        configs=json.loads(str(z['config_json'].item()));ids=z['image_ids'].copy();counts=z['counts'].copy();prompts=z['prompt_counts'].copy()
    found=[i for i,c in enumerate(configs) if c['policy']=='fixed_025' and c['budget']==settings.budget and abs(c['theta']-settings.theta)<1e-12]
    if len(found)!=1:raise ValueError('Expected one frozen fixed_025/top20 configuration')
    ci=found[0];previous=time.monotonic()
    with read_only_db(SAM/'cache.sqlite') as db:
        for j,image_id in enumerate(ids):
            image_id=int(image_id)
            if settings.smoke and image_id not in smoke_ids:continue
            if image_id in done:continue
            raw=db.execute('SELECT payload FROM detections WHERE run_id=? AND image_id=?',(samid,image_id)).fetchone()
            inferred=db.execute('SELECT masks FROM inferred WHERE image_id=?',(image_id,)).fetchone()
            if raw is None or inferred is None:raise ValueError(f'Missing SAM cache {samid}/{image_id}')
            im=unpack(raw[0]);mask_map=unpack(inferred[0]);ps=predicted(im,settings.theta,settings.budget)
            if len(ps)!=int(prompts[j,ci]):raise ValueError('Frozen SAM prompt count mismatch')
            rles=[mask_map[digest_json([float(x) for x in p['xyxy']])] for p in ps]
            # A full small-sample check, and one checkpoint per formal run.
            if settings.smoke or j==0:
                actual=evaluate_class_unions(ps,rles,coco.ground_truth(image_id),im['height'],im['width'],coco.n_classes)
                if not np.array_equal(actual,counts[j,ci]):raise ValueError('Frozen SAM pixel-count mismatch')
            rows=[];excluded=0
            for c,(tp,fp,fn) in enumerate(counts[j,ci]):
                if tp+fp+fn==0:excluded+=1;continue
                gg=[g for g in im['gts'] if g['class_id']==c];pp=[p for p in ps if p['class_id']==c]
                features=unit_features(gg,pp,im['width'],im['height'])
                rows.append(dict(image_id=str(image_id),class_id=c,sam_iou=float(tp/(tp+fp+fn)),sam_precision=float(tp/(tp+fp)) if tp+fp else 0.,**features))
            mechanism=sam_mechanism(im,ps,mask_map,annotations)
            results.save('C',rid,image_id,dict(rows=clean(rows),mechanism=clean(mechanism),both_empty_classes=excluded))
            if time.monotonic()-previous>15 or j==len(ids)-1:
                log(f'{rid}: C {j+1}/{len(ids)} held-out images checked');previous=time.monotonic()

def execute(root,mode,name):
    root=Path(root).resolve()
    if Path(name).name!=name or name in ('.','..'):raise ValueError('Use a simple run name, not a path')
    settings=Settings()
    if mode=='smoke':settings=replace(settings,smoke=True,bootstrap=50,translations=3,cv_folds=2,cv_repeats=2)
    sources,identities,annotation=preflight(root,settings)
    if mode=='preflight':return
    selected=[s for s in sources if not settings.smoke or s.seed==0]
    code_files=list(root.rglob('*.py'))+list((PROJECT/'region_eval').glob('*.py'))+list((PROJECT/'experiments/segmentation/downstream').glob('*.py'))
    hashes={str(p):sha256_file(p) for p in sorted(code_files) if 'results' not in p.parts}
    protocol=dict(version=VERSION,settings=asdict(settings),sources=[s.record() for s in selected],input_files=identities,code_sha256=hashes,annotation_sha256=sha256_file(ANNOTATION),
                  analysis_plan_sha256=sha256_file(root/'PLAN.md'),runtime=dict(python=sys.version,packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','pycocotools')}))
    protocol['fingerprint']=digest_json(protocol)
    output=root/'results'/name
    with run_lock(output):
        old=output/'protocol.json'
        if old.exists() and read_json(old)['fingerprint']!=protocol['fingerprint']:
            raise ValueError('Code, input or settings changed. Keep old results and choose a new RUN_NAME.')
        if (output/'complete.json').exists():log(f'Already complete: {output}');return
        if not old.exists():atomic_json(old,protocol)
        # Select smoke images before observing any diagnostic outcomes; include only held-out images.
        evaluation=sorted(read_json(SAM/'split.json')['evaluation_ids'])
        smoke_ids=set(evaluation[:settings.smoke_images])
        annotations={int(a['id']):a for a in annotation['annotations']}
        results=Results(output);populations={}
        try:
            for s in selected:
                population=ab_run(s,settings,results,annotations,smoke_ids)
                if s.dataset in populations and populations[s.dataset]!=population:raise ValueError('Run image universes differ')
                populations[s.dataset]=population
                if s.dataset=='mscoco':c_run(s,settings,results,annotations,smoke_ids)
            from .report import make_report
            make_report(output,results,selected,populations,settings)
            for before in identities:
                if file_identity(before['path'])!=before:raise ValueError('Source file changed during analysis: '+before['path'])
            atomic_json(output/'complete.json',dict(status='complete',smoke=settings.smoke,protocol_fingerprint=protocol['fingerprint'],runs=len(selected)))
        except BaseException as exc:
            atomic_json(output/'last_failure.json',dict(error=repr(exc),time=time.strftime('%Y-%m-%d %H:%M:%S')))
            raise
        finally:results.close()
    log(f'COMPLETE: {output}. '+('SMOKE TEST ONLY; not paper results.' if settings.smoke else 'Ready for result review.'))
