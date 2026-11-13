"""Non-destructive densification of manuscript Figures 5 and 6.

Run with the yolo environment Python. All writes stay in a new output folder.
No model, GPU, training, or inference is used. Original files are read-only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from region_eval import collective_fragmentation as cf
from region_eval import vc_incremental_value as iv
from region_eval.gaming import load_shard_sample, score_document, _normalizations
from region_eval.gaming_formal import _source_records
from region_eval.geometry import area
from dense_eval import score as fast_score

OLD5 = ROOT / 'results/fragmentation'
OLD6 = ROOT / 'results/vc_components'
OUT = ROOT / 'results/figure_data'
RHO = tuple(round(i/100, 2) for i in range(20,101,5))
FIELDS = ('mAC','VC','F1','LRP','AP50','AP75','AP50_95','AR50_95',
          'n_gt','n_pred','orphan_pred_count','assigned_pred_count')


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_json(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def parity(actual, expected, fields=FIELDS, tolerance=1e-10):
    errors={k:abs(float(actual[k])-float(expected[k])) for k in fields}
    bad={k:v for k,v in errors.items() if v>tolerance}
    if bad: raise AssertionError(f'metric parity failed: {bad}')
    return errors


def transformed(document, targets, rho):
    images=[]; intended={}; total=0.0; expected=0.0; max_error=0.0
    for im in document['images']:
        ps=[]
        for p in im['predictions']:
            t=targets.get((im['image_id'],p['id']))
            if t is None: ps.append(p); continue
            fs=cf._fragments(t,4,rho)
            got=math.fsum(area(tuple(f['xyxy'])) for f in fs)
            want=rho*t['support_area']
            max_error=max(max_error,abs(got-want)/max(1.0,want))
            total+=got; expected+=t['support_area']
            for f in fs: intended[(im['image_id'],f['id'])]=t['owner_gt_id']
            ps.extend(fs)
        images.append(dict(im,predictions=ps))
    if len(intended)!=16*len(targets): raise AssertionError('target count changed')
    if max_error>1e-12: raise AssertionError('support geometry error')
    doc=dict(document,images=images)
    audit=dict(target_pair_count=len(targets),q=4,fragment_count=16,retention=rho,
               replaced_prediction_count=len(targets),generated_fragment_count=len(intended),
               added_prediction_count=len(intended)-len(targets),
               retained_area_ratio_actual=total/expected,
               retained_area_ratio_abs_error=abs(total/expected-rho),
               max_target_retention_abs_error=max_error)
    return doc,intended,audit


def old_run(record):
    path=OLD5/'runs'/record['dataset']/record['model']/f"seed_{int(record['seed'])}"
    return path,json.loads((path/'fragmentation_complete.json').read_text(encoding='utf-8'))


def validate():
    OUT.mkdir(parents=True,exist_ok=True)
    records=_source_records(cf.MATRIX_ROOT)
    results=[]
    for dataset in ('mscoco','visdrone'):
        record=next(r for r in records if r['dataset']==dataset)
        doc,_=load_shard_sample(Path(record['output']),24,minimum_score=0.25)
        targets=cf._target_pairs(doc)
        for rho in (.2,.6,1.0,.35):
            start=time.perf_counter()
            orig,audit=cf.transform_document(doc,targets,q=4,rho=rho)
            reference=score_document(orig,include_diagnostic_ap=True)
            new,intended,new_audit=transformed(doc,targets,rho)
            if new['images']!=orig['images']: raise AssertionError('transform changed')
            actual,owner=fast_score(new,intended)
            errors=parity(actual,reference)
            if owner['ownership_change_count']!=audit['ownership_change_count']:
                raise AssertionError('owner audit mismatch')
            results.append(dict(dataset=dataset,rho=rho,errors=errors))
            print('VALIDATED',dataset,rho,'seconds',round(time.perf_counter()-start,2),flush=True)
    write_json(OUT/'evaluator_parity_samples.json',results)
    return results


def worker(record):
    run_id=f"{record['dataset']}_{record['model']}_seed{int(record['seed'])}"
    folder=OUT/'runs'/run_id
    folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    oldpath,complete=old_run(record)
    if complete['source']['run_fingerprint']!=record['source_run_fingerprint']:
        raise AssertionError('source fingerprint mismatch')
    old=read_csv(oldpath/'fragmentation_metrics.csv')
    baseline=complete['audit']['baseline_metrics']
    doc,source=load_shard_sample(Path(record['output']),int(record['n_images']),minimum_score=.25)
    if source['selected_shards']!=complete['source']['selected_shards']:
        raise AssertionError('source images changed')
    targets=cf._target_pairs(doc)
    if len(targets)!=complete['audit']['baseline_target_pair_count']:
        raise AssertionError('frozen targets changed')
    existing={float(r['level']):r for r in old if r['experiment']=='retention'}
    anchor_errors=[]
    # Recompute three old conditions on every complete source run, including
    # both retention endpoints. They must agree before accepting new samples.
    levels=[.2,.6,1.0]+[r for r in RHO if r not in existing]
    for rho in levels:
        p=folder/f'rho_{rho:.2f}.json'
        if p.exists():
            row=json.loads(p.read_text(encoding='utf-8'))
        else:
            condition_start=time.perf_counter()
            new,intended,audit=transformed(doc,targets,rho)
            metrics,owners=fast_score(new,intended)
            row=dict(experiment='retention',variant='fixed_4x4_grid',level=rho,
                     level_label=f'retained support {rho:.2f}',primary_theta=.25,
                     matching_iou=.5,**audit,**owners,**metrics)
            _normalizations(row,baseline)
            write_json(p,row)
            print('POINT',run_id,rho,'seconds',round(time.perf_counter()-condition_start,1),flush=True)
        if rho in existing:
            anchor_errors.append(dict(rho=rho,errors=parity(row,existing[rho])))
    write_json(folder/'anchor_validation.json',anchor_errors)
    additions=[]
    for rho in RHO:
        if rho in existing: continue
        row=json.loads((folder/f'rho_{rho:.2f}.json').read_text(encoding='utf-8'))
        row.update(dataset=record['dataset'],model=record['model'],seed=int(record['seed']),
                   model_id=record['model_id'],source_run_fingerprint=record['source_run_fingerprint'])
        additions.append(row)
    write_json(folder/'complete.json',dict(status='complete',source=run_id,
               new_conditions=len(additions),images=len(doc['images']),
               frozen_targets=len(targets),elapsed_seconds=time.perf_counter()-start))
    print('RUN COMPLETE',run_id,'seconds',round(time.perf_counter()-start,1),flush=True)
    return additions


def canonical():
    import numpy as np
    # Original anchors + a dense logarithmic grid + close neighbours of the
    # analytic IoU=.5 boundaries. No interpolation or smoothing is performed.
    scales=set(float(x) for x in np.geomspace(.5,3.0,101)) | set(iv.SCALE_GRID)
    for c in (1/math.sqrt(2),math.sqrt(2)):
        scales.update((c-1e-6,c,c+1e-6))
    iv.SCALE_GRID=tuple(sorted(scales))
    iv.ORPHAN_COUNTS=tuple(range(17))
    iv.GAP_RATIOS=tuple(sorted(set(round(i*.05,2) for i in range(81)) | {1e-6}))
    rows=iv.canonical_property_rows()  # Original production evaluator.
    original=read_csv(OLD6/'canonical_response.csv')
    errors=[]
    for old in original:
        new=next(r for r in rows if r['experiment']==old['experiment'] and
                 r['variant']==old['variant'] and float(r['level'])==float(old['level']))
        errors.append(parity(new,old,fields=('mAC','VC','F1','LRP','n_gt','n_pred')))
    write_csv(OUT/'fig6_canonical_dense.csv',rows)
    write_json(OUT/'fig6_anchor_validation.json',errors)
    return rows


def protected():
    paths=[ROOT/'region_eval'/n for n in ('collective_fragmentation.py','vc_incremental_value.py',
           'gaming.py','vc_v1.py','matching_analysis.py','responsibility_analysis.py','geometry.py')]
    for folder in (OLD5,OLD6,ROOT/'results/fragmentation_figure'):
        paths.extend(p for p in folder.rglob('*') if p.is_file())
    paper=Path(r'D:\Desktop\Paper\Draft\Beyond mAP\mac_latex')
    paths.extend([paper/'mac_final.tex',paper/'mac_final.pdf',
        paper/'Figure/collective_fragmentation_response.pdf',
        paper/'Figure/vc_incremental_value_response.pdf'])
    return {str(p):digest(p) for p in paths}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['validate','run','training_logs'],default='run')
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='validate': validate(); return
    if args.stage=='run':
        hashes_path=OUT/'protected_hashes.json'
        if not hashes_path.exists(): write_json(hashes_path,protected())
        if not (OUT/'evaluator_parity_samples.json').exists(): validate()
        canonical()
        records=_source_records(cf.MATRIX_ROOT)
        additions=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(worker,r) for r in records]
            for f in as_completed(futures): additions.extend(f.result())
        old=read_csv(OLD5/'fragmentation_runs.csv')
        merged=old+additions
        if len(merged)!=18*(4+17): raise AssertionError('unexpected row count')
        write_csv(OUT/'fig5_fragmentation_dense.csv',merged)
    from plot_figures import plot_all
    plot_all(OUT)
    original=json.loads((OUT/'protected_hashes.json').read_text(encoding='utf-8'))
    differences=[path for path,sha in original.items() if digest(path)!=sha]
    if differences: raise AssertionError(f'protected source changed: {differences}')
    write_json(OUT/'complete.json',dict(status='complete',training=False,inference=False,
        original_files_unchanged=True,protected_file_count=len(original),
        fig5_rows=len(read_csv(OUT/'fig5_fragmentation_dense.csv')),
        fig6_rows=len(read_csv(OUT/'fig6_canonical_dense.csv'))))
    print('COMPLETE',str(OUT),flush=True)


if __name__=='__main__': main()
