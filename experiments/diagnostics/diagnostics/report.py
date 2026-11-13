"""Unselected results, exclusions, seed SDs, primary tests and readable output."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import write_csv, atomic_json, log, MODELS
from .statistics import ratio, residual_contrast, cv_increment, holm, grouped_ols

def export(path,rows):
    from .pipeline import clean
    write_csv(path,clean(rows.to_dict('records') if isinstance(rows,pd.DataFrame) else rows))

def fraction(n,d):return float(n/d) if d else None

def a_summary(f):
    fn=f[f.is_fn==1];multi=fn[fn.intersecting_pred_count>=2];eligible=multi[multi.mask_eligible==1]
    out=dict(gt_count=len(f),fn_count=len(fn),fn_multi_count=len(multi),A1_fraction=fraction(len(multi),len(fn)),mask_eligible_fn_multi=len(eligible))
    if len(eligible):
        out.update(A2_gain_fraction=float((eligible.MaskCollectiveGain>.05).mean()),
                   A2_strict_fraction=float(((eligible.MaskAC>=.5)&(eligible.MaskBestCoverage<.5)).mean()),
                   A2_coverage_fraction=float((eligible.MaskAC>=.5).mean()))
    return out

def make_report(output,results,sources,populations,settings):
    from .pipeline import run_id, clean
    report=output/'reports';report.mkdir(exist_ok=True)
    all_a=[];all_b=[];all_c=[];all_m=[];seedrows=[];dsummary=[];exclusions=[]
    for s in sources:
        rid=run_id(s);meta=dict(dataset=s.dataset,model=s.model,seed=s.seed,run_id=rid)
        aa=[];bb=[];dd=[]
        for _,p in results.payloads('AB_D2',rid):
            aa.extend(dict(**r,**meta) for r in p['A']);bb.extend(dict(**r,**meta) for r in p['B'])
            dd.extend(dict(image_id=p['image_id'],**r,**meta) for r in p['D2'])
        a=pd.DataFrame(aa);b=pd.DataFrame(bb);d=pd.DataFrame(dd)
        export(output/'per_run'/rid/'A_targets.csv',a);export(output/'per_run'/rid/'B_enrichment.csv',b);export(output/'per_run'/rid/'D2_groups.csv',d)
        all_a.append(a);seedrows.append(dict(**meta,**a_summary(a)))
        if len(b):
            all_b.append(b)
            for status,n in b.status.value_counts().items():exclusions.append(dict(**meta,phase='B',reason=status,count=int(n)))
            eligible=b[b.status=='eligible']
            for expected,flag in [('area_expected_fg','area_reference_zero'),('shift_expected_fg','shift_reference_zero')]:
                if expected not in eligible:continue
                zero=eligible[flag]==True
                exclusions.append(dict(**meta,phase='B',reason=expected+'_zero',count=int(zero.sum())))
        if len(d):
            for (delta,kind),g in pd.concat([d.assign(group_subset='all'),d[d.group_size>1].assign(group_subset='merged')]).groupby(['delta','group_subset']):
                dsummary.append(dict(**meta,delta=delta,group_subset=kind,groups=len(g),images=g.image_id.nunique(),mean_group_size=float(g.group_size.mean()),
                    chain_fraction=float(g.is_transitive_chain.mean()),foreground_occupancy=float(g.foreground_occupancy.mean()),
                    box_AC=float(g.box_AC_raster.mean()),mask_AC=float(g.mask_AC_raster.mean()),box_minus_mask=float(g.box_minus_mask.mean()),
                    mean_abs_box_mask_gap=float(g.box_minus_mask.abs().mean()),
                    continuous_raster_box_MAD=float((g.box_AC_continuous-g.box_AC_raster).abs().mean())))
            # Correct VC aggregation uses one orphan count per image, not per group.
            for delta,g in d.groupby('delta'):
                orphans=g.groupby('image_id').n_orphans.first().sum()
                dsummary.append(dict(**meta,delta=delta,group_subset='global_VC',groups=len(g),VC=float(g.local_VC.sum()/(len(g)+orphans)),orphans=int(orphans)))
        cc=[];mm=[];both_empty=0
        for image,p in results.payloads('C',rid):
            cc.extend(dict(**r,**meta) for r in p['rows'])
            mm.extend(dict(image_id=str(image),**r,**meta) for r in p['mechanism']);both_empty+=p['both_empty_classes']
        if cc:
            c=pd.DataFrame(cc);m=pd.DataFrame(mm);all_c.append(c);all_m.append(m)
            export(output/'per_run'/rid/'C_image_class.csv',c);export(output/'per_run'/rid/'C_target_mechanism.csv',m)
            exclusions.append(dict(**meta,phase='C',reason='both_empty_class',count=both_empty))
    a=pd.concat(all_a,ignore_index=True);b=pd.concat(all_b,ignore_index=True);c=pd.concat(all_c,ignore_index=True);m=pd.concat(all_m,ignore_index=True)
    export(report/'A_per_seed.csv',seedrows);export(report/'D2_per_seed.csv',dsummary);export(report/'exclusions.csv',exclusions)
    main=[];primary=[];perseed=[];mechanism=[]
    for model in MODELS:
        log(f'Summarizing {model}: image-cluster bootstrap and grouped CV')
        f=a[(a.dataset=='mscoco')&(a.model==model)].copy();population=populations['mscoco']
        # All COCO images, including zero eligible targets, enter the bootstrap universe.
        f['num']=(f.is_fn.eq(1)&f.intersecting_pred_count.ge(2)).astype(int);f['den']=f.is_fn.astype(int)
        main.append(dict(model=model,endpoint='A1_FN_multi_fraction',**ratio(f,'num','den',population,settings)))
        eligible=f[(f.is_fn==1)&(f.intersecting_pred_count>=2)&(f.mask_eligible==1)].copy();eligible['den']=1
        for key,condition in [('A2_mask_gain',eligible.MaskCollectiveGain>.05),('A2_strict_crossing',(eligible.MaskAC>=.5)&(eligible.MaskBestCoverage<.5)),('A2_mask_coverage',eligible.MaskAC>=.5)]:
            eligible['num']=condition.astype(int);main.append(dict(model=model,endpoint=key,**ratio(eligible,'num','den',population,settings)))
        res,audit=residual_contrast(f,population,settings);row=dict(model=model,endpoint='A3_residual_mask_gain',**res);main.append(row);primary.append(row.copy())
        export(report/f'A3_strata_{model}.csv',audit)
        valid=[r for r in audit if r['status']=='eligible']
        for seed in sorted(f.seed.unique()):
            key=f'seed_{int(seed)}_contrast'
            value=sum(r['unique_targets']*r[key] for r in valid)/sum(r['unique_targets'] for r in valid) if valid and all(r.get(key) is not None for r in valid) else None
            perseed.append(dict(model=model,seed=int(seed),endpoint='A3_residual_mask_gain',value=value))
        bf=b[(b.model==model)&(b.status=='eligible')].copy()
        for endpoint,expected,flag in [('B1_area_enrichment','area_expected_fg','area_reference_zero'),('B2_shift_enrichment','shift_expected_fg','shift_reference_zero')]:
            usable=bf[bf[flag]==False].copy() if flag in bf else bf.iloc[:0].copy()
            if expected not in usable:
                usable=pd.DataFrame(columns=['image_id','observed_fg',expected])
            res=ratio(usable,'observed_fg',expected,population,settings,null=1.)
            row=dict(model=model,endpoint=endpoint,eligible_targets=len(usable),**res);main.append(row);primary.append(row.copy())
            for seed,g in usable.groupby('seed'):
                perseed.append(dict(model=model,seed=int(seed),endpoint=endpoint,value=fraction(g.observed_fg.sum(),g[expected].sum())))
            # All-target ratio is a sensitivity check; zero-reference exclusions are not hidden.
            if expected in bf:main.append(dict(model=model,endpoint=endpoint+'_all_targets_sensitivity',**ratio(bf,'observed_fg',expected,population,settings,null=1.)))
        if len(bf):
            bf['density_excess']=bf.observed_density-bf.area_reference_density;bf['one']=1
            main.append(dict(model=model,endpoint='B3_density_excess',median=float(bf.density_excess.median()),**ratio(bf,'density_excess','one',population,settings)))
            for size,g in bf.groupby('size_bin'):
                main.append(dict(model=model,endpoint='B3_density_excess_'+size,median=float(g.density_excess.median()),**ratio(g,'density_excess','one',population,settings)))
            export(report/f'B_area_shift_audit_{model}.csv',bf.groupby('seed')[['observed_fg','added_area','shift_expected_fg','shift_expected_area','shift_zero_area_draws','shift_fg_mc_se']].sum().reset_index())
        cf=c[c.model==model].copy()
        cp=sorted({str(i) for i in results.done('C',f'mscoco_{model}_seed0')})
        for outcome in ('sam_iou','sam_precision'):
            res,oof,audit=cv_increment(cf,settings,cp,outcome)
            row=dict(model=model,endpoint='C1_delta_R2_'+outcome,**res);main.append(row)
            if outcome=='sam_iou':primary.append(row.copy())
            export(report/f'C_oof_{model}_{outcome}.csv',oof);export(report/f'C_folds_{model}_{outcome}.csv',audit)
            for seed,g in oof.groupby('seed'):
                sst=((g[outcome]-g[outcome].mean())**2).sum()
                perseed.append(dict(model=model,seed=int(seed),endpoint='C1_delta_R2_'+outcome,value=fraction((g.extended_loss-g.extended_vc_loss).sum(),sst)))
        # Mechanism check: owner-independent foreground recall. Owned-mask agreement is secondary.
        mf=m[m.model==model].copy()
        for outcome,features in [('foreground_recall',['AC','ABO','BestIoGT']),('owner_mask_iou_iou',['AC','ABO','BestIoGT']),('owner_mask_iou_intersection',['AC','ABO','BestIoGT'])]:
            extra='local_VC_intersection' if outcome.endswith('intersection') else 'local_VC'
            base,_,_=grouped_ols(mf,features,settings,cp,outcome);aug,_,_=grouped_ols(mf,features+[extra],settings,cp,outcome)
            sst=((mf[outcome]-mf[outcome].mean())**2).sum()
            mechanism.append(dict(model=model,outcome=outcome,units=len(mf),delta_R2=fraction((base-aug).sum(),sst),interpretation='descriptive mechanism; no primary significance claim'))
    if len(primary)!=12:raise AssertionError('Expected exactly twelve frozen primary tests')
    export(report/'main_results.csv',main);export(report/'primary_tests_holm.csv',holm(primary));export(report/'B_C_per_seed.csv',perseed);export(report/'C_mechanism.csv',mechanism)
    ps=pd.DataFrame(perseed)
    export(report/'B_C_seed_mean_SD.csv',ps.groupby(['model','endpoint']).value.agg(['mean','std','count']).reset_index())
    seeds=pd.DataFrame(seedrows)
    export(report/'A_seed_mean_SD.csv',seeds.groupby(['dataset','model'])[['A1_fraction','A2_gain_fraction','A2_strict_fraction','A2_coverage_fraction']].agg(['mean','std','count']).reset_index().pipe(flatten_columns))
    atomic_json(report/'report_manifest.json',clean(dict(smoke=settings.smoke,primary_family=12,seed_SD='sample SD, ddof=1',
        bootstrap='paired image-cluster percentile; conditional on frozen strata, detectors, reference draws and CV fits',
        p_values='one-sided recentered bootstrap approximation; Holm across A3/B1/B2/C1 x three models',
        cv='5 image folds x 10 repetitions in formal mode; mean repeated OOF squared errors, no clipping',
        unimplemented_optional=['D1 true crowd with unfiltered predictions','E1/E2 exploratory FP and selector-set attribution','F top-k qualitative panels'])))
    lines=['# 本轮结果阅读顺序','', '此目录为小样本运行检查，不能用于论文结论。' if settings.smoke else '正式结果已生成；请根据效应大小、区间和排除比例解释，不预设正结果。','',
        '1. A_per_seed.csv / A_seed_mean_SD.csv：FN 中多预测支持及真实前景增益的比例。',
        '2. A3_strata_*.csv：哪些预设分层可估计、哪些因稀疏或 tercile 并列而排除。',
        '3. main_results.csv：A3、前景富集 B1/B2、密度差 B3、VC 增量 R²。',
        '4. primary_tests_holm.csv：12 个预定主比较统一校正；缺失比较仍占一个检验名额。',
        '5. C_oof_*.csv / C_folds_*.csv：验证折、预测和误差；同图所有 seed/class 始终在同一折。',
        '6. C_mechanism.csv：前景 recall 及两种 ownership 的机制补充，不能替代主终点。',
        '7. D2_per_seed.csv：受控标注合并、背景占比、覆盖偏差与区域 VC。',
        '', 'B 的随机平移保留每个 clipped rectangle 的尺寸，但不保证新增 union 面积固定；同时阅读 B_area_shift_audit。',
        'C 的区间是在固定交叉验证拟合下重采样图像；未包含重新训练/重新拟合的不确定性。',
        'D2 是 controlled coarsening，不是对真实粗粒度数据部署价值的证明。',
        '原有 SAM 实验中 VC 与 F1/LRP 的下游比较仍保留原结论，本轮不替换它。']
    (report/'阅读结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def flatten_columns(frame):
    frame.columns=['_'.join(str(x) for x in col if x) if isinstance(col,tuple) else col for col in frame.columns]
    return frame
