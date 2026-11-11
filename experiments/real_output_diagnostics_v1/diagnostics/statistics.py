"""Image-cluster uncertainty and leakage-free, repeated grouped OLS."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import seed_for

def bootstrap_weights(n, settings):
    # Reset to the same stream for every model/endpoint with this image universe.
    rng=np.random.default_rng(settings.random_seed)
    for start in range(0,settings.bootstrap,32):
        yield rng.multinomial(n,np.full(n,1/n),size=min(32,settings.bootstrap-start))

def interval(point, draws, null=0.):
    draws=np.asarray(draws);draws=draws[np.isfinite(draws)]
    if point is None or not np.isfinite(point) or len(draws)<20:
        return dict(estimate=float(point) if point is not None and np.isfinite(point) else None,ci_low=None,ci_high=None,p_one_sided=None,valid_bootstrap=len(draws))
    # Approximate, recentered bootstrap test of H0: effect <= null.
    p=(1+np.count_nonzero(draws-point >= point-null))/(len(draws)+1)
    return dict(estimate=float(point),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),p_one_sided=float(p),valid_bootstrap=len(draws))

def ratio(frame, numerator, denominator, population, settings, null=0.):
    arr=frame.groupby('image_id')[[numerator,denominator]].sum().reindex(population,fill_value=0).to_numpy(float)
    totals=arr.sum(axis=0);point=totals[0]/totals[1] if totals[1]>0 else None
    draws=[]
    for w in bootstrap_weights(len(population),settings):
        v=w@arr
        draws.extend(np.divide(v[:,0],v[:,1],out=np.full(len(v),np.nan),where=v[:,1]>0))
    return interval(point,draws,null)

def holm(rows):
    p=np.array([r.get('p_one_sided') if r.get('p_one_sided') is not None else 1. for r in rows])
    order=np.argsort(p);adjusted=np.maximum.accumulate(p[order]*(len(rows)-np.arange(len(rows))))
    for j,i in enumerate(order):rows[i]['p_holm']=min(1.,float(adjusted[j]));rows[i]['holm_family_size']=len(rows)
    return rows

def residual_contrast(frame,population,settings):
    """Predefined strata and within-stratum terciles; no outcome-based merging."""
    f=frame[(frame.is_fn==1)&(frame.intersecting_pred_count>=2)&(frame.BestIoGT<.9)&frame.MaskCollectiveGain.notna()].copy()
    f['best_bin']=pd.cut(f.BestIoGT,[-np.inf,.3,.5,.7,.9],right=False,labels=False)
    f['k_bin']=np.where(f.intersecting_pred_count==2,'2',np.where(f.intersecting_pred_count<=4,'3-4','5+'))
    strata=[];audit=[];index={v:i for i,v in enumerate(population)}
    for key,g in f.groupby(['best_bin','k_bin','class_id','size_bin'],sort=True):
        nt=len(g[['image_id','gt_id']].drop_duplicates());ni=g.image_id.nunique()
        lo,hi=g.CollectiveGain.quantile([1/3,2/3]).to_numpy()
        status='eligible'
        if nt<settings.min_stratum_targets or ni<settings.min_stratum_images:status='sparse'
        elif hi<=lo:status='tied_terciles'
        audit.append(dict(stratum=str(key),unique_targets=nt,images=ni,low_cut=float(lo),high_cut=float(hi),status=status))
        if status!='eligible':continue
        audit[-1].update(low_observations=int((g.CollectiveGain<=lo).sum()),high_observations=int((g.CollectiveGain>=hi).sum()))
        for seed,sg in g.groupby('seed') if 'seed' in g else []:
            low=sg[sg.CollectiveGain<=lo].MaskCollectiveGain
            high=sg[sg.CollectiveGain>=hi].MaskCollectiveGain
            audit[-1][f'seed_{int(seed)}_contrast']=float(high.mean()-low.mean()) if len(low) and len(high) else None
        arr=np.zeros((len(population),4))
        for offset,subset in ((0,g[g.CollectiveGain<=lo]),(2,g[g.CollectiveGain>=hi])):
            z=subset.groupby('image_id').MaskCollectiveGain.agg(['sum','count'])
            for im,row in z.iterrows():arr[index[im],offset:offset+2]=row.to_numpy()
        strata.append((nt,arr))
    if not strata:return interval(None,[]),audit
    weights=np.array([s[0] for s in strata],float);weights/=weights.sum()
    arr=np.stack([s[1] for s in strata],axis=1)
    def effect(t):
        valid=(t[...,1]>0)&(t[...,3]>0)
        with np.errstate(invalid='ignore',divide='ignore'):
            v=(t[...,2]/t[...,3]-t[...,0]/t[...,1])@weights
        return np.where(valid.all(axis=-1),v,np.nan)
    point=float(effect(arr.sum(axis=0)));draws=[]
    for w in bootstrap_weights(len(population),settings):draws.extend(effect(np.einsum('bi,isk->bsk',w,arr)))
    result=interval(point,draws);result.update(eligible_strata=len(strata),eligible_unique_targets=sum(s[0] for s in strata),candidate_unique_targets=len(f[['image_id','gt_id']].drop_duplicates()))
    return result,audit

COMPACT=['union_dice','n_prompts']
EXTENDED=COMPACT+['mAC','F1','LRP','score_mean','score_max','score_std','BestIoGT']

def grouped_ols(frame,features,settings,population,outcome='sam_iou'):
    """Fold assignment shared across detectors, seeds, classes and endpoints."""
    x=frame[features].to_numpy(float);y=frame[outcome].to_numpy(float)
    ids=frame.image_id.to_numpy();loss=np.zeros(len(y));pred=np.zeros(len(y));audit=[]
    for repeat in range(settings.cv_repeats):
        rng=np.random.default_rng(seed_for(settings.random_seed,'cv',repeat))
        perm=rng.permutation(np.array(population));lookup={im:i%settings.cv_folds for i,im in enumerate(perm)}
        folds=np.array([lookup[im] for im in ids])
        for fold in range(settings.cv_folds):
            test=folds==fold;train=~test
            if not test.any():continue
            if not train.any():raise ValueError('No training images in grouped CV')
            mu=x[train].mean(axis=0);sd=x[train].std(axis=0);sd[sd<1e-12]=1.
            xt=np.column_stack([np.ones(train.sum()),(x[train]-mu)/sd])
            beta,_,rank,_=np.linalg.lstsq(xt,y[train],rcond=None)
            yp=np.column_stack([np.ones(test.sum()),(x[test]-mu)/sd])@beta
            pred[test]+=yp/settings.cv_repeats;loss[test]+=(y[test]-yp)**2/settings.cv_repeats
            audit.append(dict(repeat=repeat,fold=fold,train_images=len(set(ids[train])),test_images=len(set(ids[test])),shared_images=len(set(ids[train])&set(ids[test])),rank=int(rank),columns=len(features)+1))
    return loss,pred,audit

def cv_increment(frame,settings,population,outcome='sam_iou'):
    f=frame.copy();audits=[]
    for name,features in [('compact',COMPACT),('extended',EXTENDED),('extended_vc',EXTENDED+['VC'])]:
        loss,pred,audit=grouped_ols(f,features,settings,population,outcome)
        f[name+'_loss']=loss;f[name+'_prediction']=pred
        audits.extend(dict(model_spec=name,outcome=outcome,**a) for a in audit)
    f['n']=1;f['y']=f[outcome];f['y2']=f.y**2;f['loss_improvement']=f.extended_loss-f.extended_vc_loss
    columns=['n','y','y2','loss_improvement','extended_loss','extended_vc_loss','compact_loss']
    arr=f.groupby('image_id')[columns].sum().reindex(population,fill_value=0).to_numpy(float)
    def calc(t):
        with np.errstate(divide='ignore',invalid='ignore'):
            sst=t[...,2]-t[...,1]**2/t[...,0]
            return np.where(sst>1e-12,t[...,3]/sst,np.nan)
    total=arr.sum(axis=0);point=float(calc(total));draws=[]
    for w in bootstrap_weights(len(population),settings):draws.extend(calc(w@arr))
    result=interval(point,draws);sst=total[2]-total[1]**2/total[0]
    for j,name in ((4,'extended'),(5,'extended_vc'),(6,'compact')):result[name+'_R2']=float(1-total[j]/sst) if sst>1e-12 else None
    result['units']=len(f);result['images']=f.image_id.nunique()
    return result,f,audits
