"""Read-only, equivalent fast scorer for the fixed-cache figure refinement.

No change to metric definitions: exact rectangle unions, original greedy
matching/ties, class-macro 101-point AP, and GT-macro regional aggregation.
Only image/class grouping and pairwise geometry are shared between metrics.
The driver requires parity with the original evaluator and archived anchors.
"""
from collections import defaultdict
from math import fsum
import statistics
import numpy as np

from region_eval.geometry import union_area

IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def score(document, intended=None):
    events = defaultdict(list)
    class_gt = defaultdict(int)
    coverages, agreements = [], []
    n_pred = n_gt = n_orphan = changed = intended_seen = 0
    for image in document['images']:
        groups_g, groups_p = defaultdict(list), defaultdict(list)
        for g in image['gts']:
            groups_g[g['class_id']].append(g)
        for p in image['predictions']:
            if p['score'] >= 0.25:
                groups_p[p['class_id']].append(p)
        for cls in sorted(set(groups_g) | set(groups_p)):
            gs = sorted(groups_g[cls], key=lambda x: x['id'])
            ps = sorted(groups_p[cls], key=lambda x: (-x['score'], x['id']))
            ng, npred = len(gs), len(ps)
            n_gt += ng
            n_pred += npred
            class_gt[cls] += ng
            tp = np.zeros((npred, 10), dtype=np.bool_)
            matched_iou = np.zeros(npred, dtype=np.float64)
            if ng and npred:
                g = np.asarray([r['xyxy'] for r in gs], dtype=np.float64)
                p = np.asarray([r['xyxy'] for r in ps], dtype=np.float64)
                lo = np.maximum(p[:, None, :2], g[None, :, :2])
                hi = np.minimum(p[:, None, 2:], g[None, :, 2:])
                wh = np.maximum(0.0, hi - lo)
                inter = wh[:, :, 0] * wh[:, :, 1]
                ga = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
                pa = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
                iou = inter / (pa[:, None] + ga[None, :] - inter)
                max_iou = iou.max(axis=1)
                # GTs already sorted by stable ID; argmax breaks the final tie.
                tie_areas = np.where(iou == max_iou[:, None], inter, -1.0)
                owner = tie_areas.argmax(axis=1)
                owner[max_iou == 0.0] = -1
                n_orphan += int(np.count_nonzero(owner < 0))
                for j in range(ng):
                    hits = np.flatnonzero(inter[:, j] > 0.0)
                    clips = [tuple(lo[k, j]) + tuple(hi[k, j]) for k in hits]
                    coverages.append(min(1.0, union_area(clips) / float(ga[j])))
                    own = np.flatnonzero(owner == j)
                    union = union_area([tuple(p[k]) for k in own])
                    covered = min(float(ga[j]), union_area([
                        tuple(lo[k, j]) + tuple(hi[k, j]) for k in own]))
                    agreements.append(2.0 * covered / (union + float(ga[j])))
                if intended is not None:
                    for k, pr in enumerate(ps):
                        expected = intended.get((image['image_id'], pr['id']))
                        if expected is not None:
                            intended_seen += 1
                            if owner[k] < 0:
                                raise AssertionError('generated fragment became orphaned')
                            changed += gs[int(owner[k])]['id'] != expected
                free = np.ones((10, ng), dtype=np.bool_)
                # Almost all fragments have max IoU below .5; skip them safely.
                for k in np.flatnonzero(max_iou >= 0.5):
                    candidates = np.flatnonzero(iou[k] >= 0.5)
                    candidates = sorted(candidates, key=lambda j: (
                        -iou[k, j], -inter[k, j], int(j)))
                    for t, threshold in enumerate(IOU_THRESHOLDS):
                        for j in candidates:
                            if iou[k, j] < threshold:
                                break
                            if free[t, j]:
                                free[t, j] = False
                                tp[k, t] = True
                                if t == 0:
                                    matched_iou[k] = iou[k, j]
                                break
            else:
                coverages.extend([0.0] * ng)
                agreements.extend([0.0] * ng)
                n_orphan += npred
            if npred:
                # Original AP order: descending score, image ID, prediction ID.
                events[cls].extend((pr['score'], image['image_id'], pr['id'],
                                    tp[k].copy(), float(matched_iou[k]))
                                   for k, pr in enumerate(ps))
    ap, ar = [], []
    tp_total = 0
    loc_terms = []
    for cls in sorted(set(events) | set(class_gt)):
        es = events[cls]
        es.sort(key=lambda e: (-e[0], e[1], e[2]))
        gt = class_gt[cls]
        if es:
            flags = np.asarray([e[3] for e in es], dtype=np.bool_)
            tp_total += int(flags[:, 0].sum())
            loc_terms.extend((1.0-e[4]) / 0.5 for e in es if e[3][0])
        if not gt:
            continue
        if not es:
            ap.append(np.zeros(10)); ar.append(np.zeros(10)); continue
        cumulative = np.cumsum(flags, axis=0)
        recalls = cumulative / gt
        precision = cumulative / np.arange(1, len(es)+1)[:, None]
        precision = np.maximum.accumulate(precision[::-1], axis=0)[::-1]
        values = []
        for t in range(10):
            ix = np.searchsorted(recalls[:, t], np.arange(101)/100.0, side='left')
            padded = np.append(precision[:, t], 0.0)
            values.append(statistics.fmean(padded[ix].tolist()))
        ap.append(np.asarray(values)); ar.append(recalls[-1])
    fp, fn = n_pred-tp_total, n_gt-tp_total
    denom = tp_total + fp + fn
    class_ap = np.asarray(ap)
    result = dict(theta=0.25, n_images=len(document['images']), n_gt=n_gt,
                  n_pred=n_pred, assigned_pred_count=n_pred-n_orphan,
                  orphan_pred_count=n_orphan,
                  orphan_rate=n_orphan/n_pred if n_pred else 0.0,
                  orphan_per_gt=n_orphan/n_gt if n_gt else None,
                  mAC=fsum(coverages)/n_gt if n_gt else None,
                  VC=fsum(agreements)/(n_gt+n_orphan) if n_gt+n_orphan else None,
                  F1=2*tp_total/(n_pred+n_gt) if n_pred+n_gt else 0.0,
                  LRP=(fsum(loc_terms)+fp+fn)/denom if denom else None,
                  AP50=statistics.fmean(class_ap[:,0].tolist()) if ap else None,
                  AP75=statistics.fmean(class_ap[:,5].tolist()) if ap else None,
                  AP50_95=statistics.fmean([statistics.fmean(a.tolist()) for a in ap]) if ap else None,
                  AR50_95=statistics.fmean([statistics.fmean(a.tolist()) for a in ar]) if ar else None)
    return result, dict(generated_fragments_checked=intended_seen,
                        ownership_change_count=int(changed),
                        ownership_change_rate=changed/intended_seen if intended_seen else 0.0,
                        fragment_orphan_count=0)
