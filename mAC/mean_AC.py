from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


# ---------- Basic Data Structure ----------
@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: Optional[int] = None
    conf: Optional[float] = None

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def intersect(a: Box, b: Box) -> Optional[Box]:
    """Return rectangle a∩b; return None if there is no overlap."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)


def union_area(rects: List[Box]) -> float:
    """
    Stable x-slicing method. Precisely computes union area for axis-aligned rectangles.
    Complexity is O(K^2) (K = number of rectangles), but K is usually small and robust for evaluation.
    """
    # Filter invalid rectangles
    rects = [r for r in rects if r.x2 > r.x1 and r.y2 > r.y1]
    if not rects:
        return 0.0

    # 1) Collect and sort all x boundaries
    xs = sorted({r.x1 for r in rects} | {r.x2 for r in rects})
    area = 0.0

    # 2) For each adjacent x-slice, accumulate y-covered length * slice width
    for i in range(len(xs) - 1):
        xL, xR = xs[i], xs[i + 1]
        dx = xR - xL
        if dx <= 0:
            continue

        # 2.1 Rectangles active in this x-slice (overlap with current slice)
        intervals = []
        for r in rects:
            if r.x1 < xR and r.x2 > xL:  # Overlaps current x-slice
                intervals.append((r.y1, r.y2))

        if not intervals:
            continue

        # 2.2 Merge y-intervals to get covered y-length
        intervals.sort()
        covered_y = 0.0
        cur_s, cur_e = None, None
        for y1, y2 in intervals:
            if y2 <= y1:
                continue
            if cur_s is None:
                cur_s, cur_e = y1, y2
            else:
                if y1 <= cur_e:
                    cur_e = max(cur_e, y2)  # Extend
                else:
                    covered_y += (cur_e - cur_s)
                    cur_s, cur_e = y1, y2
        if cur_s is not None:
            covered_y += (cur_e - cur_s)

        area += covered_y * dx

    return area

    # Helper: compute y-covered length from active intervals
    def y_covered_length(intervals: List[Tuple[float, float, int]]) -> float:
        # Merge and sum intervals where count>0
        if not intervals:
            return 0.0
        segs = []
        for y1, y2, c in intervals:
            if c > 0 and y2 > y1:
                segs.append((y1, y2))
        if not segs:
            return 0.0
        segs.sort()
        total = 0.0
        cur_s, cur_e = segs[0]
        for s, e in segs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += (cur_e - cur_s)
                cur_s, cur_e = s, e
        total += (cur_e - cur_s)
        return total

    # Maintain a multi-interval list; update counts via discretized segments
    # For simplicity and robustness, rebuild active segments after each event point (usually small scale)
    counts: List[Tuple[float, float, int]] = []  # (y1,y2,count)

    def add_interval(y1, y2, delta):
        nonlocal counts
        # Simple implementation: split existing intervals into non-overlapping segments, then update counts
        new = []
        inserted = False
        for a, b, c in counts:
            if y2 <= a or y1 >= b:
                new.append((a, b, c))
                continue
            # Overlap exists, split into at most three segments
            if y1 > a:
                new.append((a, y1, c))
                a = y1
            mid_r = min(b, y2)
            new.append((a, mid_r, c + delta))
            if mid_r < b:
                new.append((mid_r, b, c))
            y1 = mid_r
            inserted = True
        if not inserted:
            new.append((y1, y2, delta))
        # Merge adjacent segments with the same count
        new.sort()
        merged = []
        for a, b, c in new:
            if not merged:
                merged.append((a, b, c))
            else:
                pa, pb, pc = merged[-1]
                if abs(pb - a) < 1e-9 and pc == c:
                    merged[-1] = (pa, b, pc)
                else:
                    merged.append((a, b, c))
        counts = merged

    for x, typ, y1, y2 in events:
        dx = x - prev_x
        if dx > 0:
            area += dx * y_covered_length(counts)
            prev_x = x
        add_interval(y1, y2, +1 if typ == +1 else -1)

    return area


# ---------- Coverage for One GT ----------
def coverage_for_one_gt(gt: Box, preds: List[Box]) -> float:
    """Coverage(G_i, P(G_i)) = |⋃_j (P_j ∩ G_i)| / |G_i|"""
    gt_area = gt.area()
    if gt_area <= 0:
        return 0.0
    # First clip all predicted boxes to GT, then compute union
    clipped: List[Box] = []
    for p in preds:
        inter = intersect(p, gt)
        if inter is not None:
            clipped.append(inter)
    cov_area = union_area(clipped)
    return float(cov_area / gt_area)


# ---------- mAC / Hit on One Image ----------
# parameters
def image_ac(
    gts: List[Box],
    preds: List[Box],
    tau: float = 0.5,
    score_thresh: float = 0.0,
    class_aware: bool = True
) -> Tuple[float, List[float], List[bool]]:
    """
    Return (mAC, per_gt_coverages, per_gt_detected_flags).
    - If class_aware=True, only same-class predictions are used for coverage; otherwise class is ignored.
    - Low-score predictions can be filtered by score_thresh.
    """
    if score_thresh is not None:
        preds = [p for p in preds if (p.conf is None or p.conf >= score_thresh)]

    coverages = []
    detected = []
    for gt in gts:
        cand = preds
        if class_aware and gt.cls is not None:
            cand = [p for p in preds if p.cls == gt.cls]
        c = coverage_for_one_gt(gt, cand)
        coverages.append(c)
        detected.append(c >= tau)

    mAC = float(np.mean(coverages)) if coverages else 0.0
    return mAC, coverages, detected


# ---------- Multi-image Aggregation ----------
def dataset_mac(
    dataset: List[Tuple[List[Box], List[Box]]],
    tau: float = 0.5,
    score_thresh: float = 0.0,
    class_aware: bool = True
) -> Dict[str, float]:
    """
    dataset: [(gts_img0, preds_img0), (gts_img1, preds_img1), ...]
    Return dataset-level mAC and recall (judged by tau).
    """
    all_cov = []
    tp = 0
    total = 0
    for gts, preds in dataset:
        m, covs, det = image_ac(gts, preds, tau, score_thresh, class_aware)
        all_cov.extend(covs)
        tp += sum(det)
        total += len(gts)
    mac = float(np.mean(all_cov)) if all_cov else 0.0
    recall = (tp / total) if total > 0 else 0.0
    return {"mAC": mac, "Recall@tau": recall, "N_gt": total}
