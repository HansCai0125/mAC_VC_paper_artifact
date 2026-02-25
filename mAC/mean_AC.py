from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


# ---------- 基础数据结构 ----------
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
    """返回 a∩b 的矩形（若无交集则 None）。"""
    x1 = max(a.x1, b.x1); y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2); y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)

def union_area(rects: List[Box]) -> float:
    """
    稳定版：x-切片法。对任意一组轴对齐矩形，精确计算并集面积。
    复杂度 O(K^2)（K为矩形数），但K通常很小；对评测足够快且更不容易踩边界坑。
    """
    # 过滤非法矩形
    rects = [r for r in rects if r.x2 > r.x1 and r.y2 > r.y1]
    if not rects:
        return 0.0

    # 1) 收集所有x边界并排序
    xs = sorted({r.x1 for r in rects} | {r.x2 for r in rects})
    area = 0.0

    # 2) 在每个相邻x切片上累加 y 覆盖长度 * 切片宽度
    for i in range(len(xs) - 1):
        xL, xR = xs[i], xs[i + 1]
        dx = xR - xL
        if dx <= 0:
            continue

        # 2.1 当前x切片里“活跃”的矩形：与该切片有交
        intervals = []
        for r in rects:
            if r.x1 < xR and r.x2 > xL:     # 与当前x条相交
                intervals.append((r.y1, r.y2))

        if not intervals:
            continue

        # 2.2 合并y区间求覆盖长度
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
                    cur_e = max(cur_e, y2)   # 扩展
                else:
                    covered_y += (cur_e - cur_s)
                    cur_s, cur_e = y1, y2
        if cur_s is not None:
            covered_y += (cur_e - cur_s)

        area += covered_y * dx

    return area


    # 辅助：根据 active 覆盖区间计算 y 覆盖长度
    def y_covered_length(intervals: List[Tuple[float, float, int]]) -> float:
        # 将 count>0 的区间合并求长度
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

    # 我们维护一个“多重区间”列表：用离散化更新 counts
    # 为了简单可靠，每个事件点后重建一次 active（规模通常很小）
    counts: List[Tuple[float, float, int]] = []  # (y1,y2,count)

    def add_interval(y1, y2, delta):
        nonlocal counts
        # 简单实现：将现有区间拆分到不重叠段后更新计数
        new = []
        inserted = False
        for a, b, c in counts:
            if y2 <= a or y1 >= b:
                new.append((a, b, c))
                continue
            # 有重叠，拆成最多三段
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
        # 合并相邻相同 count 段
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

# ---------- 单个 GT 的 Coverage ----------
def coverage_for_one_gt(gt: Box, preds: List[Box]) -> float:
    """Coverage(G_i, P(G_i)) = |⋃_j (P_j ∩ G_i)| / |G_i|"""
    gt_area = gt.area()
    if gt_area <= 0:
        return 0.0
    # 先把所有预测框裁剪到 GT 内部，再做并集
    clipped: List[Box] = []
    for p in preds:
        inter = intersect(p, gt)
        if inter is not None:
            clipped.append(inter)
    cov_area = union_area(clipped)
    return float(cov_area / gt_area)

# ---------- 一张图上的 mAC / 命中 ----------
#parameters
def image_ac(
    gts: List[Box],
    preds: List[Box],
    tau: float = 0.5,
    score_thresh: float = 0.0,
    class_aware: bool = True
) -> Tuple[float, List[float], List[bool]]:
    """
    返回 (mAC, per_gt_coverages, per_gt_detected_flags).
    - 若 class_aware=True，仅用同类预测框计算覆盖；否则忽略类别。
    - 可通过 score_thresh 过滤低分预测。
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

# ---------- 多图聚合 ----------
def dataset_mac(
    dataset: List[Tuple[List[Box], List[Box]]],
    tau: float = 0.5,
    score_thresh: float = 0.0,
    class_aware: bool = True
) -> Dict[str, float]:
    """
    dataset: [(gts_img0, preds_img0), (gts_img1, preds_img1), ...]
    返回整体 mAC 与召回(按 tau 判定)。
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