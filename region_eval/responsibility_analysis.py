"""Prediction-to-GT responsibility ablation on an existing diagnostic cache."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from .case_analysis import (
    SIZE_BINS,
    _file_hash,
    _write_csv,
    _write_json,
    harmonic,
    local_gt_diagnostics,
    validate_source,
)
from .geometry import area, intersection, union_area
from .offline import PROJECT_ROOT, evaluate_document


ANALYSIS_VERSION = "region-responsibility-ablation-0.1.0"
METHODS = ("hard_iou", "center", "soft_overlap")


def rectangle_iou(first, second):
    overlap = intersection(tuple(first), tuple(second))
    overlap_area = area(overlap) if overlap is not None else 0.0
    denominator = area(tuple(first)) + area(tuple(second)) - overlap_area
    return overlap_area / denominator if denominator else 0.0


def _hit_rows(gts, prediction):
    rows = []
    pred_rect = tuple(prediction["xyxy"])
    for gt in gts:
        overlap = intersection(tuple(gt["xyxy"]), pred_rect)
        if overlap is not None:
            rows.append((gt, area(overlap), rectangle_iou(gt["xyxy"], pred_rect)))
    return rows


def assignment_weights(gts, predictions, method):
    """Return pred_id -> {gt_id: responsibility}; weights sum to at most one."""
    if method not in METHODS:
        raise ValueError(f"unknown responsibility method: {method}")
    result = {}
    for prediction in sorted(predictions, key=lambda row: row["id"]):
        hits = _hit_rows(gts, prediction)
        if not hits:
            result[prediction["id"]] = {}
            continue
        if method == "soft_overlap":
            total = sum(overlap_area for _, overlap_area, _ in hits)
            result[prediction["id"]] = {
                gt["id"]: overlap_area / total
                for gt, overlap_area, _ in sorted(hits, key=lambda row: row[0]["id"])
            }
            continue
        candidates = hits
        if method == "center":
            x1, y1, x2, y2 = prediction["xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            candidates = [
                row for row in hits
                if row[0]["xyxy"][0] <= cx <= row[0]["xyxy"][2]
                and row[0]["xyxy"][1] <= cy <= row[0]["xyxy"][3]
            ]
            if not candidates:
                result[prediction["id"]] = {}
                continue
        selected = sorted(
            candidates,
            key=lambda row: (-row[2], -row[1], row[0]["id"]),
        )[0][0]
        result[prediction["id"]] = {selected["id"]: 1.0}
    return result


def weighted_union_area(rectangle_weights, clip=None):
    """Exact integral of the maximum active responsibility weight."""
    rects = []
    for rect, weight in rectangle_weights:
        if not 0.0 < weight <= 1.0:
            raise ValueError("responsibility weights must be in (0, 1]")
        rect = tuple(rect)
        if clip is not None:
            rect = intersection(rect, tuple(clip))
            if rect is None:
                continue
        if area(rect) > 0:
            rects.append((rect, float(weight)))
    xs = sorted({value for rect, _ in rects for value in (rect[0], rect[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        active_x = [(rect, weight) for rect, weight in rects
                    if rect[0] < right and rect[2] > left]
        ys = sorted({value for rect, _ in active_x for value in (rect[1], rect[3])})
        for top, bottom in zip(ys, ys[1:]):
            weights = [weight for rect, weight in active_x
                       if rect[1] < bottom and rect[3] > top]
            if weights:
                occupancy = max(weights)
                total += (right - left) * (bottom - top) * occupancy
    return total


def _score_pair(image_id, class_id, gts, predictions, method, theta, tau):
    weights = assignment_weights(gts, predictions, method)
    hits = {
        prediction["id"]: _hit_rows(gts, prediction)
        for prediction in predictions
    }
    rows = []
    for gt in sorted(gts, key=lambda row: row["id"]):
        gt_rect = tuple(gt["xyxy"])
        gt_area = area(gt_rect)
        assigned = []
        for prediction in predictions:
            weight = weights[prediction["id"]].get(gt["id"], 0.0)
            if weight > 0:
                assigned.append((prediction, weight))
        weighted_rects = [(tuple(prediction["xyxy"]), weight)
                          for prediction, weight in assigned]
        support_area = weighted_union_area(weighted_rects)
        covered_area = min(gt_area, weighted_union_area(weighted_rects, clip=gt_rect))
        responsibility_coverage = covered_area / gt_area
        responsibility_precision = covered_area / support_area if support_area else 0.0
        responsibility_dice = harmonic(responsibility_coverage, responsibility_precision)
        best_responsibility_coverage = max(
            (weight * area(intersection(gt_rect, tuple(prediction["xyxy"]))) / gt_area
             for prediction, weight in assigned),
            default=0.0,
        )
        original_intersections = [
            overlap for prediction in predictions
            if (overlap := intersection(gt_rect, tuple(prediction["xyxy"]))) is not None
        ]
        original_coverage = min(1.0, union_area(original_intersections) / gt_area)
        original_best = max(
            (area(overlap) / gt_area for overlap in original_intersections),
            default=0.0,
        )
        rows.append(dict(
            image_id=image_id,
            gt_id=gt["id"],
            class_id=class_id,
            theta=theta,
            method=method,
            gt_area=gt_area,
            original_AC=original_coverage,
            original_BestIoGT=original_best,
            original_CollectiveGain=original_coverage - original_best,
            responsibility_coverage=responsibility_coverage,
            responsibility_precision=responsibility_precision,
            responsibility_dice=responsibility_dice,
            best_responsibility_coverage=best_responsibility_coverage,
            responsibility_gain=max(
                0.0, responsibility_coverage - best_responsibility_coverage),
            coverage_hit=responsibility_coverage >= tau,
            assigned_pred_count=len(assigned),
            assigned_weight_sum=sum(weight for _, weight in assigned),
            assigned_multigt_source_count=sum(
                len(hits[prediction["id"]]) >= 2 for prediction, _ in assigned),
            assigned_pred_ids=";".join(prediction["id"] for prediction, _ in assigned),
            assigned_weights=";".join(
                f"{weight:.12g}" for _, weight in assigned),
            coverage_removed=max(0.0, original_coverage - responsibility_coverage),
        ))
    assigned_predictions = sum(bool(value) for value in weights.values())
    unassigned_intersecting = sum(
        bool(hits[prediction["id"]]) and not weights[prediction["id"]]
        for prediction in predictions
    )
    ambiguous_weight = sum(
        1.0 - max(value.values()) for value in weights.values() if value
    )
    pair = dict(
        image_id=image_id,
        class_id=class_id,
        theta=theta,
        method=method,
        n_gt=len(gts),
        n_pred=len(predictions),
        assigned_pred_count=assigned_predictions,
        unassigned_pred_count=len(predictions) - assigned_predictions,
        orphan_pred_count=sum(not hits[prediction["id"]] for prediction in predictions),
        unassigned_intersecting_pred_count=unassigned_intersecting,
        multigt_source_pred_count=sum(
            len(hits[prediction["id"]]) >= 2 for prediction in predictions),
        ambiguous_responsibility_weight=ambiguous_weight,
    )
    return rows, pair


def score_document(document, theta, method, tau=0.5):
    """Score all GT after assigning each same-class prediction responsibility."""
    class_ids = tuple(document["metadata"]["class_ids"])
    gt_rows, pair_rows = [], []
    for image in document["images"]:
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)
        for gt in image["gts"]:
            gt_by_class[gt["class_id"]].append(gt)
        for prediction in image["predictions"]:
            if prediction["score"] >= theta:
                pred_by_class[prediction["class_id"]].append(prediction)
        active_classes = sorted(set(gt_by_class) | set(pred_by_class))
        if any(class_id not in class_ids for class_id in active_classes):
            raise ValueError("document contains an undeclared class")
        for class_id in active_classes:
            rows, pair = _score_pair(
                image["image_id"],
                class_id,
                gt_by_class[class_id],
                pred_by_class[class_id],
                method,
                theta,
                tau,
            )
            gt_rows.extend(rows)
            pair_rows.append(pair)
    return (
        sorted(gt_rows, key=lambda row: (
            row["image_id"], row["class_id"], row["gt_id"])),
        sorted(pair_rows, key=lambda row: (row["image_id"], row["class_id"])),
    )


def aggregate_responsibility(gt_rows, pair_rows, original_summary, class_pair_rows):
    def mean(field):
        return statistics.fmean(row[field] for row in gt_rows) if gt_rows else None

    pair_precision = [
        (row["SpatialPrecision"] if row["SpatialPrecision"] is not None else 0.0)
        for image in class_pair_rows for row in image.per_class
    ]
    responsibility_dice = mean("responsibility_dice")
    class_dice = original_summary["ClassUnionDice"]
    return dict(
        method=gt_rows[0]["method"] if gt_rows else pair_rows[0]["method"],
        theta=original_summary["theta"],
        n_gt=len(gt_rows),
        n_pred=original_summary["n_pred"],
        original_mAC=original_summary["mAC"],
        original_CollectiveGain=original_summary["CollectiveGain"],
        ClassUnionDice=class_dice,
        ClassUnionIoU=original_summary["ClassUnionIoU"],
        mean_pair_spatial_precision=(
            statistics.fmean(pair_precision) if pair_precision else None),
        mean_responsibility_coverage=mean("responsibility_coverage"),
        mean_responsibility_precision=mean("responsibility_precision"),
        mean_responsibility_dice=responsibility_dice,
        mean_best_responsibility_coverage=mean(
            "best_responsibility_coverage"),
        mean_responsibility_gain=mean("responsibility_gain"),
        responsibility_gain_positive_fraction=statistics.fmean(
            row["responsibility_gain"] > 1e-9 for row in gt_rows),
        responsibility_recall_tau=statistics.fmean(
            row["coverage_hit"] for row in gt_rows),
        mean_coverage_removed=mean("coverage_removed"),
        two_level_H_responsibility_Dice_ClassUnionDice=harmonic(
            responsibility_dice, class_dice),
        assigned_pred_count=sum(row["assigned_pred_count"] for row in pair_rows),
        unassigned_pred_count=sum(row["unassigned_pred_count"] for row in pair_rows),
        orphan_pred_count=sum(row["orphan_pred_count"] for row in pair_rows),
        unassigned_intersecting_pred_count=sum(
            row["unassigned_intersecting_pred_count"] for row in pair_rows),
        multigt_source_pred_count=sum(
            row["multigt_source_pred_count"] for row in pair_rows),
        ambiguous_responsibility_weight=sum(
            row["ambiguous_responsibility_weight"] for row in pair_rows),
    )


def _size_summary(rows, method):
    result = []
    for size_name, lower, upper in SIZE_BINS:
        values = [row for row in rows if lower <= row["gt_area"] < upper]
        result.append(dict(
            method=method,
            size=size_name,
            lower_area=lower,
            upper_area=None if upper == float("inf") else upper,
            n_gt=len(values),
            mean_responsibility_coverage=statistics.fmean(
                row["responsibility_coverage"] for row in values),
            mean_responsibility_precision=statistics.fmean(
                row["responsibility_precision"] for row in values),
            mean_responsibility_dice=statistics.fmean(
                row["responsibility_dice"] for row in values),
            mean_responsibility_gain=statistics.fmean(
                row["responsibility_gain"] for row in values),
            responsibility_recall_tau=statistics.fmean(
                row["coverage_hit"] for row in values),
            mean_coverage_removed=statistics.fmean(
                row["coverage_removed"] for row in values),
        ))
    return result


def _candidate_groups(local_rows):
    positive = {
        (row["image_id"], row["gt_id"]) for row in local_rows
        if row["CollectiveGain_i"] > 1e-9
    }
    multigt = {
        (row["image_id"], row["gt_id"]) for row in local_rows
        if row["CollectiveGain_i"] > 1e-9
        and row["contributing_multigt_count"] > 0
    }
    low_overlap = {
        (row["image_id"], row["gt_id"]) for row in local_rows
        if row["CollectiveGain_i"] > 1e-9
        and row["contributing_pred_count"] >= 2
        and row["contributing_multigt_count"] == 0
        and row["max_contributing_pairwise_iou"] is not None
        and row["max_contributing_pairwise_iou"] < 0.3
    }
    return {
        "all_original_positive_CG": positive,
        "original_positive_CG_with_multigt": multigt,
        "single_gt_low_overlap_candidates": low_overlap,
    }


def _candidate_summary(rows_by_method, groups):
    output = []
    for method, rows in rows_by_method.items():
        by_key = {(row["image_id"], row["gt_id"]): row for row in rows}
        for group, keys in groups.items():
            values = [by_key[key] for key in sorted(keys)]
            original_total = sum(row["original_AC"] for row in values)
            output.append(dict(
                method=method,
                group=group,
                n_gt=len(values),
                mean_original_AC=statistics.fmean(
                    row["original_AC"] for row in values),
                mean_responsibility_coverage=statistics.fmean(
                    row["responsibility_coverage"] for row in values),
                coverage_retention=(
                    sum(row["responsibility_coverage"] for row in values)
                    / original_total if original_total else None),
                mean_coverage_removed=statistics.fmean(
                    row["coverage_removed"] for row in values),
                positive_responsibility_gain_count=sum(
                    row["responsibility_gain"] > 1e-9 for row in values),
                positive_responsibility_gain_fraction=statistics.fmean(
                    row["responsibility_gain"] > 1e-9 for row in values),
                mean_responsibility_gain=statistics.fmean(
                    row["responsibility_gain"] for row in values),
                mean_responsibility_dice=statistics.fmean(
                    row["responsibility_dice"] for row in values),
            ))
    return output


def _correction_cases(rows_by_method, class_names, count=12):
    method_maps = {
        method: {(row["image_id"], row["gt_id"]): row for row in rows}
        for method, rows in rows_by_method.items()
    }
    keys = method_maps[METHODS[0]]
    candidates = []
    for key, base in keys.items():
        if base["original_CollectiveGain"] <= 1e-9:
            continue
        method_rows = [method_maps[method][key] for method in METHODS]
        correction = statistics.fmean(
            row["coverage_removed"] for row in method_rows)
        candidates.append(dict(
            image_id=base["image_id"],
            gt_id=base["gt_id"],
            class_id=base["class_id"],
            class_name=class_names[base["class_id"]],
            gt_area=base["gt_area"],
            original_AC=base["original_AC"],
            original_CollectiveGain=base["original_CollectiveGain"],
            mean_coverage_correction=correction,
            hard_iou_coverage=method_rows[0]["responsibility_coverage"],
            hard_iou_dice=method_rows[0]["responsibility_dice"],
            hard_iou_gain=method_rows[0]["responsibility_gain"],
            center_coverage=method_rows[1]["responsibility_coverage"],
            center_dice=method_rows[1]["responsibility_dice"],
            center_gain=method_rows[1]["responsibility_gain"],
            soft_overlap_coverage=method_rows[2]["responsibility_coverage"],
            soft_overlap_dice=method_rows[2]["responsibility_dice"],
            soft_overlap_gain=method_rows[2]["responsibility_gain"],
        ))
    return sorted(
        candidates,
        key=lambda row: (
            -row["mean_coverage_correction"],
            -row["original_CollectiveGain"],
            row["image_id"],
            row["gt_id"],
        ),
    )[:count]


def overview_plot(threshold_rows, size_rows, candidate_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for method in METHODS:
        rows = [row for row in threshold_rows if row["method"] == method]
        axes[0, 0].plot(
            [row["theta"] for row in rows],
            [row["mean_responsibility_dice"] for row in rows],
            marker="o",
            label=method,
        )
        axes[0, 1].plot(
            [row["theta"] for row in rows],
            [row["two_level_H_responsibility_Dice_ClassUnionDice"]
             for row in rows],
            marker="o",
            label=method,
        )
    axes[0, 0].set(
        title="Instance responsibility quality",
        xlabel="Confidence threshold",
        ylabel="Mean per-GT responsibility Dice",
    )
    axes[0, 1].set(
        title="Exploratory two-level score",
        xlabel="Confidence threshold",
        ylabel="H(instance Dice, class-union Dice)",
    )
    for axis in axes[0]:
        axis.grid(alpha=.25)
        axis.legend(frameon=False)
        axis.set_ylim(0, .55)

    primary_sizes = [row for row in size_rows if row["size"] == "small"]
    x = np.arange(len(METHODS))
    axes[1, 0].bar(
        x - .18,
        [next(row for row in size_rows
              if row["method"] == method and row["size"] == "small")[
                  "mean_responsibility_coverage"] for method in METHODS],
        .36,
        label="coverage",
    )
    axes[1, 0].bar(
        x + .18,
        [next(row for row in primary_sizes if row["method"] == method)[
            "mean_responsibility_dice"] for method in METHODS],
        .36,
        label="Dice",
    )
    axes[1, 0].set(
        title="Small GT at theta=0.25",
        ylabel="Score",
        xticks=x,
        xticklabels=METHODS,
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_ylim(0, .6)

    groups = (
        "all_original_positive_CG",
        "original_positive_CG_with_multigt",
        "single_gt_low_overlap_candidates",
    )
    width = .24
    for index, method in enumerate(METHODS):
        values = [
            next(row for row in candidate_rows
                 if row["method"] == method and row["group"] == group)[
                     "positive_responsibility_gain_fraction"]
            for group in groups
        ]
        axes[1, 1].bar(
            np.arange(len(groups)) + (index - 1) * width,
            values,
            width,
            label=method,
        )
    axes[1, 1].set(
        title="Original CG cases retaining multi-prediction gain",
        ylabel="Fraction",
        xticks=np.arange(len(groups)),
        xticklabels=("all CG+", "CG+ with multi-GT", "low-overlap"),
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(frameon=False)
    figure.suptitle("VisDrone prediction-responsibility ablation", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _report_text(primary_rows, candidate_rows, source_run):
    by_method = {row["method"]: row for row in primary_rows}
    hard = by_method["hard_iou"]
    center = by_method["center"]
    soft = by_method["soft_overlap"]
    multigt = {
        row["method"]: row for row in candidate_rows
        if row["group"] == "original_positive_CG_with_multigt"
    }
    low = {
        row["method"]: row for row in candidate_rows
        if row["group"] == "single_gt_low_overlap_candidates"
    }
    return f"""# 第四轮：预测责任归属消融

来源运行：{source_run}。本轮只复用预测缓存，没有推理或训练。

## theta=0.25 主要结果

| 方法 | 责任覆盖 | 责任精度 | 逐 GT Dice | 责任 CG | 未归属相交框 | 两层 H |
|---|---:|---:|---:|---:|---:|---:|
| hard_iou | {hard['mean_responsibility_coverage']:.6f} | {hard['mean_responsibility_precision']:.6f} | {hard['mean_responsibility_dice']:.6f} | {hard['mean_responsibility_gain']:.6f} | {hard['unassigned_intersecting_pred_count']} | {hard['two_level_H_responsibility_Dice_ClassUnionDice']:.6f} |
| center | {center['mean_responsibility_coverage']:.6f} | {center['mean_responsibility_precision']:.6f} | {center['mean_responsibility_dice']:.6f} | {center['mean_responsibility_gain']:.6f} | {center['unassigned_intersecting_pred_count']} | {center['two_level_H_responsibility_Dice_ClassUnionDice']:.6f} |
| soft_overlap | {soft['mean_responsibility_coverage']:.6f} | {soft['mean_responsibility_precision']:.6f} | {soft['mean_responsibility_dice']:.6f} | {soft['mean_responsibility_gain']:.6f} | {soft['unassigned_intersecting_pred_count']} | {soft['two_level_H_responsibility_Dice_ClassUnionDice']:.6f} |

原始 mAC={hard['original_mAC']:.6f}，ClassUnionDice={hard['ClassUnionDice']:.6f}。两层 H 只是把实例责任 Dice 与图像—类别 Dice 做调和平均的探索值，不是最终 VC。

## 对原始 CG 的修正

原始正 CG 且含 multi-GT 贡献预测的 GT 共 {multigt['hard_iou']['n_gt']} 个。责任归属后仍保留正多框增益的比例：

- hard_iou：{multigt['hard_iou']['positive_responsibility_gain_fraction']:.2%}
- center：{multigt['center']['positive_responsibility_gain_fraction']:.2%}
- soft_overlap：{multigt['soft_overlap']['positive_responsibility_gain_fraction']:.2%}

原先 18 个 single-GT、低框间重叠候选中，仍保留正多框增益的比例：

- hard_iou：{low['hard_iou']['positive_responsibility_gain_fraction']:.2%}
- center：{low['center']['positive_responsibility_gain_fraction']:.2%}
- soft_overlap：{low['soft_overlap']['positive_responsibility_gain_fraction']:.2%}

hard_iou 强制每个预测最多归给一个 GT；center 还要求预测中心位于负责 GT 内；soft_overlap 按预测与各 GT 的交叠面积分配权重，并对每个位置取覆盖该位置的最大责任权重。完全重复预测因此不会增加软区域。

## 如何解释

若 hard_iou 能显著减少 multi-GT 组的增益，同时保留低重叠组，说明“每个预测只服务一个实例”是合理起点。center 若额外损失明显，说明中心点规则会拒绝真实的边缘碎片。soft_overlap 若仍保留过多 multi-GT 增益，说明软分摊没有充分解决跨实例借覆盖。

责任精度把分配预测落在当前 GT 之外的全部区域视为代价；远处 orphan 则继续由图像—类别层的 ClassUnionDice 处理。这样两个层级各承担一次误差，避免把全局误检重复分摊到每个 GT。

## 输出

- method_threshold.csv：三种方法、五个阈值的汇总。
- responsibility_gt.csv：theta=0.25 的全部逐 GT 结果。
- size_method_summary.csv：small、medium、large 分层。
- candidate_preservation.csv：原始 CG 分组在责任归属后的保留情况。
- correction_cases.csv：责任归属修正最大的 12 个 GT。
- overview.png：阈值、两层探索值、小目标和 CG 保留率。

## 下一步判断

根据本表优先选择一种责任规则作为实例层基线，再对 correction_cases 和之前的低重叠联系表做一次针对性看图。只有责任规则稳定后，才确定实例层与图像—类别层是否合成为一个 VC，以及采用调和平均还是保持二维报告。
"""


def analyze_responsibility(
        source_run,
        *,
        thresholds=(0.05, 0.10, 0.25, 0.50, 0.75),
        primary_theta=0.25,
        tau=0.5,
        output=None):
    source_run, document, _, _, complete, _ = validate_source(source_run)
    thresholds = tuple(float(value) for value in thresholds)
    if primary_theta not in thresholds:
        raise ValueError("primary theta must be included in thresholds")
    output = (
        Path(output).resolve() if output else
        PROJECT_ROOT / "outputs" / "region_eval" /
        ("responsibility_ablation_" +
         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    archive = (PROJECT_ROOT / "bin" / "old").resolve()
    if output.exists() or output == archive or output.is_relative_to(archive):
        raise ValueError("output must be a new directory outside bin/old")
    output.mkdir(parents=True)
    _write_json(output / "analysis_manifest.json", dict(
        status="started",
        analysis_version=ANALYSIS_VERSION,
        source_run=str(source_run),
        source_filtered_sha256=_file_hash(
            source_run / "filtered_predictions.json"),
        parameters=dict(
            thresholds=thresholds,
            primary_theta=primary_theta,
            tau=tau,
            methods=METHODS,
        ),
        source_code_sha256={
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"):
            _file_hash(path)
            for path in (
                PROJECT_ROOT / "region_eval" / "responsibility_analysis.py",
                PROJECT_ROOT / "run_region_responsibility.py",
            )
        },
    ))

    threshold_rows = []
    primary_by_method = {}
    primary_original_images = None
    for theta in thresholds:
        _, original_images, original_summary = evaluate_document(
            document, theta=theta, tau=tau)
        for method in METHODS:
            gt_rows, pair_rows = score_document(document, theta, method, tau)
            summary = aggregate_responsibility(
                gt_rows, pair_rows, original_summary, original_images)
            threshold_rows.append(summary)
            if theta == primary_theta:
                primary_by_method[method] = gt_rows
                primary_original_images = original_images
    threshold_rows.sort(key=lambda row: (row["method"], row["theta"]))

    class_names = document["metadata"]["class_names"]
    local_rows = local_gt_diagnostics(
        primary_original_images, document, class_names, primary_theta)
    groups = _candidate_groups(local_rows)
    candidate_rows = _candidate_summary(primary_by_method, groups)
    size_rows = [
        row for method in METHODS
        for row in _size_summary(primary_by_method[method], method)
    ]
    correction_rows = _correction_cases(primary_by_method, class_names)
    responsibility_rows = [
        dict(row, class_name=class_names[row["class_id"]])
        for method in METHODS for row in primary_by_method[method]
    ]

    threshold_fields = list(threshold_rows[0])
    gt_fields = [
        "image_id", "gt_id", "class_id", "class_name", "theta", "method",
        "gt_area", "original_AC", "original_BestIoGT",
        "original_CollectiveGain", "responsibility_coverage",
        "responsibility_precision", "responsibility_dice",
        "best_responsibility_coverage", "responsibility_gain",
        "coverage_hit", "assigned_pred_count", "assigned_weight_sum",
        "assigned_multigt_source_count", "assigned_pred_ids",
        "assigned_weights", "coverage_removed",
    ]
    _write_csv(
        output / "method_threshold.csv", threshold_rows, threshold_fields)
    _write_csv(
        output / "responsibility_gt.csv", responsibility_rows, gt_fields)
    _write_csv(
        output / "size_method_summary.csv", size_rows, list(size_rows[0]))
    _write_csv(
        output / "candidate_preservation.csv",
        candidate_rows,
        list(candidate_rows[0]),
    )
    _write_csv(
        output / "correction_cases.csv",
        correction_rows,
        list(correction_rows[0]),
    )
    overview_plot(
        threshold_rows, size_rows, candidate_rows, output / "overview.png")
    primary_rows = [
        row for row in threshold_rows if row["theta"] == primary_theta
    ]
    (output / "README.md").write_text(
        _report_text(primary_rows, candidate_rows, source_run),
        encoding="utf-8",
        newline="\n",
    )
    counts = dict(
        source_images=len(document["images"]),
        source_gt=complete["summary"]["n_gt"],
        methods=len(METHODS),
        thresholds=len(thresholds),
        primary_gt_rows=len(responsibility_rows),
        original_positive_CG_candidates=len(
            groups["all_original_positive_CG"]),
        low_overlap_candidates=len(
            groups["single_gt_low_overlap_candidates"]),
    )
    _write_json(output / "analysis_complete.json", dict(
        status="complete",
        analysis_version=ANALYSIS_VERSION,
        source_run=str(source_run),
        counts=counts,
        primary_method_summary={
            row["method"]: row for row in primary_rows
        },
        candidate_preservation={
            f"{row['method']}:{row['group']}": row
            for row in candidate_rows
        },
        output_sha256={
            str(path.relative_to(output)).replace("\\", "/"):
            _file_hash(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name not in (
                "analysis_manifest.json", "analysis_complete.json")
        },
    ))
    print(json.dumps(dict(output=str(output), **counts),
                     ensure_ascii=False), flush=True)
    print(
        "SUCCESS: responsibility ablation complete. "
        "No inference or training performed.",
        flush=True,
    )
    return output
