"""One-to-one detection metrics and controlled metric-behavior experiments."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from .case_analysis import (
    _file_hash,
    _write_csv,
    _write_json,
    harmonic,
    validate_source,
)
from .diagnostics import Box, Config, aggregate, image_region_diagnostics
from .geometry import area, intersection
from .offline import PROJECT_ROOT, evaluate_document
from .responsibility_analysis import (
    aggregate_responsibility,
    rectangle_iou,
    score_document,
)


ANALYSIS_VERSION = "region-matching-analysis-0.1.0"
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
FIXED_IOU_THRESHOLDS = (0.50, 0.75)
RECALL_THRESHOLDS = tuple(index / 100.0 for index in range(101))


def harmonic3(first, second, third):
    if first <= 0 or second <= 0 or third <= 0:
        return 0.0
    return 3.0 / (1.0 / first + 1.0 / second + 1.0 / third)


def match_class(document, class_id, iou_threshold, confidence_threshold=None):
    """Greedy score-ordered one-to-one matching for one class."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    events = []
    n_gt = 0
    for image in document["images"]:
        gts = sorted(
            (row for row in image["gts"] if row["class_id"] == class_id),
            key=lambda row: row["id"],
        )
        predictions = sorted(
            (
                row for row in image["predictions"]
                if row["class_id"] == class_id
                and (confidence_threshold is None
                     or row["score"] >= confidence_threshold)
            ),
            key=lambda row: (-row["score"], row["id"]),
        )
        n_gt += len(gts)
        unmatched = {row["id"]: row for row in gts}
        for prediction in predictions:
            candidates = []
            for gt_id, gt in unmatched.items():
                iou = rectangle_iou(prediction["xyxy"], gt["xyxy"])
                if iou >= iou_threshold:
                    overlap = intersection(
                        tuple(prediction["xyxy"]), tuple(gt["xyxy"]))
                    candidates.append((iou, area(overlap), gt_id))
            if candidates:
                _, matched_area, gt_id = sorted(
                    candidates,
                    key=lambda row: (-row[0], -row[1], row[2]),
                )[0]
                gt = unmatched.pop(gt_id)
                iou = rectangle_iou(prediction["xyxy"], gt["xyxy"])
                events.append(dict(
                    image_id=image["image_id"],
                    class_id=class_id,
                    pred_id=prediction["id"],
                    score=prediction["score"],
                    is_tp=True,
                    matched_gt_id=gt_id,
                    matched_iou=iou,
                    matched_intersection_area=matched_area,
                ))
            else:
                events.append(dict(
                    image_id=image["image_id"],
                    class_id=class_id,
                    pred_id=prediction["id"],
                    score=prediction["score"],
                    is_tp=False,
                    matched_gt_id=None,
                    matched_iou=None,
                    matched_intersection_area=None,
                ))
    events.sort(key=lambda row: (
        -row["score"], row["image_id"], row["pred_id"]))
    return events, n_gt


def summarize_events(events, n_gt, iou_threshold):
    tp = sum(row["is_tp"] for row in events)
    fp = len(events) - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else None
    f1 = harmonic(precision, recall or 0.0)
    matched_ious = [
        row["matched_iou"] for row in events if row["is_tp"]
    ]
    normalized_localization_sum = sum(
        (1.0 - value) / (1.0 - iou_threshold)
        for value in matched_ious
    ) if iou_threshold < 1.0 else 0.0
    z = tp + fp + fn
    lrp = (
        (normalized_localization_sum + fp + fn) / z if z else None
    )
    return dict(
        n_gt=n_gt,
        n_pred=len(events),
        TP=tp,
        FP=fp,
        FN=fn,
        precision=precision,
        recall=recall,
        F1=f1,
        mean_matched_IoU=(
            statistics.fmean(matched_ious) if matched_ious else None),
        LRP=lrp,
        LRP_localization_error=(
            normalized_localization_sum / tp if tp else None),
        LRP_FP_error=fp / (tp + fp) if tp + fp else None,
        LRP_FN_error=fn / n_gt if n_gt else None,
    )


def _mean_defined(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return statistics.fmean(values) if values else None


def fixed_threshold_metrics(
        document,
        class_names,
        confidence_thresholds,
        iou_thresholds=FIXED_IOU_THRESHOLDS):
    summary_rows, class_rows = [], []
    for confidence in confidence_thresholds:
        for iou_threshold in iou_thresholds:
            per_class = []
            all_events = []
            total_gt = 0
            for class_id, class_name in enumerate(class_names):
                events, n_gt = match_class(
                    document, class_id, iou_threshold, confidence)
                metrics = summarize_events(events, n_gt, iou_threshold)
                row = dict(
                    confidence_threshold=confidence,
                    iou_threshold=iou_threshold,
                    class_id=class_id,
                    class_name=class_name,
                    **metrics,
                )
                class_rows.append(row)
                if n_gt:
                    per_class.append(row)
                all_events.extend(events)
                total_gt += n_gt
            micro = summarize_events(all_events, total_gt, iou_threshold)
            summary_rows.append(dict(
                confidence_threshold=confidence,
                iou_threshold=iou_threshold,
                **{f"micro_{key}": value for key, value in micro.items()},
                macro_precision=_mean_defined(per_class, "precision"),
                macro_recall=_mean_defined(per_class, "recall"),
                macro_F1=_mean_defined(per_class, "F1"),
                macro_mean_matched_IoU=_mean_defined(
                    per_class, "mean_matched_IoU"),
                macro_LRP=_mean_defined(per_class, "LRP"),
                macro_LRP_localization_error=_mean_defined(
                    per_class, "LRP_localization_error"),
                macro_LRP_FP_error=_mean_defined(
                    per_class, "LRP_FP_error"),
                macro_LRP_FN_error=_mean_defined(
                    per_class, "LRP_FN_error"),
            ))
    return summary_rows, class_rows


def average_precision(events, n_gt, recall_thresholds=RECALL_THRESHOLDS):
    """COCO-style 101-point interpolated AP for already matched events."""
    if n_gt <= 0:
        return None
    if not events:
        return 0.0
    tp, fp = 0, 0
    recalls, precisions = [], []
    for event in events:
        if event["is_tp"]:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    samples = []
    for recall_threshold in recall_thresholds:
        value = 0.0
        for recall, precision in zip(recalls, precisions):
            if recall >= recall_threshold:
                value = precision
                break
        samples.append(value)
    return statistics.fmean(samples)


def ap_metrics(document, class_names):
    class_rows = []
    for class_id, class_name in enumerate(class_names):
        values = {}
        recalls = {}
        n_gt = 0
        n_pred = 0
        for iou_threshold in IOU_THRESHOLDS:
            events, current_gt = match_class(
                document, class_id, iou_threshold, None)
            n_gt = current_gt
            n_pred = len(events)
            values[iou_threshold] = average_precision(events, n_gt)
            recalls[iou_threshold] = (
                sum(row["is_tp"] for row in events) / n_gt if n_gt else None)
        class_rows.append(dict(
            class_id=class_id,
            class_name=class_name,
            n_gt=n_gt,
            n_pred=n_pred,
            AP50=values[0.50],
            AP75=values[0.75],
            AP50_95=(statistics.fmean(
                value for value in values.values() if value is not None)
                if any(value is not None for value in values.values()) else None),
            AR50_95=(statistics.fmean(
                value for value in recalls.values() if value is not None)
                if any(value is not None for value in recalls.values()) else None),
        ))
    valid = [row for row in class_rows if row["n_gt"]]
    summary = dict(
        aggregation="class_macro",
        iou_thresholds=IOU_THRESHOLDS,
        recall_thresholds=RECALL_THRESHOLDS,
        max_detections_note=(
            "uses the post-NMS source cache, which was capped at 500 "
            "detections per image across classes"),
        AP50=_mean_defined(valid, "AP50"),
        AP75=_mean_defined(valid, "AP75"),
        AP50_95=_mean_defined(valid, "AP50_95"),
        AR50_95=_mean_defined(valid, "AR50_95"),
    )
    return summary, class_rows


def _metrics_from_prefix(tp, fp, n_gt, localization_sum, iou_threshold):
    fn = n_gt - tp
    z = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        precision=precision,
        recall=recall,
        F1=harmonic(precision, recall),
        LRP=(localization_sum + fp + fn) / z if z else None,
        LRP_localization_error=localization_sum / tp if tp else None,
        LRP_FP_error=fp / (tp + fp) if tp + fp else None,
        LRP_FN_error=fn / n_gt if n_gt else None,
    )


def optimal_lrp(events, n_gt, iou_threshold):
    """Find minimum LRP and maximum F1 at score-group boundaries."""
    if n_gt <= 0:
        return None
    initial = _metrics_from_prefix(0, 0, n_gt, 0.0, iou_threshold)
    best_lrp = dict(
        confidence_threshold=1.0,
        n_pred=0,
        **initial,
    )
    best_f1 = dict(best_lrp)
    tp, fp, localization_sum = 0, 0, 0.0
    index = 0
    while index < len(events):
        score = events[index]["score"]
        while index < len(events) and events[index]["score"] == score:
            event = events[index]
            if event["is_tp"]:
                tp += 1
                localization_sum += (
                    (1.0 - event["matched_iou"])
                    / (1.0 - iou_threshold)
                )
            else:
                fp += 1
            index += 1
        metrics = _metrics_from_prefix(
            tp, fp, n_gt, localization_sum, iou_threshold)
        candidate = dict(
            confidence_threshold=score,
            n_pred=tp + fp,
            **metrics,
        )
        if candidate["LRP"] < best_lrp["LRP"] - 1e-15:
            best_lrp = candidate
        if candidate["F1"] > best_f1["F1"] + 1e-15:
            best_f1 = candidate
    return dict(
        optimal_LRP_threshold=best_lrp["confidence_threshold"],
        oLRP=best_lrp["LRP"],
        oLRP_localization_error=best_lrp["LRP_localization_error"],
        oLRP_FP_error=best_lrp["LRP_FP_error"],
        oLRP_FN_error=best_lrp["LRP_FN_error"],
        oLRP_TP=best_lrp["TP"],
        oLRP_FP=best_lrp["FP"],
        oLRP_FN=best_lrp["FN"],
        optimal_F1_threshold=best_f1["confidence_threshold"],
        max_F1=best_f1["F1"],
        max_F1_precision=best_f1["precision"],
        max_F1_recall=best_f1["recall"],
    )


def optimal_metrics(document, class_names):
    rows = []
    for iou_threshold in FIXED_IOU_THRESHOLDS:
        for class_id, class_name in enumerate(class_names):
            events, n_gt = match_class(
                document, class_id, iou_threshold, None)
            result = optimal_lrp(events, n_gt, iou_threshold)
            rows.append(dict(
                iou_threshold=iou_threshold,
                class_id=class_id,
                class_name=class_name,
                n_gt=n_gt,
                n_cached_pred=len(events),
                **result,
            ))
    summary = []
    for iou_threshold in FIXED_IOU_THRESHOLDS:
        values = [
            row for row in rows
            if row["iou_threshold"] == iou_threshold and row["n_gt"]
        ]
        all_events = []
        total_gt = 0
        for class_id in range(len(class_names)):
            events, n_gt = match_class(
                document, class_id, iou_threshold, None)
            all_events.extend(events)
            total_gt += n_gt
        all_events.sort(key=lambda row: (
            -row["score"], row["class_id"], row["image_id"], row["pred_id"]))
        micro = optimal_lrp(all_events, total_gt, iou_threshold)
        summary.append(dict(
            iou_threshold=iou_threshold,
            aggregation="class_macro_at_class_specific_optima",
            oLRP=_mean_defined(values, "oLRP"),
            oLRP_localization_error=_mean_defined(
                values, "oLRP_localization_error"),
            oLRP_FP_error=_mean_defined(values, "oLRP_FP_error"),
            oLRP_FN_error=_mean_defined(values, "oLRP_FN_error"),
            mean_optimal_LRP_threshold=_mean_defined(
                values, "optimal_LRP_threshold"),
            mean_max_F1=_mean_defined(values, "max_F1"),
            mean_optimal_F1_threshold=_mean_defined(
                values, "optimal_F1_threshold"),
            micro_oLRP=micro["oLRP"],
            micro_optimal_LRP_threshold=micro[
                "optimal_LRP_threshold"],
            micro_max_F1=micro["max_F1"],
            micro_optimal_F1_threshold=micro[
                "optimal_F1_threshold"],
        ))
    return summary, rows


def _counterfactual_documents():
    gts = [
        dict(id="gt_a", class_id=0, xyxy=[5, 5, 15, 15]),
        dict(id="gt_b", class_id=0, xyxy=[20, 5, 30, 15]),
    ]

    def pred(identifier, score, xyxy):
        return dict(id=identifier, class_id=0, score=score, xyxy=xyxy)

    base = [
        pred("pred_a", .95, [5, 5, 15, 15]),
        pred("pred_b", .80, [20, 5, 30, 15]),
    ]
    scenarios = {
        "perfect_base": (
            base,
            "two exact predictions for two GT"),
        "exact_duplicate": (
            base + [pred("pred_a_duplicate", .85, [5, 5, 15, 15])],
            "duplicate must not improve region coverage and becomes one FP"),
        "small_orphan": (
            base + [pred("small_orphan", .85, [35, 0, 36, 1])],
            "one tiny remote FP has little area but full count penalty"),
        "oversized_boxes": (
            [
                pred("large_a", .95, [0, 0, 20, 20]),
                pred("large_b", .80, [15, 0, 35, 20]),
            ],
            "coverage remains high while local precision and IoU fall"),
        "cross_gt_box": (
            [pred("cross", .95, [5, 5, 30, 15])],
            "one prediction covers both GT regions, owns one, and fails IoU=0.50 matching"),
        "complementary_fragments": (
            [
                pred("a_left", .95, [5, 5, 10, 15]),
                pred("a_right", .85, [10, 5, 15, 15]),
                pred("b_left", .90, [20, 5, 25, 15]),
                pred("b_right", .80, [25, 5, 30, 15]),
            ],
            "two non-overlapping half boxes jointly cover each GT"),
    }
    documents = {}
    for name, (predictions, note) in scenarios.items():
        documents[name] = (
            dict(
                metadata=dict(class_ids=[0]),
                images=[dict(
                    image_id="synthetic",
                    width=40,
                    height=20,
                    gts=[dict(row) for row in gts],
                    predictions=[dict(row) for row in predictions],
                )],
            ),
            note,
        )
    return documents


def counterfactual_metrics():
    rows = []
    for scenario, (document, note) in _counterfactual_documents().items():
        responsibility_rows, _ = score_document(
            document, 0.25, "hard_iou", 0.5)
        diagnostic = image_region_diagnostics(
            [
                Box(row["id"], tuple(row["xyxy"]), row["class_id"])
                for row in document["images"][0]["gts"]
            ],
            [
                Box(
                    row["id"],
                    tuple(row["xyxy"]),
                    row["class_id"],
                    row["score"],
                )
                for row in document["images"][0]["predictions"]
            ],
            (40, 20),
            image_id="synthetic",
            config=Config(theta=.25, tau=.5, class_ids=(0,)),
        )
        region_summary = aggregate([diagnostic])
        events, n_gt = match_class(document, 0, .5, .25)
        matching = summarize_events(events, n_gt, .5)
        ap50 = average_precision(
            match_class(document, 0, .5, None)[0], n_gt)
        rows.append(dict(
            scenario=scenario,
            intended_change=note,
            n_gt=n_gt,
            n_pred=len(events),
            ResponsibilityCoverage=statistics.fmean(
                row["responsibility_coverage"]
                for row in responsibility_rows),
            ResponsibilityPrecision=statistics.fmean(
                row["responsibility_precision"]
                for row in responsibility_rows),
            ResponsibilityDice=statistics.fmean(
                row["responsibility_dice"] for row in responsibility_rows),
            ResponsibilityGain=statistics.fmean(
                row["responsibility_gain"] for row in responsibility_rows),
            ClassUnionDice=region_summary["ClassUnionDice"],
            ClassUnionIoU=region_summary["ClassUnionIoU"],
            TP=matching["TP"],
            FP=matching["FP"],
            FN=matching["FN"],
            Precision=matching["precision"],
            Recall=matching["recall"],
            F1=matching["F1"],
            LRP=matching["LRP"],
            one_minus_LRP=1.0 - matching["LRP"],
            AP50=ap50,
        ))
    base = next(row for row in rows if row["scenario"] == "perfect_base")
    for row in rows:
        for field in (
                "ResponsibilityDice", "ClassUnionDice", "F1",
                "one_minus_LRP", "AP50"):
            row[f"delta_{field}_vs_base"] = row[field] - base[field]
    return rows


def three_layer_metrics(
        document,
        confidence_thresholds,
        matching_summary):
    matching_by_theta = {
        row["confidence_threshold"]: row
        for row in matching_summary if row["iou_threshold"] == 0.50
    }
    rows = []
    for confidence in confidence_thresholds:
        _, original_images, original_summary = evaluate_document(
            document, theta=confidence, tau=.5)
        gt_rows, pair_rows = score_document(
            document, confidence, "hard_iou", .5)
        responsibility = aggregate_responsibility(
            gt_rows, pair_rows, original_summary, original_images)
        detection = matching_by_theta[confidence]
        instance = responsibility["mean_responsibility_dice"]
        region = original_summary["ClassUnionDice"]
        f1 = detection["micro_F1"]
        lrp_quality = 1.0 - detection["micro_LRP"]
        rows.append(dict(
            confidence_threshold=confidence,
            instance_ResponsibilityDice=instance,
            region_ClassUnionDice=region,
            detection_F1=f1,
            detection_one_minus_LRP=lrp_quality,
            H3_instance_region_F1=harmonic3(instance, region, f1),
            H3_instance_region_one_minus_LRP=harmonic3(
                instance, region, lrp_quality),
            TP=detection["micro_TP"],
            FP=detection["micro_FP"],
            FN=detection["micro_FN"],
            orphan_pred_count=original_summary["orphan_pred_count"],
        ))
    return rows


def overview_plot(
        matching_rows,
        ap_rows,
        counterfactual_rows,
        three_layer_rows,
        output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fixed = [
        row for row in matching_rows if row["iou_threshold"] == .5
    ]
    theta = [row["confidence_threshold"] for row in fixed]
    for field, label in (
            ("micro_precision", "Precision"),
            ("micro_recall", "Recall"),
            ("micro_F1", "F1"),
            ("micro_mean_matched_IoU", "Mean matched IoU")):
        axes[0, 0].plot(
            theta, [row[field] for row in fixed], marker="o", label=label)
    axes[0, 0].set(
        title="One-to-one matching at IoU=0.50",
        xlabel="Confidence threshold",
        ylabel="Score",
    )
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(alpha=.25)
    axes[0, 0].legend(frameon=False)

    for field, label in (
            ("instance_ResponsibilityDice", "Instance"),
            ("region_ClassUnionDice", "Region"),
            ("detection_F1", "Detection F1"),
            ("H3_instance_region_F1", "Exploratory H3")):
        axes[0, 1].plot(
            [row["confidence_threshold"] for row in three_layer_rows],
            [row[field] for row in three_layer_rows],
            marker="o",
            label=label,
        )
    axes[0, 1].set(
        title="Three-layer threshold comparison",
        xlabel="Confidence threshold",
        ylabel="Score",
    )
    axes[0, 1].set_ylim(0, .65)
    axes[0, 1].grid(alpha=.25)
    axes[0, 1].legend(frameon=False)

    y = np.arange(len(ap_rows))
    axes[1, 0].barh(
        y + .18, [row["AP50"] for row in ap_rows], .34, label="AP50")
    axes[1, 0].barh(
        y - .18, [row["AP75"] for row in ap_rows], .34, label="AP75")
    axes[1, 0].set(
        title="COCO-style diagnostic AP by class",
        xlabel="AP",
        yticks=y,
        yticklabels=[row["class_name"] for row in ap_rows],
    )
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].legend(frameon=False)

    fields = (
        ("ResponsibilityDice", "Instance"),
        ("ClassUnionDice", "Region"),
        ("F1", "F1"),
        ("one_minus_LRP", "1-LRP"),
    )
    x = np.arange(len(counterfactual_rows))
    width = .19
    for index, (field, label) in enumerate(fields):
        axes[1, 1].bar(
            x + (index - 1.5) * width,
            [row[field] for row in counterfactual_rows],
            width,
            label=label,
        )
    axes[1, 1].set(
        title="Controlled behavior checks",
        ylabel="Quality score",
        xticks=x,
        xticklabels=[
            row["scenario"].replace("_", "\n")
            for row in counterfactual_rows
        ],
    )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(frameon=False)
    figure.suptitle(
        "VisDrone matching metrics and controlled counterfactuals",
        fontsize=16,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _report_text(
        source_run,
        ap_summary,
        optimal_summary,
        matching_rows,
        counterfactual_rows,
        three_layer_rows):
    fixed50 = [
        row for row in matching_rows if row["iou_threshold"] == .5
    ]
    best_f1 = max(
        fixed50, key=lambda row: (
            row["micro_F1"], row["confidence_threshold"]))
    best_lrp = min(
        fixed50, key=lambda row: (
            row["micro_LRP"], -row["confidence_threshold"]))
    olrp50 = next(
        row for row in optimal_summary if row["iou_threshold"] == .5)
    best_h3 = max(
        three_layer_rows,
        key=lambda row: (
            row["H3_instance_region_F1"],
            row["confidence_threshold"],
        ),
    )
    counter_lines = [
        (
            f"| {row['scenario']} | {row['ResponsibilityDice']:.3f} | "
            f"{row['ClassUnionDice']:.3f} | {row['F1']:.3f} | "
            f"{row['one_minus_LRP']:.3f} | {row['AP50']:.3f} |"
        )
        for row in counterfactual_rows
    ]
    return f"""# 第五轮：匹配指标与受控反例

来源运行：{source_run}。本轮只读取已有 50 图缓存，没有推理或训练。

## 真实缓存结果

COCO 风格 101 点插值、类别宏平均诊断：

| AP50 | AP75 | AP50:95 | AR50:95 |
|---:|---:|---:|---:|
| {ap_summary['AP50']:.6f} | {ap_summary['AP75']:.6f} | {ap_summary['AP50_95']:.6f} | {ap_summary['AR50_95']:.6f} |

固定 IoU=0.50 时，五个置信度点中 micro F1 最高的是 theta={best_f1['confidence_threshold']:.2f}，F1={best_f1['micro_F1']:.6f}；micro LRP 最低的是 theta={best_lrp['confidence_threshold']:.2f}，LRP={best_lrp['micro_LRP']:.6f}。

搜索全部缓存分数后，全数据集 micro F1 的最优阈值为 {olrp50['micro_optimal_F1_threshold']:.6f}，F1={olrp50['micro_max_F1']:.6f}；micro oLRP50 的最优阈值为 {olrp50['micro_optimal_LRP_threshold']:.6f}，oLRP={olrp50['micro_oLRP']:.6f}。按类别分别优化后的 oLRP50 类别宏平均为 {olrp50['oLRP']:.6f}，类别最优阈值均值为 {olrp50['mean_optimal_LRP_threshold']:.6f}。oLRP 是误差，越低越好。

三层探索 H3(ResponsibilityDice, ClassUnionDice, F1) 在五个固定阈值中选择 theta={best_h3['confidence_threshold']:.2f}，得分 {best_h3['H3_instance_region_F1']:.6f}。这仍是设计诊断，不是最终 VC。

## 受控反例

| 场景 | 实例 Dice | 区域 Dice | F1 | 1-LRP | AP50 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(counter_lines)}

exact_duplicate 和 small_orphan 对 F1 的计数影响相同，但 small_orphan 对 ClassUnionDice 的影响很小。complementary_fragments 保持完整实例与区域覆盖，却会因为一对一匹配产生额外 FP；它的 AP50 仍为 1，因为额外框排在已经达到完整 recall 之后。由此可见，F1/LRP 能补足小误检惩罚，也会惩罚论文希望识别的有效多框互补；AP 又可能忽略达到完整 recall 后的低排名额外框。

## 协议

- 同类预测按 score 降序，在每张图中贪心匹配尚未使用且 IoU 最大的 GT。
- AP 使用 IoU 0.50:0.05:0.95 和 101 个 recall 点的精度包络。
- LRP 总误差由归一化定位误差、FP 和 FN 在 TP+FP+FN 上共同归一化。
- oLRP 在每个类别的全部缓存置信度边界上寻找最小 LRP。
- VisDrone ignore 已在来源缓存上游过滤。本报告不会被称为官方 VisDrone AP。
- 17 张图的来源预测达到 max_det=500；缺失部分位于低分尾部，因此完整 AP/oLRP 仍记录此缓存上限限制。

## 对 VC 的判断

第五轮支持三层分开报告：

1. hard_iou ResponsibilityDice 表达同一实例的责任覆盖与局部空间质量；
2. ClassUnionDice 表达区域背景面积；
3. F1/LRP 表达一对一检测的 FP、FN 和定位错误。

现在不应直接把 F1 塞入 VC。互补碎片的反例表明，F1 会把第二个有效碎片视为 FP。下一步应定义边际贡献规则：对同一 GT 的额外预测，只有带来足够独有覆盖且没有过量外溢时，才从检测层的 FP 中转为有效 fragment。之后再重新计算 fragment-aware F1/LRP，并与标准 F1/LRP 并列报告。
"""


def analyze_matching(
        source_run,
        *,
        confidence_thresholds=(0.05, 0.10, 0.25, 0.50, 0.75),
        output=None):
    source_run, document, _, _, complete, _ = validate_source(source_run)
    class_names = document["metadata"]["class_names"]
    confidence_thresholds = tuple(
        float(value) for value in confidence_thresholds)
    if any(not 0.0 <= value <= 1.0 for value in confidence_thresholds):
        raise ValueError("confidence thresholds must be in [0, 1]")
    if len(set(confidence_thresholds)) != len(confidence_thresholds):
        raise ValueError("confidence thresholds must be unique")
    output = (
        Path(output).resolve() if output else
        PROJECT_ROOT / "outputs" / "region_eval" /
        ("matching_analysis_" +
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
            confidence_thresholds=confidence_thresholds,
            fixed_iou_thresholds=FIXED_IOU_THRESHOLDS,
            ap_iou_thresholds=IOU_THRESHOLDS,
            ap_recall_thresholds=RECALL_THRESHOLDS,
        ),
        source_code_sha256={
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"):
            _file_hash(path)
            for path in (
                PROJECT_ROOT / "region_eval" / "matching_analysis.py",
                PROJECT_ROOT / "run_region_matching.py",
            )
        },
    ))

    matching_rows, class_threshold_rows = fixed_threshold_metrics(
        document, class_names, confidence_thresholds)
    ap_summary, ap_rows = ap_metrics(document, class_names)
    optimal_summary, optimal_rows = optimal_metrics(document, class_names)
    counterfactual_rows = counterfactual_metrics()
    three_layer_rows = three_layer_metrics(
        document, confidence_thresholds, matching_rows)

    _write_csv(
        output / "matching_threshold.csv",
        matching_rows,
        list(matching_rows[0]),
    )
    _write_csv(
        output / "matching_by_class_threshold.csv",
        class_threshold_rows,
        list(class_threshold_rows[0]),
    )
    _write_json(output / "ap_summary.json", ap_summary)
    _write_csv(
        output / "ap_by_class.csv", ap_rows, list(ap_rows[0]))
    _write_json(output / "olrp_summary.json", optimal_summary)
    _write_csv(
        output / "olrp_by_class.csv",
        optimal_rows,
        list(optimal_rows[0]),
    )
    _write_csv(
        output / "counterfactual_matrix.csv",
        counterfactual_rows,
        list(counterfactual_rows[0]),
    )
    _write_csv(
        output / "three_layer_threshold.csv",
        three_layer_rows,
        list(three_layer_rows[0]),
    )
    overview_plot(
        matching_rows,
        ap_rows,
        counterfactual_rows,
        three_layer_rows,
        output / "overview.png",
    )
    (output / "README.md").write_text(
        _report_text(
            source_run,
            ap_summary,
            optimal_summary,
            matching_rows,
            counterfactual_rows,
            three_layer_rows,
        ),
        encoding="utf-8",
        newline="\n",
    )
    counts = dict(
        source_images=len(document["images"]),
        source_gt=complete["summary"]["n_gt"],
        source_cached_predictions=sum(
            len(image["predictions"]) for image in document["images"]),
        classes=len(class_names),
        confidence_thresholds=len(confidence_thresholds),
        matching_threshold_rows=len(matching_rows),
        class_threshold_rows=len(class_threshold_rows),
        counterfactual_scenarios=len(counterfactual_rows),
    )
    _write_json(output / "analysis_complete.json", dict(
        status="complete",
        analysis_version=ANALYSIS_VERSION,
        source_run=str(source_run),
        counts=counts,
        ap_summary=ap_summary,
        optimal_summary=optimal_summary,
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
        "SUCCESS: matching analysis complete. "
        "No inference or training performed.",
        flush=True,
    )
    return output
