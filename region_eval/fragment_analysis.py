"""Fragment-aware matching after exclusive hard-IoU prediction ownership."""

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
    local_gt_diagnostics,
    validate_source,
)
from .diagnostics import Box, Config, aggregate, image_region_diagnostics
from .geometry import area, intersection, union_area
from .matching_analysis import (
    _counterfactual_documents,
    average_precision,
    match_class,
    summarize_events,
)
from .offline import PROJECT_ROOT, evaluate_document
from .responsibility_analysis import (
    assignment_weights,
    rectangle_iou,
    score_document,
)


ANALYSIS_VERSION = "fragment-aware-matching-0.1.0"
UNIQUE_COVERAGE_THRESHOLDS = (0.01, 0.02, 0.05, 0.10)
FRAGMENT_PRECISION_THRESHOLDS = (0.25, 0.50, 0.75)
REFERENCE_UNIQUE_COVERAGE = 0.02
REFERENCE_FRAGMENT_PRECISION = 0.50


def _group_geometry(gt, predictions):
    gt_rect = tuple(gt["xyxy"])
    gt_area = area(gt_rect)
    pred_rects = [tuple(row["xyxy"]) for row in predictions]
    intersections = [
        overlap for row in predictions
        if (overlap := intersection(gt_rect, tuple(row["xyxy"]))) is not None
    ]
    pred_area = union_area(pred_rects)
    intersection_area = min(gt_area, pred_area, union_area(intersections))
    coverage = intersection_area / gt_area
    precision = intersection_area / pred_area if pred_area else 0.0
    denominator = gt_area + pred_area - intersection_area
    iou = intersection_area / denominator if denominator else 0.0
    return dict(
        group_pred_union_area=pred_area,
        group_intersection_area=intersection_area,
        group_coverage=coverage,
        group_precision=precision,
        group_dice=harmonic(coverage, precision),
        group_iou=iou,
    )


def _fragment_lrp(group_ious, tp, fp, fn, matching_iou):
    localization_sum = sum(
        (1.0 - value) / (1.0 - matching_iou)
        for value in group_ious
    )
    z = tp + fp + fn
    return dict(
        fragment_LRP=(
            (localization_sum + fp + fn) / z if z else None),
        fragment_LRP_localization_error=(
            localization_sum / tp if tp else None),
        fragment_LRP_FP_error=fp / (tp + fp) if tp + fp else None,
        fragment_LRP_FN_error=fn / (tp + fn) if tp + fn else None,
    )


def evaluate_fragment_aware(
        document,
        confidence_threshold,
        *,
        matching_iou=0.50,
        unique_coverage_threshold=REFERENCE_UNIQUE_COVERAGE,
        fragment_precision_threshold=REFERENCE_FRAGMENT_PRECISION,
        group_iou_threshold=None):
    """Evaluate physical predictions plus virtual one-TP fragment groups."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    for name, value in (
            ("matching_iou", matching_iou),
            ("unique_coverage_threshold", unique_coverage_threshold),
            ("fragment_precision_threshold", fragment_precision_threshold)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    group_iou_threshold = (
        matching_iou if group_iou_threshold is None
        else group_iou_threshold
    )
    if not 0.0 <= group_iou_threshold <= 1.0:
        raise ValueError("group_iou_threshold must be in [0, 1]")

    gt_rows, decisions = [], []
    tp = fp = fn = single_tp = group_tp = support_count = 0
    orphan_fp = rejected_fp = failed_group_fp = 0
    detected_group_ious = []
    n_pred = 0

    for image in document["images"]:
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)
        for gt in image["gts"]:
            gt_by_class[gt["class_id"]].append(gt)
        for prediction in image["predictions"]:
            if prediction["score"] >= confidence_threshold:
                pred_by_class[prediction["class_id"]].append(prediction)
                n_pred += 1

        for class_id in sorted(set(gt_by_class) | set(pred_by_class)):
            gts = sorted(gt_by_class[class_id], key=lambda row: row["id"])
            predictions = sorted(
                pred_by_class[class_id],
                key=lambda row: (-row["score"], row["id"]),
            )
            weights = assignment_weights(gts, predictions, "hard_iou")
            hits = {
                row["id"]: sum(
                    intersection(
                        tuple(row["xyxy"]), tuple(gt["xyxy"])) is not None
                    for gt in gts
                )
                for row in predictions
            }
            owner = {
                prediction_id: (
                    next(iter(values)) if values else None)
                for prediction_id, values in weights.items()
            }
            pred_by_id = {row["id"]: row for row in predictions}
            for prediction in predictions:
                if owner[prediction["id"]] is None:
                    fp += 1
                    orphan_fp += 1
                    decisions.append(dict(
                        image_id=image["image_id"],
                        class_id=class_id,
                        pred_id=prediction["id"],
                        score=prediction["score"],
                        owner_gt_id=None,
                        source_gt_hit_count=hits[prediction["id"]],
                        owner_iou=None,
                        fragment_precision=0.0,
                        marginal_unique_area=0.0,
                        marginal_unique_coverage=0.0,
                        provisional_accept=False,
                        final_status="orphan_fp",
                    ))

            for gt in gts:
                gt_rect = tuple(gt["xyxy"])
                gt_area = area(gt_rect)
                assigned = [
                    pred_by_id[prediction_id]
                    for prediction_id, gt_id in owner.items()
                    if gt_id == gt["id"]
                ]
                assigned.sort(key=lambda row: (-row["score"], row["id"]))
                anchor_candidates = [
                    row for row in assigned
                    if rectangle_iou(row["xyxy"], gt_rect) >= matching_iou
                ]
                anchor = (
                    sorted(
                        anchor_candidates,
                        key=lambda row: (
                            -row["score"],
                            -rectangle_iou(row["xyxy"], gt_rect),
                            row["id"],
                        ),
                    )[0]
                    if anchor_candidates else None
                )
                accepted = [anchor] if anchor is not None else []
                candidate_decisions = []
                current_intersections = []
                if anchor is not None:
                    current_intersections.append(
                        intersection(gt_rect, tuple(anchor["xyxy"])))
                    anchor_overlap = current_intersections[0]
                    decisions.append(dict(
                        image_id=image["image_id"],
                        class_id=class_id,
                        pred_id=anchor["id"],
                        score=anchor["score"],
                        owner_gt_id=gt["id"],
                        source_gt_hit_count=hits[anchor["id"]],
                        owner_iou=rectangle_iou(anchor["xyxy"], gt_rect),
                        fragment_precision=(
                            area(anchor_overlap)
                            / area(tuple(anchor["xyxy"]))),
                        marginal_unique_area=area(anchor_overlap),
                        marginal_unique_coverage=(
                            area(anchor_overlap) / gt_area),
                        provisional_accept=True,
                        final_status="anchor_tp",
                    ))

                for prediction in [
                        row for row in assigned
                        if anchor is None or row["id"] != anchor["id"]]:
                    overlap = intersection(
                        gt_rect, tuple(prediction["xyxy"]))
                    overlap_area = area(overlap)
                    before = union_area(current_intersections)
                    after = union_area(current_intersections + [overlap])
                    marginal_area = max(0.0, after - before)
                    marginal_coverage = marginal_area / gt_area
                    precision = overlap_area / area(tuple(prediction["xyxy"]))
                    accepted_now = (
                        marginal_coverage + 1e-15
                        >= unique_coverage_threshold
                        and precision + 1e-15
                        >= fragment_precision_threshold
                    )
                    if accepted_now:
                        accepted.append(prediction)
                        current_intersections.append(overlap)
                    candidate_decisions.append(dict(
                        image_id=image["image_id"],
                        class_id=class_id,
                        pred_id=prediction["id"],
                        score=prediction["score"],
                        owner_gt_id=gt["id"],
                        source_gt_hit_count=hits[prediction["id"]],
                        owner_iou=rectangle_iou(
                            prediction["xyxy"], gt_rect),
                        fragment_precision=precision,
                        marginal_unique_area=marginal_area,
                        marginal_unique_coverage=marginal_coverage,
                        provisional_accept=accepted_now,
                        final_status=(
                            "provisional_support"
                            if accepted_now else "rejected_fp"),
                    ))

                geometry = _group_geometry(gt, accepted)
                if anchor is not None:
                    detection_type = "single_anchor"
                    detected = True
                    single_tp += 1
                    tp += 1
                    accepted_support = len(accepted) - 1
                    support_count += accepted_support
                    rejected = len(assigned) - len(accepted)
                    fp += rejected
                    rejected_fp += rejected
                    for row in candidate_decisions:
                        row["final_status"] = (
                            "support"
                            if row["provisional_accept"]
                            else "rejected_fp")
                    detected_group_ious.append(geometry["group_iou"])
                elif (
                        len(accepted) >= 2
                        and geometry["group_iou"] >= group_iou_threshold):
                    detection_type = "fragment_group"
                    detected = True
                    group_tp += 1
                    tp += 1
                    support_count += len(accepted)
                    rejected = len(assigned) - len(accepted)
                    fp += rejected
                    rejected_fp += rejected
                    for row in candidate_decisions:
                        row["final_status"] = (
                            "group_support"
                            if row["provisional_accept"]
                            else "rejected_fp")
                    detected_group_ious.append(geometry["group_iou"])
                else:
                    detection_type = "miss"
                    detected = False
                    fn += 1
                    rejected = len(assigned)
                    fp += rejected
                    failed_group_fp += rejected
                    for row in candidate_decisions:
                        row["final_status"] = "failed_group_fp"
                decisions.extend(candidate_decisions)
                gt_rows.append(dict(
                    image_id=image["image_id"],
                    gt_id=gt["id"],
                    class_id=class_id,
                    confidence_threshold=confidence_threshold,
                    matching_iou=matching_iou,
                    group_iou_threshold=group_iou_threshold,
                    unique_coverage_threshold=unique_coverage_threshold,
                    fragment_precision_threshold=(
                        fragment_precision_threshold),
                    detection_type=detection_type,
                    detected=detected,
                    anchor_pred_id=anchor["id"] if anchor else None,
                    assigned_pred_count=len(assigned),
                    accepted_pred_count=len(accepted),
                    support_pred_count=(
                        len(accepted) - 1 if anchor else
                        len(accepted) if detected else 0),
                    rejected_pred_count=rejected,
                    **geometry,
                ))

    expected_fn = sum(not row["detected"] for row in gt_rows)
    if fn != expected_fn:
        raise AssertionError(f"GT accounting mismatch: fn={fn}, expected={expected_fn}")
    if n_pred != single_tp + support_count + fp:
        raise AssertionError(
            "physical prediction accounting must equal "
            "anchors + neutral supports + FP")
    n_gt = len(gt_rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else None
    lrp = _fragment_lrp(
        detected_group_ious, tp, fp, fn, matching_iou)
    summary = dict(
        confidence_threshold=confidence_threshold,
        matching_iou=matching_iou,
        group_iou_threshold=group_iou_threshold,
        unique_coverage_threshold=unique_coverage_threshold,
        fragment_precision_threshold=fragment_precision_threshold,
        n_gt=n_gt,
        n_pred=n_pred,
        TP=tp,
        single_anchor_TP=single_tp,
        fragment_group_TP=group_tp,
        neutral_support_pred_count=support_count,
        FP=fp,
        orphan_FP=orphan_fp,
        rejected_fragment_FP=rejected_fp,
        failed_group_FP=failed_group_fp,
        FN=fn,
        precision=precision,
        recall=recall,
        F1=harmonic(precision, recall or 0.0),
        mean_detected_group_IoU=(
            statistics.fmean(detected_group_ious)
            if detected_group_ious else None),
        **lrp,
    )
    return summary, gt_rows, sorted(
        decisions,
        key=lambda row: (
            row["image_id"], row["class_id"], row["pred_id"]),
    )


def _standard_micro(document, confidence_threshold, matching_iou=.5):
    events = []
    n_gt = 0
    for class_id in document["metadata"]["class_ids"]:
        class_events, class_gt = match_class(
            document, class_id, matching_iou, confidence_threshold)
        events.extend(class_events)
        n_gt += class_gt
    return summarize_events(events, n_gt, matching_iou)


def _candidate_sets(local_rows):
    positive_multigt = {
        (row["image_id"], row["gt_id"])
        for row in local_rows
        if row["CollectiveGain_i"] > 1e-9
        and row["contributing_multigt_count"] > 0
    }
    low_overlap = {
        (row["image_id"], row["gt_id"])
        for row in local_rows
        if row["CollectiveGain_i"] > 1e-9
        and row["contributing_pred_count"] >= 2
        and row["contributing_multigt_count"] == 0
        and row["max_contributing_pairwise_iou"] is not None
        and row["max_contributing_pairwise_iou"] < 0.3
    }
    return dict(
        original_positive_CG_with_multigt=positive_multigt,
        single_gt_low_overlap_candidates=low_overlap,
    )


def _candidate_counts(gt_rows, candidate_sets):
    by_key = {
        (row["image_id"], row["gt_id"]): row for row in gt_rows
    }
    result = {}
    for name, keys in candidate_sets.items():
        values = [by_key[key] for key in keys]
        result.update({
            f"{name}_n": len(values),
            f"{name}_detected": sum(row["detected"] for row in values),
            f"{name}_with_support": sum(
                row["support_pred_count"] > 0 for row in values),
            f"{name}_group_TP": sum(
                row["detection_type"] == "fragment_group"
                for row in values),
            f"{name}_mean_group_IoU": statistics.fmean(
                row["group_iou"] for row in values),
        })
    return result


def fragment_grid(
        document,
        confidence_thresholds,
        candidate_sets,
        *,
        unique_thresholds=UNIQUE_COVERAGE_THRESHOLDS,
        precision_thresholds=FRAGMENT_PRECISION_THRESHOLDS):
    rows = []
    reference_gt = reference_decisions = None
    standard_by_theta = {
        theta: _standard_micro(document, theta)
        for theta in confidence_thresholds
    }
    for theta in confidence_thresholds:
        standard = standard_by_theta[theta]
        for unique_threshold in unique_thresholds:
            for precision_threshold in precision_thresholds:
                summary, gt_rows, decisions = evaluate_fragment_aware(
                    document,
                    theta,
                    unique_coverage_threshold=unique_threshold,
                    fragment_precision_threshold=precision_threshold,
                )
                row = dict(
                    **summary,
                    standard_TP=standard["TP"],
                    standard_FP=standard["FP"],
                    standard_FN=standard["FN"],
                    standard_F1=standard["F1"],
                    standard_LRP=standard["LRP"],
                    delta_TP_vs_standard=summary["TP"] - standard["TP"],
                    delta_FP_vs_standard=summary["FP"] - standard["FP"],
                    delta_FN_vs_standard=summary["FN"] - standard["FN"],
                    delta_F1_vs_standard=summary["F1"] - standard["F1"],
                    **_candidate_counts(gt_rows, candidate_sets),
                )
                rows.append(row)
                if (
                        theta == .25
                        and unique_threshold
                        == REFERENCE_UNIQUE_COVERAGE
                        and precision_threshold
                        == REFERENCE_FRAGMENT_PRECISION):
                    reference_gt = gt_rows
                    reference_decisions = decisions
    if reference_gt is None:
        raise ValueError("grid must include the reference configuration")
    return rows, reference_gt, reference_decisions


def _extended_counterfactual_documents():
    documents = dict(_counterfactual_documents())

    def pred(identifier, score, xyxy):
        return dict(id=identifier, class_id=0, score=score, xyxy=xyxy)

    documents["grazing_extra"] = (
        dict(
            metadata=dict(class_ids=[0]),
            images=[dict(
                image_id="synthetic",
                width=40,
                height=20,
                gts=[
                    dict(id="gt_a", class_id=0, xyxy=[5, 5, 15, 15]),
                    dict(id="gt_b", class_id=0, xyxy=[20, 5, 30, 15]),
                ],
                predictions=[
                    pred("mostly_a", .95, [5, 5, 14, 15]),
                    pred("pred_b", .90, [20, 5, 30, 15]),
                    pred("grazing", .85, [14, 14, 34, 20]),
                ],
            )],
        ),
        "a low-precision box adds only one percent unique GT coverage",
    )
    documents["three_fragment_group"] = (
        dict(
            metadata=dict(class_ids=[0]),
            images=[dict(
                image_id="synthetic",
                width=20,
                height=10,
                gts=[
                    dict(id="gt", class_id=0, xyxy=[0, 0, 12, 10]),
                ],
                predictions=[
                    pred("left", .95, [0, 0, 4, 10]),
                    pred("middle", .90, [4, 0, 8, 10]),
                    pred("right", .85, [8, 0, 12, 10]),
                ],
            )],
        ),
        "three individually sub-threshold boxes form one exact group",
    )
    return documents


def _region_and_responsibility(document):
    responsibility_rows, _ = score_document(
        document, .25, "hard_iou", .5)
    diagnostics = []
    for image in document["images"]:
        diagnostics.append(image_region_diagnostics(
            [
                Box(row["id"], tuple(row["xyxy"]), row["class_id"])
                for row in image["gts"]
            ],
            [
                Box(
                    row["id"],
                    tuple(row["xyxy"]),
                    row["class_id"],
                    row["score"],
                )
                for row in image["predictions"]
            ],
            (image["width"], image["height"]),
            image_id=image["image_id"],
            config=Config(theta=.25, tau=.5, class_ids=(0,)),
        ))
    region = aggregate(diagnostics)
    return (
        statistics.fmean(
            row["responsibility_dice"] for row in responsibility_rows),
        region["ClassUnionDice"],
    )


def counterfactual_fragment_metrics():
    rows = []
    for scenario, (document, note) in (
            _extended_counterfactual_documents().items()):
        standard = _standard_micro(document, .25)
        fragment, _, _ = evaluate_fragment_aware(
            document,
            .25,
            unique_coverage_threshold=REFERENCE_UNIQUE_COVERAGE,
            fragment_precision_threshold=REFERENCE_FRAGMENT_PRECISION,
        )
        responsibility_dice, class_dice = _region_and_responsibility(
            document)
        events, n_gt = match_class(document, 0, .5, None)
        rows.append(dict(
            scenario=scenario,
            intended_change=note,
            ResponsibilityDice=responsibility_dice,
            ClassUnionDice=class_dice,
            standard_TP=standard["TP"],
            standard_FP=standard["FP"],
            standard_FN=standard["FN"],
            standard_F1=standard["F1"],
            standard_one_minus_LRP=1.0 - standard["LRP"],
            AP50=average_precision(events, n_gt),
            fragment_TP=fragment["TP"],
            fragment_group_TP=fragment["fragment_group_TP"],
            fragment_support=fragment["neutral_support_pred_count"],
            fragment_FP=fragment["FP"],
            fragment_FN=fragment["FN"],
            fragment_F1=fragment["F1"],
            fragment_one_minus_LRP=1.0 - fragment["fragment_LRP"],
        ))
    return rows


def overview_plot(grid_rows, counterfactual_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    primary = [row for row in grid_rows if row["confidence_threshold"] == .25]
    heat = []
    for unique in UNIQUE_COVERAGE_THRESHOLDS:
        heat.append([
            next(
                row for row in primary
                if row["unique_coverage_threshold"] == unique
                and row["fragment_precision_threshold"] == precision
            )["F1"]
            for precision in FRAGMENT_PRECISION_THRESHOLDS
        ])
    image = axes[0, 0].imshow(heat, vmin=0, vmax=1, cmap="viridis")
    axes[0, 0].set(
        title="Fragment-aware F1 at theta=0.25",
        xlabel="Fragment precision threshold",
        ylabel="Unique coverage threshold",
        xticks=np.arange(len(FRAGMENT_PRECISION_THRESHOLDS)),
        xticklabels=FRAGMENT_PRECISION_THRESHOLDS,
        yticks=np.arange(len(UNIQUE_COVERAGE_THRESHOLDS)),
        yticklabels=UNIQUE_COVERAGE_THRESHOLDS,
    )
    for y, values in enumerate(heat):
        for x, value in enumerate(values):
            axes[0, 0].text(
                x, y, f"{value:.3f}", ha="center", va="center",
                color="white" if value < .7 else "black")
    figure.colorbar(image, ax=axes[0, 0], fraction=.046)

    reference = [
        row for row in grid_rows
        if row["unique_coverage_threshold"]
        == REFERENCE_UNIQUE_COVERAGE
        and row["fragment_precision_threshold"]
        == REFERENCE_FRAGMENT_PRECISION
    ]
    theta = [row["confidence_threshold"] for row in reference]
    axes[0, 1].plot(
        theta, [row["standard_F1"] for row in reference],
        marker="o", label="Standard F1")
    axes[0, 1].plot(
        theta, [row["F1"] for row in reference],
        marker="o", label="Fragment-aware F1")
    axes[0, 1].plot(
        theta, [1.0 - row["fragment_LRP"] for row in reference],
        marker="s", label="Fragment-aware 1-LRP")
    axes[0, 1].set(
        title="Reference fragment rule across confidence",
        xlabel="Confidence threshold",
        ylabel="Quality",
    )
    axes[0, 1].set_ylim(0, .7)
    axes[0, 1].grid(alpha=.25)
    axes[0, 1].legend(frameon=False)

    selected = next(
        row for row in primary
        if row["unique_coverage_threshold"]
        == REFERENCE_UNIQUE_COVERAGE
        and row["fragment_precision_threshold"]
        == REFERENCE_FRAGMENT_PRECISION
    )
    labels = ("TP", "FP", "FN")
    x = np.arange(len(labels))
    axes[1, 0].bar(
        x - .18,
        [selected[f"standard_{name}"] for name in labels],
        .36,
        label="Standard",
    )
    axes[1, 0].bar(
        x + .18,
        [selected[name] for name in labels],
        .36,
        label="Fragment-aware",
    )
    axes[1, 0].set(
        title="Reference counts at theta=0.25",
        ylabel="Count",
        xticks=x,
        xticklabels=labels,
    )
    axes[1, 0].legend(frameon=False)

    x = np.arange(len(counterfactual_rows))
    width = .23
    for index, (field, label) in enumerate((
            ("ResponsibilityDice", "Instance"),
            ("standard_F1", "Standard F1"),
            ("fragment_F1", "Fragment-aware F1"))):
        axes[1, 1].bar(
            x + (index - 1) * width,
            [row[field] for row in counterfactual_rows],
            width,
            label=label,
        )
    axes[1, 1].set(
        title="Controlled behavior checks",
        ylabel="Score",
        xticks=x,
        xticklabels=[
            row["scenario"].replace("_", "\n")
            for row in counterfactual_rows
        ],
    )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Fragment-aware matching ablation", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _report_text(source_run, grid_rows, counterfactual_rows):
    primary = [
        row for row in grid_rows if row["confidence_threshold"] == .25
    ]
    reference = next(
        row for row in primary
        if row["unique_coverage_threshold"]
        == REFERENCE_UNIQUE_COVERAGE
        and row["fragment_precision_threshold"]
        == REFERENCE_FRAGMENT_PRECISION
    )
    best = max(primary, key=lambda row: (
        row["F1"],
        row["unique_coverage_threshold"],
        row["fragment_precision_threshold"],
    ))
    counter_lines = [
        (
            f"| {row['scenario']} | {row['standard_F1']:.3f} | "
            f"{row['fragment_F1']:.3f} | {row['fragment_support']} | "
            f"{row['fragment_group_TP']} | {row['fragment_FP']} |"
        )
        for row in counterfactual_rows
    ]
    return f"""# 第六轮：Fragment-aware matching

来源运行：{source_run}。本轮只读取已有缓存，没有推理或训练。

## 参考规则

参考配置只用于读数，不是最终超参数：

~~~text
confidence theta = 0.25
matching IoU = 0.50
minimum UniqueCoverage = {REFERENCE_UNIQUE_COVERAGE:.2f}
minimum FragmentPrecision = {REFERENCE_FRAGMENT_PRECISION:.2f}
group IoU = 0.50
~~~

每个预测先通过 hard_iou 独占一个 GT。每个 GT 内按 score 降序处理额外预测；新增 GT 覆盖达到门槛且预测至少一半面积位于负责 GT 内时，标记为 neutral support。支持碎片不增加 TP，也不计 FP。没有单框达到 IoU50时，至少两个支持碎片的并集 IoU 达到 0.50才形成一个 group TP。

## 真实缓存参考结果

| 版本 | TP | FP | FN | F1 | LRP |
|---|---:|---:|---:|---:|---:|
| standard | {reference['standard_TP']} | {reference['standard_FP']} | {reference['standard_FN']} | {reference['standard_F1']:.6f} | {reference['standard_LRP']:.6f} |
| fragment-aware | {reference['TP']} | {reference['FP']} | {reference['FN']} | {reference['F1']:.6f} | {reference['fragment_LRP']:.6f} |

fragment-aware 结果包含 {reference['single_anchor_TP']} 个单框 anchor TP、{reference['fragment_group_TP']} 个 group TP 和 {reference['neutral_support_pred_count']} 个 neutral support。与标准匹配相比，TP 变化 {reference['delta_TP_vs_standard']:+d}，FP 变化 {reference['delta_FP_vs_standard']:+d}，FN 变化 {reference['delta_FN_vs_standard']:+d}。

第三轮 18 个 single-GT 低重叠候选中，参考规则检测 {reference['single_gt_low_overlap_candidates_detected']} 个，{reference['single_gt_low_overlap_candidates_with_support']} 个包含 support，{reference['single_gt_low_overlap_candidates_group_TP']} 个依靠 group TP。

theta=0.25 的阈值网格中，数值最高的 F1 配置为 UniqueCoverage={best['unique_coverage_threshold']:.2f}、FragmentPrecision={best['fragment_precision_threshold']:.2f}，F1={best['F1']:.6f}。这一最大值不能用于定稿，因为放宽 support 规则会机械地减少 FP；必须结合反例和人工案例判断。

## 受控反例

| 场景 | 标准 F1 | Fragment F1 | Support | Group TP | Fragment FP |
|---|---:|---:|---:|---:|---:|
{chr(10).join(counter_lines)}

complementary_fragments 应从标准 F1=0.667 恢复到 1.0；exact_duplicate 与 small_orphan 应继续受到 FP 惩罚；grazing_extra 的低精度、极小新增框应被拒绝；three_fragment_group 检查三个单框都低于 IoU50时能否由合格并集形成一个 TP。

## 解释边界

Fragment-aware F1/LRP 是研究候选，不是标准 F1/LRP。标准结果必须始终并列报告。UniqueCoverage 采用相对已接受高分预测的顺序边际，因此是确定且置信度相关的，但不是顺序无关的 Shapley 分配。门槛不能用单个模型的 F1 最大值直接选择。

当前规则只说明几何上的独有覆盖和外溢，没有证明碎片在语义或下游任务上有用。第三轮看图发现的擦边与相邻目标歧义仍需通过 fragment_decisions.csv 和案例可视化检查。

## 下一步

先分析阈值网格、18 个低重叠候选和 group TP 明细，判断参考门槛是否过松。随后把最有争议的 support、rejected 和 group TP 生成可视化并人工标注。只有人工结果支持规则后，才冻结 fragment-aware 定义并扩展多个模型和 seed。
"""


def analyze_fragments(
        source_run,
        *,
        confidence_thresholds=(0.05, 0.10, 0.25, 0.50, 0.75),
        output=None):
    source_run, document, _, _, complete, _ = validate_source(source_run)
    class_names = document["metadata"]["class_names"]
    _, primary_images, _ = evaluate_document(
        document, theta=.25, tau=.5)
    local_rows = local_gt_diagnostics(
        primary_images, document, class_names, .25)
    candidate_sets = _candidate_sets(local_rows)
    grid_rows, reference_gt, reference_decisions = fragment_grid(
        document, confidence_thresholds, candidate_sets)
    counterfactual_rows = counterfactual_fragment_metrics()

    output = (
        Path(output).resolve() if output else
        PROJECT_ROOT / "outputs" / "region_eval" /
        ("fragment_analysis_" +
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
            unique_coverage_thresholds=UNIQUE_COVERAGE_THRESHOLDS,
            fragment_precision_thresholds=(
                FRAGMENT_PRECISION_THRESHOLDS),
            reference_unique_coverage=REFERENCE_UNIQUE_COVERAGE,
            reference_fragment_precision=(
                REFERENCE_FRAGMENT_PRECISION),
            matching_iou=.5,
            group_iou=.5,
        ),
        source_code_sha256={
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"):
            _file_hash(path)
            for path in (
                PROJECT_ROOT / "region_eval" / "fragment_analysis.py",
                PROJECT_ROOT / "run_region_fragments.py",
            )
        },
    ))

    _write_csv(
        output / "fragment_grid.csv",
        grid_rows,
        list(grid_rows[0]),
    )
    reference_rows = [
        row for row in grid_rows
        if row["unique_coverage_threshold"]
        == REFERENCE_UNIQUE_COVERAGE
        and row["fragment_precision_threshold"]
        == REFERENCE_FRAGMENT_PRECISION
    ]
    _write_csv(
        output / "reference_threshold.csv",
        reference_rows,
        list(reference_rows[0]),
    )
    gt_rows = []
    for row in reference_gt:
        key = (row["image_id"], row["gt_id"])
        gt_rows.append(dict(
            row,
            class_name=class_names[row["class_id"]],
            original_positive_CG_with_multigt_candidate=(
                key in candidate_sets[
                    "original_positive_CG_with_multigt"]),
            single_gt_low_overlap_candidate=(
                key in candidate_sets[
                    "single_gt_low_overlap_candidates"]),
        ))
    gt_fields = [
        "image_id", "gt_id", "class_id", "class_name",
        "original_positive_CG_with_multigt_candidate",
        "single_gt_low_overlap_candidate",
        "confidence_threshold", "matching_iou", "group_iou_threshold",
        "unique_coverage_threshold", "fragment_precision_threshold",
        "detection_type", "detected", "anchor_pred_id",
        "assigned_pred_count", "accepted_pred_count",
        "support_pred_count", "rejected_pred_count",
        "group_pred_union_area", "group_intersection_area",
        "group_coverage", "group_precision", "group_dice", "group_iou",
    ]
    _write_csv(
        output / "fragment_gt.csv", gt_rows, gt_fields)
    _write_csv(
        output / "fragment_decisions.csv",
        reference_decisions,
        list(reference_decisions[0]),
    )
    _write_csv(
        output / "counterfactual_fragment_matrix.csv",
        counterfactual_rows,
        list(counterfactual_rows[0]),
    )
    overview_plot(
        grid_rows, counterfactual_rows, output / "overview.png")
    (output / "README.md").write_text(
        _report_text(source_run, grid_rows, counterfactual_rows),
        encoding="utf-8",
        newline="\n",
    )
    counts = dict(
        source_images=len(document["images"]),
        source_gt=complete["summary"]["n_gt"],
        source_predictions_at_reference=next(
            row["n_pred"] for row in reference_rows
            if row["confidence_threshold"] == .25),
        grid_rows=len(grid_rows),
        reference_gt_rows=len(reference_gt),
        reference_prediction_decisions=len(reference_decisions),
        low_overlap_candidates=len(
            candidate_sets["single_gt_low_overlap_candidates"]),
        counterfactual_scenarios=len(counterfactual_rows),
    )
    reference_summary = next(
        row for row in reference_rows
        if row["confidence_threshold"] == .25)
    _write_json(output / "analysis_complete.json", dict(
        status="complete",
        analysis_version=ANALYSIS_VERSION,
        source_run=str(source_run),
        counts=counts,
        reference_summary=reference_summary,
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
        "SUCCESS: fragment-aware analysis complete. "
        "No inference or training performed.",
        flush=True,
    )
    return output
