from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

from .case_analysis import (
    _file_hash,
    _read_json,
    _write_csv,
    _write_json,
    harmonic,
    validate_source,
)
from .fragment_analysis import evaluate_fragment_aware, _standard_micro
from .geometry import area, intersection
from .offline import PROJECT_ROOT
from .responsibility_analysis import rectangle_iou


ANALYSIS_VERSION = "vc-conservative-refinement-0.1.0"
NOVELTY_REFERENCE = 0.25


def count_metrics(tp, fp, fn):
    if min(tp, fp, fn) < 0:
        raise ValueError("TP, FP, and FN must be nonnegative")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        precision=precision,
        recall=recall,
        F1=harmonic(precision, recall),
    )


def novelty_ratio(marginal_unique_area, owner_overlap_area):
    if marginal_unique_area < 0 or owner_overlap_area < 0:
        raise ValueError("areas must be nonnegative")
    if not owner_overlap_area:
        return 0.0
    return min(1.0, marginal_unique_area / owner_overlap_area)


def _validate_completed_run(path, expected_version=None):
    path = Path(path).resolve()
    manifest_path = path / "analysis_manifest.json"
    complete_path = path / "analysis_complete.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError(f"analysis run is incomplete: {path}")
    manifest = _read_json(manifest_path)
    complete = _read_json(complete_path)
    if complete.get("status") != "complete":
        raise ValueError(f"analysis run is not complete: {path}")
    if expected_version and complete.get("analysis_version") != expected_version:
        raise ValueError(f"unexpected analysis version: {path}")
    for relative, expected in complete.get("output_sha256", {}).items():
        target = path / relative
        if not target.is_file() or _file_hash(target) != expected:
            raise ValueError(f"output hash mismatch: {target}")
    return path, manifest, complete


def _geometry_indexes(document):
    gt_index = {}
    pred_index = {}
    for image in document["images"]:
        for row in image["gts"]:
            gt_index[(image["image_id"], row["id"])] = row
        for row in image["predictions"]:
            pred_index[(image["image_id"], row["id"])] = row
    return gt_index, pred_index


def enrich_support_geometry(document, gt_rows, decisions, matching_iou):
    gt_index, pred_index = _geometry_indexes(document)
    gt_results = {
        (row["image_id"], row["gt_id"]): row for row in gt_rows
    }
    accepted_by_gt = defaultdict(list)
    for row in decisions:
        if row["final_status"] in ("anchor_tp", "support", "group_support"):
            accepted_by_gt[(row["image_id"], row["owner_gt_id"])].append(row)
    rows = []
    for decision in decisions:
        if decision["final_status"] not in ("support", "group_support"):
            continue
        key = (decision["image_id"], decision["owner_gt_id"])
        gt = gt_index[key]
        prediction = pred_index[(decision["image_id"], decision["pred_id"])]
        overlap = intersection(tuple(gt["xyxy"]), tuple(prediction["xyxy"]))
        overlap_area = area(overlap)
        peers = [
            row for row in accepted_by_gt[key]
            if row["pred_id"] != decision["pred_id"]
        ]
        pairwise = [
            rectangle_iou(
                prediction["xyxy"],
                pred_index[(decision["image_id"], row["pred_id"])]["xyxy"],
            )
            for row in peers
        ]
        gt_result = gt_results[key]
        rows.append(dict(
            image_id=decision["image_id"],
            gt_id=decision["owner_gt_id"],
            pred_id=decision["pred_id"],
            final_status=decision["final_status"],
            detection_type=gt_result["detection_type"],
            score=decision["score"],
            source_gt_hit_count=decision["source_gt_hit_count"],
            owner_iou=decision["owner_iou"],
            individual_iou_ge_matching=(
                decision["owner_iou"] >= matching_iou),
            fragment_precision=decision["fragment_precision"],
            marginal_unique_area=decision["marginal_unique_area"],
            marginal_unique_coverage=(
                decision["marginal_unique_coverage"]),
            owner_overlap_area=overlap_area,
            novelty_ratio=novelty_ratio(
                decision["marginal_unique_area"], overlap_area),
            max_pairwise_iou_with_accepted=(
                max(pairwise) if pairwise else None),
            group_iou=gt_result["group_iou"],
            group_precision=gt_result["group_precision"],
        ))
    return rows


def policy_rows(document, confidence_thresholds, matching_iou=.5):
    rows = []
    support_rows_at_reference = None
    for theta in confidence_thresholds:
        standard = _standard_micro(document, theta, matching_iou)
        current, gt_rows, decisions = evaluate_fragment_aware(
            document,
            theta,
            matching_iou=matching_iou,
            unique_coverage_threshold=.02,
            fragment_precision_threshold=.50,
            group_iou_threshold=.50,
        )
        support_geometry = enrich_support_geometry(
            document, gt_rows, decisions, matching_iou)
        ordinary = [
            row for row in support_geometry if row["final_status"] == "support"]
        sub_iou = [
            row for row in ordinary if row["owner_iou"] < matching_iou]
        strict = [
            row for row in sub_iou
            if row["novelty_ratio"] >= NOVELTY_REFERENCE]

        def add(policy, metrics, neutral_support, group_tp, note):
            rows.append(dict(
                confidence_threshold=theta,
                policy=policy,
                n_gt=current["n_gt"],
                n_pred=current["n_pred"],
                neutral_support_pred_count=neutral_support,
                fragment_group_TP=group_tp,
                **metrics,
                delta_F1_vs_standard=metrics["F1"] - standard["F1"],
                interpretation=note,
            ))

        add(
            "standard_one_to_one_recommended",
            {key: standard[key] for key in (
                "TP", "FP", "FN", "precision", "recall", "F1")},
            0,
            0,
            "Retain standard FP/FN accounting; report regional evidence separately.",
        )
        add(
            "round6_fragment_aware",
            {key: current[key] for key in (
                "TP", "FP", "FN", "precision", "recall", "F1")},
            current["neutral_support_pred_count"],
            current["fragment_group_TP"],
            "Round-6 rule shown for comparison; rejected as primary VC count layer.",
        )
        for policy, accepted, note in (
            (
                "no_group_all_ordinary_support",
                ordinary,
                "Remove group TP but keep every ordinary round-6 support.",
            ),
            (
                "no_group_support_iou_lt_050",
                sub_iou,
                "Also reject support boxes that individually meet IoU50.",
            ),
            (
                "no_group_support_iou_lt_050_novelty_ge_025",
                strict,
                "Keep only sub-IoU50 boxes with NoveltyRatio at least 0.25.",
            ),
        ):
            tp = current["single_anchor_TP"]
            metrics = count_metrics(
                tp,
                current["n_pred"] - tp - len(accepted),
                current["n_gt"] - tp,
            )
            add(policy, metrics, len(accepted), 0, note)
        if theta == .25:
            support_rows_at_reference = support_geometry
    return rows, support_rows_at_reference


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _assistant_pre_review(audit_run, strict_support_rows):
    queue = _read_csv(audit_run / "audit_queue.csv")
    rows = []
    for case in queue:
        if case["category"] not in (
                "group_tp", "support_multigt", "support_single_gt",
                "borderline_support"):
            continue
        if case["category"] == "support_multigt":
            label = "duplicate_or_neighbor"
            note = (
                "Contact-sheet pre-review shows overlapping whole-object "
                "proposals in crowded context; owner assignment does not "
                "resolve semantic neighbor ambiguity.")
        else:
            label = "duplicate_or_jitter"
            note = (
                "Contact-sheet pre-review shows shifted overlapping "
                "whole-object proposals rather than clear part fragments.")
        rows.append(dict(
            case_id=case["case_id"],
            source_group=case["category"],
            image_id=case["image_id"],
            gt_id=case["gt_id"],
            pred_id=case["focus_pred_id"] or None,
            assistant_preliminary_label=label,
            assistant_recommend_keep_as_support="no",
            assistant_confidence="medium",
            requires_user_confirmation=True,
            notes=note,
        ))
    strict_ids = {
        ("0000269_00001_d_0000348", "gt_000065", "pred_000039"),
        ("0000022_00000_d_0000004", "gt_000012", "pred_000021"),
    }
    strict_lookup = {
        (row["image_id"], row["gt_id"], row["pred_id"]): row
        for row in strict_support_rows
    }
    for rank, key in enumerate(sorted(strict_ids), 1):
        if key not in strict_lookup:
            raise ValueError(f"strict visual candidate disappeared: {key}")
        rows.append(dict(
            case_id=f"strict_gate_visual_check_{rank:02d}",
            source_group="strict_geometry_survivor",
            image_id=key[0],
            gt_id=key[1],
            pred_id=key[2],
            assistant_preliminary_label="duplicate_or_jitter",
            assistant_recommend_keep_as_support="no",
            assistant_confidence="medium",
            requires_user_confirmation=True,
            notes=(
                "Extra enlarged pre-review still shows an offset whole-object "
                "proposal, despite IoU<0.50 and NoveltyRatio>=0.25."),
        ))
    return rows


def _vc_profile(matching_run):
    rows = _read_csv(matching_run / "three_layer_threshold.csv")
    row = next(
        value for value in rows
        if float(value["confidence_threshold"]) == .25)
    ap = _read_json(matching_run / "ap_summary.json")
    return dict(
        status="provisional_profile_not_frozen_scalar",
        confidence_threshold=.25,
        instance_ResponsibilityDice=float(row["instance_ResponsibilityDice"]),
        region_ClassUnionDice=float(row["region_ClassUnionDice"]),
        standard_detection_F1=float(row["detection_F1"]),
        standard_detection_one_minus_LRP=float(
            row["detection_one_minus_LRP"]),
        standard_AP50=float(ap["AP50"]),
        standard_AP50_95=float(ap["AP50_95"]),
        provisional_H3_instance_region_F1=float(
            row["H3_instance_region_F1"]),
        provisional_H3_instance_region_one_minus_LRP=float(
            row["H3_instance_region_one_minus_LRP"]),
        collective_gain_role="diagnostic_only",
        fragment_aware_count_role="diagnostic_only_rejected_as_primary",
    )


def overview_plot(policy, support_rows, profile, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    theta25 = [row for row in policy if row["confidence_threshold"] == .25]
    labels = [
        "Standard", "Round 6", "No group", "IoU<.50", "+ novelty"
    ]
    axes[0, 0].bar(
        np.arange(len(theta25)),
        [row["F1"] for row in theta25],
        color=["#0072B2", "#D55E00", "#CC79A7", "#E69F00", "#009E73"],
    )
    axes[0, 0].set(
        title="Count-policy F1 at theta=0.25",
        ylabel="F1",
        xticks=np.arange(len(theta25)),
        xticklabels=labels,
    )
    axes[0, 0].tick_params(axis="x", rotation=18)
    axes[0, 0].set_ylim(.50, .545)

    ordinary = [row for row in support_rows if row["final_status"] == "support"]
    counts = [
        sum(row["owner_iou"] >= .5 for row in ordinary),
        sum(row["owner_iou"] < .5 for row in ordinary),
        sum(
            row["owner_iou"] < .5
            and row["novelty_ratio"] >= NOVELTY_REFERENCE
            for row in ordinary),
    ]
    axes[0, 1].bar(
        ["IoU >= .50", "IoU < .50", "IoU < .50\nand novelty >= .25"],
        counts,
        color=["#D55E00", "#E69F00", "#009E73"],
    )
    axes[0, 1].set(title="Round-6 ordinary support geometry", ylabel="Predictions")
    for index, value in enumerate(counts):
        axes[0, 1].text(index, value + 1, str(value), ha="center")

    policies = {
        "standard_one_to_one_recommended": "Standard",
        "round6_fragment_aware": "Round 6",
        "no_group_support_iou_lt_050_novelty_ge_025": "Strict gate",
    }
    for name, label in policies.items():
        values = [row for row in policy if row["policy"] == name]
        axes[1, 0].plot(
            [row["confidence_threshold"] for row in values],
            [row["F1"] for row in values],
            marker="o",
            label=label,
        )
    axes[1, 0].set(
        title="F1 across confidence thresholds",
        xlabel="Confidence threshold",
        ylabel="F1",
    )
    axes[1, 0].grid(alpha=.25)
    axes[1, 0].legend(frameon=False)

    component_names = ["Instance", "Region", "F1", "1-LRP", "AP50:95"]
    component_values = [
        profile["instance_ResponsibilityDice"],
        profile["region_ClassUnionDice"],
        profile["standard_detection_F1"],
        profile["standard_detection_one_minus_LRP"],
        profile["standard_AP50_95"],
    ]
    axes[1, 1].bar(
        component_names,
        component_values,
        color="#0072B2",
    )
    axes[1, 1].set(
        title="Conservative VC profile at theta=0.25",
        ylabel="Score",
    )
    axes[1, 1].set_ylim(0, .6)
    axes[1, 1].tick_params(axis="x", rotation=18)
    figure.suptitle(
        "Round 8: visual audit rejects fragment exemptions in the count layer",
        fontsize=16,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _report_text(
        source_run,
        fragment_run,
        audit_run,
        policy,
        support_rows,
        pre_review,
        profile):
    ordinary = [row for row in support_rows if row["final_status"] == "support"]
    ge_iou = sum(row["owner_iou"] >= .5 for row in ordinary)
    sub_iou = [row for row in ordinary if row["owner_iou"] < .5]
    strict = [
        row for row in sub_iou if row["novelty_ratio"] >= NOVELTY_REFERENCE]
    theta25 = {
        row["policy"]: row for row in policy
        if row["confidence_threshold"] == .25
    }
    policy_lines = "\n".join(
        f"| {row['policy']} | {row['neutral_support_pred_count']} | "
        f"{row['fragment_group_TP']} | {row['TP']} | {row['FP']} | "
        f"{row['FN']} | {row['F1']:.6f} |"
        for row in policy if row["confidence_threshold"] == .25
    )
    return f"""# 第八轮：视觉证据后的 VC 收紧

本轮只读取已有缓存和第六、七轮结果，没有训练或推理。

来源缓存：`{source_run}`  
第六轮：`{fragment_run}`  
第七轮：`{audit_run}`

## 视觉预审结论

已放大检查全部 4 个 group TP，以及第七轮抽取的 18 个 ordinary support。没有发现清楚的“多个局部片段共同组成一个目标”案例；画面主要是同一目标的偏移完整框、重复检测，或拥挤场景中的相邻实例歧义。

几何统计与视觉结果一致：82 个 ordinary support 中，{ge_iou} 个本身已经达到 IoU50，占 {ge_iou / len(ordinary):.1%}。它们在标准一对一匹配中是重复检测，不应因为增加少量边缘覆盖就免除 FP。

剩余 {len(sub_iou)} 个单框 IoU<0.50 的 support 中，只有 {len(strict)} 个同时达到 NoveltyRatio≥{NOVELTY_REFERENCE:.2f}。这两个最严格候选也已额外放大检查，仍表现为偏移的完整框，而不是有明确语义的局部碎片。

`assistant_visual_pre_review.csv` 保存了 {len(pre_review)} 条预审记录。它是模型视觉判断，全部标记为需要用户确认，不会写入第七轮空白人工标签表。

## 策略对照

theta=0.25：

| 计数策略 | Neutral support | Group TP | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|---:|
{policy_lines}

严格门槛只留下 {len(strict)} 个 support，F1={theta25['no_group_support_iou_lt_050_novelty_ge_025']['F1']:.6f}，与标准 F1={theta25['standard_one_to_one_recommended']['F1']:.6f} 的差异仅为 {theta25['no_group_support_iou_lt_050_novelty_ge_025']['delta_F1_vs_standard']:+.6f}；而这两个候选视觉上仍不支持免罚。因此继续添加阈值只会让规则更复杂，不能解决语义问题。

## 当前决定

Fragment-aware matching 保留为诊断实验，但不进入主 VC 的计数层：

1. 不使用 group TP。
2. 不把额外预测改成 neutral support。
3. FP/FN 继续采用标准一对一匹配、AP 和 LRP/F1。
4. 多框联合覆盖由 ResponsibilityDice、ClassUnionDice 和 CollectiveGain 描述；CollectiveGain 只用于解释“额外框增加了多少覆盖”，不直接抵消 FP。

这样，大框会被区域/责任精度惩罚，重复框会被标准 FP 惩罚，漏检会进入 FN；多框确实改善覆盖时，实例与区域分量仍会提高，但不能逃避计数代价。

## 保守 VC profile

theta=0.25 的三层 profile：

~~~text
I = ResponsibilityDice = {profile['instance_ResponsibilityDice']:.6f}
R = ClassUnionDice      = {profile['region_ClassUnionDice']:.6f}
D_F1 = standard F1      = {profile['standard_detection_F1']:.6f}
D_LRP = 1 - LRP         = {profile['standard_detection_one_minus_LRP']:.6f}
AP50:95                 = {profile['standard_AP50_95']:.6f}
~~~

当前建议先把 `(I, R, D)` 作为主 profile，并列报告 AP50:95 与 CollectiveGain。若论文或模型排序必须使用单一标量，可以暂时研究等权调和平均：

~~~text
VC-H3-F1 = 3 / (1/I + 1/R + 1/D_F1)
         = {profile['provisional_H3_instance_region_F1']:.6f}
~~~

这个标量仍未冻结，因为只有一个模型时无法检验不同权重是否改变模型排序。使用 `1-LRP` 的对应候选为 {profile['provisional_H3_instance_region_one_minus_LRP']:.6f}，差异说明计数分量的选择会明显影响总分。

## 训练边界

上述决定完全由已有预测得到，不需要重新训练。下一项计算工作应是收集其他已有 checkpoint 或模型的预测缓存；这需要推理，但不需要训练。只有以后把 VC 用作 loss 或训练期选模依据，才需要重新训练。

## 输出

- `policy_threshold.csv`：五种计数策略在五档 confidence 下的对照。
- `support_geometry.csv`：全部 round-6 support 的 IoU、NoveltyRatio 和框间重叠。
- `assistant_visual_pre_review.csv`：24 条需要用户确认的视觉预审。
- `vc_profile.json`：保守三层 profile 与两个临时 H3 候选。
- `overview.png`：策略、support 几何和 profile 概览。

## 下一步

先由用户核对 `assistant_visual_pre_review.csv` 和第七轮图片。如果同意“这些主要是重复/偏移完整框”，就正式撤下 Fragment-aware count，并把第九轮用于收集至少 3 个已有 checkpoint 或不同规模模型的预测，检查 `(I,R,D)` 与 H3 的排序稳定性。该步骤优先复用已有权重，只做推理缓存，不重新训练。
"""


def analyze_vc_refinement(
        fragment_run,
        audit_run,
        matching_run,
        *,
        output=None):
    fragment_run, fragment_manifest, fragment_complete = _validate_completed_run(
        fragment_run, "fragment-aware-matching-0.1.0")
    audit_run, audit_manifest, audit_complete = _validate_completed_run(
        audit_run, "fragment-visual-audit-0.1.0")
    matching_run, matching_manifest, matching_complete = _validate_completed_run(
        matching_run, "region-matching-analysis-0.1.0")
    if Path(audit_complete["fragment_run"]).resolve() != fragment_run:
        raise ValueError("audit run does not reference the selected fragment run")
    source_run = Path(fragment_complete["source_run"]).resolve()
    if Path(matching_complete["source_run"]).resolve() != source_run:
        raise ValueError("matching and fragment runs use different source caches")
    source_run, document, _, _, _, _ = validate_source(source_run)
    confidence_thresholds = tuple(
        fragment_manifest["parameters"]["confidence_thresholds"])
    policy, support_rows = policy_rows(document, confidence_thresholds)
    strict = [
        row for row in support_rows
        if row["final_status"] == "support"
        and row["owner_iou"] < .5
        and row["novelty_ratio"] >= NOVELTY_REFERENCE
    ]
    pre_review = _assistant_pre_review(audit_run, strict)
    profile = _vc_profile(matching_run)

    output = (
        Path(output).resolve() if output else
        PROJECT_ROOT / "results" / "region_eval" /
        ("vc_refinement_" +
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
        fragment_run=str(fragment_run),
        audit_run=str(audit_run),
        matching_run=str(matching_run),
        source_filtered_sha256=_file_hash(
            source_run / "filtered_predictions.json"),
        input_complete_sha256={
            "fragment": _file_hash(fragment_run / "analysis_complete.json"),
            "audit": _file_hash(audit_run / "analysis_complete.json"),
            "matching": _file_hash(matching_run / "analysis_complete.json"),
        },
        parameters=dict(
            confidence_thresholds=confidence_thresholds,
            matching_iou=.5,
            original_unique_coverage=.02,
            original_fragment_precision=.5,
            novelty_reference=NOVELTY_REFERENCE,
        ),
        source_code_sha256={
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"):
            _file_hash(path)
            for path in (
                PROJECT_ROOT / "region_eval" / "vc_refinement.py",
                PROJECT_ROOT / "run_region_vc_refinement.py",
            )
        },
    ))
    _write_csv(output / "policy_threshold.csv", policy, list(policy[0]))
    _write_csv(
        output / "support_geometry.csv", support_rows, list(support_rows[0]))
    _write_csv(
        output / "assistant_visual_pre_review.csv",
        pre_review,
        list(pre_review[0]),
    )
    _write_json(output / "vc_profile.json", profile)
    overview_plot(policy, support_rows, profile, output / "overview.png")
    (output / "README.md").write_text(
        _report_text(
            source_run,
            fragment_run,
            audit_run,
            policy,
            support_rows,
            pre_review,
            profile,
        ),
        encoding="utf-8",
        newline="\n",
    )
    ordinary = [
        row for row in support_rows if row["final_status"] == "support"]
    counts = dict(
        source_images=len(document["images"]),
        source_gt=fragment_complete["reference_summary"]["n_gt"],
        source_predictions=fragment_complete["reference_summary"]["n_pred"],
        policy_rows=len(policy),
        ordinary_support=len(ordinary),
        ordinary_support_individual_iou_ge_050=sum(
            row["owner_iou"] >= .5 for row in ordinary),
        ordinary_support_individual_iou_lt_050=sum(
            row["owner_iou"] < .5 for row in ordinary),
        strict_support_candidates=len(strict),
        group_tp=fragment_complete["reference_summary"]["fragment_group_TP"],
        assistant_pre_review_rows=len(pre_review),
    )
    _write_json(output / "analysis_complete.json", dict(
        status="complete",
        analysis_version=ANALYSIS_VERSION,
        counts=counts,
        decision=dict(
            fragment_aware_count_primary=False,
            standard_one_to_one_count_primary=True,
            vc_profile_primary_components=[
                "ResponsibilityDice", "ClassUnionDice",
                "standard_detection_quality"],
            CollectiveGain="diagnostic_only",
        ),
        output_sha256={
            str(path.relative_to(output)).replace("\\", "/"):
            _file_hash(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name not in (
                "analysis_manifest.json", "analysis_complete.json")
        },
    ))
    print(json.dumps(dict(output=str(output), **counts), ensure_ascii=False), flush=True)
    print(
        "SUCCESS: conservative VC refinement complete. "
        "No inference or training performed.",
        flush=True,
    )
    return output
