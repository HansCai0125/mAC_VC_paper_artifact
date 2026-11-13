from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .case_analysis import (
    _file_hash,
    _fit_crop,
    _font,
    _read_json,
    _write_csv,
    _write_json,
    contact_sheet,
    detail_crop,
    local_gt_diagnostics,
    select_diverse,
    validate_source,
)
from .fragment_analysis import (
    ANALYSIS_VERSION as FRAGMENT_ANALYSIS_VERSION,
    _candidate_sets,
    evaluate_fragment_aware,
)
from .offline import PROJECT_ROOT, evaluate_document


ANALYSIS_VERSION = "fragment-visual-audit-0.1.0"
AUDIT_GROUPS = (
    "group_tp",
    "support_multigt",
    "support_single_gt",
    "borderline_support",
    "borderline_rejected",
    "low_overlap_candidates",
)
HUMAN_LABELS = (
    "valid_complement",
    "duplicate_or_jitter",
    "neighbor_instance_borrowing",
    "background_grazing",
    "annotation_or_scale_ambiguity",
    "uncertain",
)


ROLE_COLORS = {
    "anchor_tp": (0, 114, 178, 235),
    "support": (230, 159, 0, 235),
    "group_support": (151, 77, 191, 235),
    "rejected_fp": (213, 94, 0, 235),
    "failed_group_fp": (166, 54, 55, 235),
    "orphan_fp": (70, 70, 70, 210),
    "unclassified": (160, 160, 160, 170),
}
ROLE_ABBREVIATIONS = {
    "anchor_tp": "A",
    "support": "S",
    "group_support": "G",
    "rejected_fp": "R",
    "failed_group_fp": "F",
    "orphan_fp": "O",
    "unclassified": "?",
}


def _check_fraction(value, name):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def support_boundary_margin(
        decision,
        unique_coverage_threshold,
        fragment_precision_threshold):
    """Normalized distance inside both support thresholds; zero is boundary."""
    unique = _check_fraction(
        unique_coverage_threshold, "unique_coverage_threshold")
    precision = _check_fraction(
        fragment_precision_threshold, "fragment_precision_threshold")
    unique_scale = unique if unique > 0 else 1.0
    precision_scale = precision if precision > 0 else 1.0
    return min(
        (decision["marginal_unique_coverage"] - unique) / unique_scale,
        (decision["fragment_precision"] - precision) / precision_scale,
    )


def rejection_boundary_deficit(
        decision,
        unique_coverage_threshold,
        fragment_precision_threshold):
    """Normalized amount needed to pass both support thresholds."""
    unique = _check_fraction(
        unique_coverage_threshold, "unique_coverage_threshold")
    precision = _check_fraction(
        fragment_precision_threshold, "fragment_precision_threshold")
    unique_scale = unique if unique > 0 else 1.0
    precision_scale = precision if precision > 0 else 1.0
    return (
        max(0.0, unique - decision["marginal_unique_coverage"])
        / unique_scale
        + max(0.0, precision - decision["fragment_precision"])
        / precision_scale
    )


def validate_fragment_run(fragment_run):
    fragment_run = Path(fragment_run).resolve()
    manifest_path = fragment_run / "analysis_manifest.json"
    complete_path = fragment_run / "analysis_complete.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError("fragment run is missing manifest or completion marker")
    manifest = _read_json(manifest_path)
    complete = _read_json(complete_path)
    if manifest.get("analysis_version") != FRAGMENT_ANALYSIS_VERSION:
        raise ValueError("fragment run version does not match this audit")
    if complete.get("status") != "complete":
        raise ValueError("fragment run is not complete")
    for relative, expected in complete.get("output_sha256", {}).items():
        path = fragment_run / relative
        if not path.is_file() or _file_hash(path) != expected:
            raise ValueError(f"fragment output hash mismatch: {relative}")
    source_run = Path(complete["source_run"]).resolve()
    if _file_hash(source_run / "filtered_predictions.json") != manifest[
            "source_filtered_sha256"]:
        raise ValueError("fragment source cache hash has changed")
    return fragment_run, manifest, complete, source_run


def _roles(decisions):
    counts = Counter(row["final_status"] for row in decisions)
    return ";".join(f"{name}:{counts[name]}" for name in sorted(counts))


def _make_case(
        category,
        selection_score,
        gt,
        owned,
        local,
        class_names,
        focus=None):
    return dict(
        category=category,
        selection_score=float(selection_score),
        image_id=gt["image_id"],
        class_id=gt["class_id"],
        class_name=class_names[gt["class_id"]],
        gt_id=gt["gt_id"],
        focus_pred_id=focus["pred_id"] if focus else None,
        focus_status=focus["final_status"] if focus else None,
        focus_score=focus["score"] if focus else None,
        source_gt_hit_count=focus["source_gt_hit_count"] if focus else None,
        owner_iou=focus["owner_iou"] if focus else None,
        fragment_precision=focus["fragment_precision"] if focus else None,
        marginal_unique_coverage=(
            focus["marginal_unique_coverage"] if focus else None),
        detection_type=gt["detection_type"],
        detected=gt["detected"],
        assigned_pred_count=gt["assigned_pred_count"],
        accepted_pred_count=gt["accepted_pred_count"],
        support_pred_count=gt["support_pred_count"],
        rejected_pred_count=gt["rejected_pred_count"],
        group_coverage=gt["group_coverage"],
        group_precision=gt["group_precision"],
        group_dice=gt["group_dice"],
        group_iou=gt["group_iou"],
        owned_prediction_roles=_roles(owned),
        owned_prediction_ids=";".join(
            row["pred_id"] for row in sorted(
                owned, key=lambda value: (-value["score"], value["pred_id"]))),
        original_positive_CG_with_multigt_candidate=(
            bool(local)
            and local["CollectiveGain_i"] > 1e-9
            and local["contributing_multigt_count"] > 0),
        single_gt_low_overlap_candidate=(
            bool(local)
            and local["CollectiveGain_i"] > 1e-9
            and local["contributing_pred_count"] >= 2
            and local["contributing_multigt_count"] == 0
            and local["max_contributing_pairwise_iou"] is not None
            and local["max_contributing_pairwise_iou"] < .3),
        BestIoGT_i=local["BestIoGT_i"] if local else None,
        CollectiveGain_i=local["CollectiveGain_i"] if local else None,
    )


def select_audit_cases(
        gt_rows,
        decisions,
        local_rows,
        class_names,
        *,
        unique_coverage_threshold,
        fragment_precision_threshold,
        cases_per_group=6):
    if type(cases_per_group) is not int or cases_per_group < 1:
        raise ValueError("cases_per_group must be a positive integer")
    gt_by_key = {
        (row["image_id"], row["gt_id"]): row for row in gt_rows
    }
    local_by_key = {
        (row["image_id"], row["gt_id"]): row for row in local_rows
    }
    decisions_by_gt = defaultdict(list)
    for row in decisions:
        if row["owner_gt_id"] is not None:
            decisions_by_gt[(row["image_id"], row["owner_gt_id"])].append(
                row)

    chosen = []
    used_predictions = set()

    group_candidates = []
    for key, gt in gt_by_key.items():
        if gt["detection_type"] != "fragment_group":
            continue
        owned = decisions_by_gt[key]
        group_candidates.append(_make_case(
            "group_tp",
            1.0 / (1.0 + abs(gt["group_iou"] - .5)),
            gt,
            owned,
            local_by_key.get(key),
            class_names,
        ))
    group_candidates.sort(key=lambda row: (
        row["group_iou"], row["image_id"], row["gt_id"]))
    chosen.extend(group_candidates)
    for case in group_candidates:
        used_predictions.update(case["owned_prediction_ids"].split(";"))

    def support_candidates(category, predicate, score):
        candidates = []
        for decision in decisions:
            if decision["final_status"] != "support":
                continue
            if decision["pred_id"] in used_predictions or not predicate(decision):
                continue
            key = (decision["image_id"], decision["owner_gt_id"])
            candidates.append(_make_case(
                category,
                score(decision),
                gt_by_key[key],
                decisions_by_gt[key],
                local_by_key.get(key),
                class_names,
                decision,
            ))
        selected = select_diverse(candidates, cases_per_group)
        if len(selected) < cases_per_group:
            raise ValueError(f"not enough cases for {category}")
        chosen.extend(selected)
        used_predictions.update(row["focus_pred_id"] for row in selected)

    support_candidates(
        "support_multigt",
        lambda row: row["source_gt_hit_count"] >= 2,
        lambda row: (
            10.0 * row["source_gt_hit_count"]
            + row["marginal_unique_coverage"]
            + row["fragment_precision"]),
    )
    support_candidates(
        "support_single_gt",
        lambda row: row["source_gt_hit_count"] == 1,
        lambda row: (
            row["marginal_unique_coverage"]
            + row["fragment_precision"]),
    )
    support_candidates(
        "borderline_support",
        lambda row: True,
        lambda row: 1.0 / (
            1.0 + max(0.0, support_boundary_margin(
                row,
                unique_coverage_threshold,
                fragment_precision_threshold))),
    )

    rejected_candidates = []
    for decision in decisions:
        if (
                decision["final_status"] != "rejected_fp"
                or decision["owner_gt_id"] is None
                or decision["fragment_precision"] <= 0.0):
            continue
        key = (decision["image_id"], decision["owner_gt_id"])
        deficit = rejection_boundary_deficit(
            decision,
            unique_coverage_threshold,
            fragment_precision_threshold,
        )
        rejected_candidates.append(_make_case(
            "borderline_rejected",
            1.0 / (1.0 + deficit),
            gt_by_key[key],
            decisions_by_gt[key],
            local_by_key.get(key),
            class_names,
            decision,
        ))
    selected_rejected = select_diverse(
        rejected_candidates, cases_per_group)
    if len(selected_rejected) < cases_per_group:
        raise ValueError("not enough cases for borderline_rejected")
    chosen.extend(selected_rejected)

    candidates = _candidate_sets(local_rows)[
        "single_gt_low_overlap_candidates"]
    for key in sorted(candidates, key=lambda value: (
            -local_by_key[value]["CollectiveGain_i"], value[0], value[1])):
        gt = gt_by_key[key]
        chosen.append(_make_case(
            "low_overlap_candidates",
            local_by_key[key]["CollectiveGain_i"],
            gt,
            decisions_by_gt.get(key, []),
            local_by_key[key],
            class_names,
        ))

    ranked = []
    for group in AUDIT_GROUPS:
        values = [row for row in chosen if row["category"] == group]
        values.sort(key=lambda row: (
            -row["selection_score"], row["image_id"],
            row["class_id"], row["gt_id"], row["focus_pred_id"] or ""))
        for rank, row in enumerate(values, 1):
            row = dict(row)
            row["rank"] = rank
            row["case_id"] = f"{group}_{rank:02d}"
            ranked.append(row)
    return ranked


def _render_fragment_view(
        source,
        crop,
        slot_size,
        gts,
        predictions,
        ignore_rects,
        selected_gt_id,
        focus_pred_id,
        decisions_by_pred,
        *,
        label_owned):
    from PIL import Image, ImageDraw

    crop = _fit_crop(crop, source.width, source.height)
    cut = source.crop(tuple(int(round(value)) for value in crop))
    scale = min(slot_size[0] / cut.width, slot_size[1] / cut.height)
    size = (
        max(1, round(cut.width * scale)),
        max(1, round(cut.height * scale)),
    )
    cut = cut.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", slot_size, (245, 245, 245))
    offset = (
        (slot_size[0] - size[0]) // 2,
        (slot_size[1] - size[1]) // 2,
    )
    canvas.paste(cut, offset)
    overlay = Image.new("RGBA", slot_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def mapped(rect):
        return tuple(
            offset[index % 2]
            + (rect[index] - crop[index % 2]) * scale
            for index in range(4)
        )

    for rect in ignore_rects:
        draw.rectangle(
            mapped(rect),
            fill=(100, 100, 100, 45),
            outline=(90, 90, 90, 140),
            width=2,
        )
    for prediction in predictions:
        decision = decisions_by_pred.get(prediction["id"])
        role = decision["final_status"] if decision else "unclassified"
        color = ROLE_COLORS[role]
        focus = prediction["id"] == focus_pred_id
        owned = decision and decision["owner_gt_id"] == selected_gt_id
        width = 7 if focus else (4 if owned else 2)
        rectangle = mapped(prediction["xyxy"])
        draw.rectangle(rectangle, outline=color, width=width)
        if label_owned and (owned or focus):
            x, y, _, _ = rectangle
            label = (
                f"{ROLE_ABBREVIATIONS[role]} "
                f"{prediction['id'][-4:]} {prediction['score']:.2f}"
            )
            draw.text(
                (x + 2, max(0, y - 19)),
                label,
                fill=color,
                font=_font(15, True),
                stroke_width=2,
                stroke_fill=(255, 255, 255, 235),
            )
    for gt in gts:
        selected = gt["id"] == selected_gt_id
        draw.rectangle(
            mapped(gt["xyxy"]),
            outline=(0, 105, 70, 255) if selected else (0, 170, 120, 210),
            width=6 if selected else 2,
        )
    return Image.alpha_composite(
        canvas.convert("RGBA"), overlay).convert("RGB")


def _format_optional(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def render_audit_case(
        case,
        image_record,
        raw_record,
        image_path,
        decisions_by_pred,
        output,
        confidence_threshold):
    from PIL import Image, ImageDraw

    source = Image.open(image_path).convert("RGB")
    if source.size != (image_record["width"], image_record["height"]):
        raise ValueError(f"image dimensions differ from cache: {image_path}")
    class_id = case["class_id"]
    gts = [
        row for row in image_record["gts"] if row["class_id"] == class_id
    ]
    selected_gt = next(row for row in gts if row["id"] == case["gt_id"])
    predictions = [
        row for row in image_record["predictions"]
        if row["class_id"] == class_id
        and row["score"] >= confidence_threshold
    ]
    ignore_rects = [
        row["xyxy"] for row in raw_record["annotations"]
        if row["category"] == 0
    ]
    canvas = Image.new("RGB", (1660, 960), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{case['case_id']} | {case['image_id']} | "
        f"{case['class_name']} | {case['gt_id']}"
    )
    draw.text((30, 18), title, fill=(25, 25, 25), font=_font(26, True))
    focus = case["focus_pred_id"] or "whole GT group"
    draw.text(
        (30, 52),
        f"focus={focus} | category={case['category']}",
        fill=(45, 45, 45),
        font=_font(18),
    )
    crop = detail_crop(
        selected_gt, predictions, source.width, source.height)
    panels = [
        (_render_fragment_view(
            source,
            (0, 0, source.width, source.height),
            (790, 650),
            gts,
            predictions,
            ignore_rects,
            selected_gt["id"],
            case["focus_pred_id"],
            decisions_by_pred,
            label_owned=False,
        ), "Full image-class context"),
        (_render_fragment_view(
            source,
            crop,
            (790, 650),
            gts,
            predictions,
            ignore_rects,
            selected_gt["id"],
            case["focus_pred_id"],
            decisions_by_pred,
            label_owned=True,
        ), "Selected GT detail; labels show final roles"),
    ]
    for column, (panel, panel_title) in enumerate(panels):
        x = 30 + column * 810
        canvas.paste(panel, (x, 110))
        draw.text((x, 82), panel_title, fill=(45, 45, 45), font=_font(18, True))
    draw.text(
        (30, 780),
        "Roles: A=anchor TP | S=support | G=group support | "
        "R=rejected FP | F=failed group FP | O=orphan FP | selected GT=dark green",
        fill=(45, 45, 45),
        font=_font(16),
    )
    draw.text(
        (30, 815),
        f"GT result: {case['detection_type']} | roles={case['owned_prediction_roles']} | "
        f"group IoU={_format_optional(case['group_iou'])} | "
        f"group precision={_format_optional(case['group_precision'])}",
        fill=(25, 25, 25),
        font=_font(18),
    )
    draw.text(
        (30, 850),
        f"Focus: status={case['focus_status'] or 'NA'} | "
        f"score={_format_optional(case['focus_score'])} | "
        f"owner IoU={_format_optional(case['owner_iou'])} | "
        f"UniqueCoverage={_format_optional(case['marginal_unique_coverage'])} | "
        f"FragmentPrecision={_format_optional(case['fragment_precision'])} | "
        f"GT hits={case['source_gt_hit_count'] if case['source_gt_hit_count'] is not None else 'NA'}",
        fill=(25, 25, 25),
        font=_font(18),
    )
    draw.text(
        (30, 890),
        "Human check: valid complement, duplicate/jitter, neighbor borrowing, "
        "background grazing, or annotation/scale ambiguity?",
        fill=(120, 35, 35),
        font=_font(18, True),
    )
    canvas.save(output, format="PNG", optimize=True)


def _report_text(
        fragment_run,
        source_run,
        cases,
        pool_counts,
        role_counts,
        reference):
    selected = Counter(row["category"] for row in cases)
    group_lines = "\n".join(
        f"| {group} | {pool_counts[group]} | {selected[group]} |"
        for group in AUDIT_GROUPS
    )
    return f"""# 第七轮：Fragment-aware 人工审计包

本轮只读取第六轮结果、已有预测缓存、GT 和原图路径，没有训练或模型推理。

来源 Fragment 运行：`{fragment_run}`  
来源预测缓存：`{source_run}`

## 为什么 VC 仍未冻结

第六轮已经证明公式可以执行，并通过八个受控反例；但参考点的变化主要是把 {reference['neutral_support_pred_count']} 个预测从 FP 中移出，只新增 {reference['fragment_group_TP']} 个 group TP。公式是否正确取决于这些 support 在真实图里是不是有效补充，而不能由分数提高本身决定。若 support 实际属于相邻实例、重复抖动或背景擦边，直接定稿会把错误行为写进 VC。

当前数据也只有一个模型、一个 seed 和 50 张图。UniqueCoverage 与 FragmentPrecision 的门槛若按这一份数据的最高 F1 选择，会产生明显的单模型过拟合。人工审计用于先检查语义方向；之后还需要跨模型、seed 和完整验证集检验排序稳定性。

## 是否需要重新训练

当前阶段不需要。缓存已经保存了原图坐标系中的 GT、类别、预测框、置信度、忽略区域和原图路径。当前 VC 候选只依赖这些量，因此可以反复重算公式、改门槛、做匹配和绘图。

重新训练只在两个更晚的目标下需要：一是把 VC 变成训练损失或模型选择信号，验证它能否改善模型；二是需要新的模型/seed 来检验泛化。扩展已有权重到更多图像只需要重新推理，不等于重新训练。

## 审计队列

参考配置：`theta={reference['confidence_threshold']}`、`UniqueCoverage={reference['unique_coverage_threshold']}`、`FragmentPrecision={reference['fragment_precision_threshold']}`、`group IoU={reference['group_iou_threshold']}`。

全部预测角色：`{json.dumps(dict(role_counts), ensure_ascii=False, sort_keys=True)}`。

| 组 | 候选池 | 已选案例 |
|---|---:|---:|
{group_lines}

`group_tp` 包含全部 group TP；`low_overlap_candidates` 包含第三轮全部 18 个候选。其余四组使用确定性、跨图像/类别优先的抽样。

## 看图方法

每张面板左侧是整图同类上下文，右侧是选中 GT 的放大图。深绿色是当前 GT；预测角色为：A anchor、S support、G group support、R rejected FP、F failed-group FP、O orphan FP。右下角显示 focus 预测的置信度、owner IoU、UniqueCoverage、FragmentPrecision 和相交 GT 数。

请在 `review_template.csv` 的 `human_label` 中填写以下之一：

~~~text
valid_complement
duplicate_or_jitter
neighbor_instance_borrowing
background_grazing
annotation_or_scale_ambiguity
uncertain
~~~

`keep_as_support` 填 `yes`、`no` 或 `uncertain`；`confidence` 填 `high`、`medium` 或 `low`。Group TP 评价整组；其他组重点评价 focus 预测。需要放大时打开 `cases/` 中的单张 1660×960 PNG，联系图只用于快速浏览。

## 输出

- `audit_queue.csv`：全部机器选择依据与 Fragment 数值。
- `review_template.csv`：待填写的人工标签表。
- `cases/<group>/`：可放大的逐案例面板。
- `contact_sheets/<group>.png`：每组总览。
- `analysis_manifest.json` 与 `analysis_complete.json`：来源哈希、参数、计数和输出哈希。

## 完成本轮的判断标准

先填完全部 group TP，再审查 multi-GT support 和门槛边界案例。只有当 group TP 大多确实代表同一目标的互补片段，并且 support 中的相邻实例借用率可接受时，才考虑冻结责任、门槛和计数规则。否则应根据错误类型修改规则，再用同一缓存重算，不需要训练模型。
"""


def analyze_fragment_audit(
        fragment_run,
        *,
        cases_per_group=6,
        output=None):
    fragment_run, fragment_manifest, fragment_complete, source_run = (
        validate_fragment_run(fragment_run))
    source_run, document, raw, _, source_complete, _ = validate_source(
        source_run)
    reference = fragment_complete["reference_summary"]
    confidence = reference["confidence_threshold"]
    unique_threshold = reference["unique_coverage_threshold"]
    precision_threshold = reference["fragment_precision_threshold"]
    summary, gt_rows, decisions = evaluate_fragment_aware(
        document,
        confidence,
        matching_iou=reference["matching_iou"],
        unique_coverage_threshold=unique_threshold,
        fragment_precision_threshold=precision_threshold,
        group_iou_threshold=reference["group_iou_threshold"],
    )
    for key in (
            "n_gt", "n_pred", "TP", "fragment_group_TP",
            "neutral_support_pred_count", "FP", "FN", "F1"):
        if summary[key] != reference[key]:
            raise ValueError(f"recomputed fragment summary differs at {key}")
    class_names = document["metadata"]["class_names"]
    _, evaluated_images, evaluated_summary = evaluate_document(
        document,
        theta=confidence,
        tau=source_complete["summary"]["tau"],
    )
    if evaluated_summary["n_pred"] != reference["n_pred"]:
        raise ValueError("source prediction count differs from fragment run")
    local_rows = local_gt_diagnostics(
        evaluated_images, document, class_names, confidence)
    cases = select_audit_cases(
        gt_rows,
        decisions,
        local_rows,
        class_names,
        unique_coverage_threshold=unique_threshold,
        fragment_precision_threshold=precision_threshold,
        cases_per_group=cases_per_group,
    )

    role_counts = Counter(row["final_status"] for row in decisions)
    support_rows = [
        row for row in decisions if row["final_status"] == "support"]
    rejected_rows = [
        row for row in decisions
        if row["final_status"] == "rejected_fp"
        and row["owner_gt_id"] is not None
        and row["fragment_precision"] > 0.0
    ]
    candidate_sets = _candidate_sets(local_rows)
    pool_counts = dict(
        group_tp=sum(row["detection_type"] == "fragment_group" for row in gt_rows),
        support_multigt=sum(
            row["source_gt_hit_count"] >= 2 for row in support_rows),
        support_single_gt=sum(
            row["source_gt_hit_count"] == 1 for row in support_rows),
        borderline_support=len(support_rows),
        borderline_rejected=len(rejected_rows),
        low_overlap_candidates=len(
            candidate_sets["single_gt_low_overlap_candidates"]),
    )

    output = (
        Path(output).resolve() if output else
        PROJECT_ROOT / "results" / "region_eval" /
        ("fragment_audit_" +
         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    archive = (PROJECT_ROOT / "bin" / "old").resolve()
    if output.exists() or output == archive or output.is_relative_to(archive):
        raise ValueError("output must be a new directory outside bin/old")
    output.mkdir(parents=True)
    _write_json(output / "analysis_manifest.json", dict(
        status="started",
        analysis_version=ANALYSIS_VERSION,
        fragment_run=str(fragment_run),
        fragment_complete_sha256=_file_hash(
            fragment_run / "analysis_complete.json"),
        source_run=str(source_run),
        source_filtered_sha256=_file_hash(
            source_run / "filtered_predictions.json"),
        parameters=dict(
            cases_per_group=cases_per_group,
            confidence_threshold=confidence,
            unique_coverage_threshold=unique_threshold,
            fragment_precision_threshold=precision_threshold,
            matching_iou=reference["matching_iou"],
            group_iou_threshold=reference["group_iou_threshold"],
        ),
        source_code_sha256={
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"):
            _file_hash(path)
            for path in (
                PROJECT_ROOT / "region_eval" / "fragment_audit.py",
                PROJECT_ROOT / "run_region_fragment_audit.py",
            )
        },
    ))

    document_by_id = {
        row["image_id"]: row for row in document["images"]}
    raw_by_id = {row["image_id"]: row for row in raw["images"]}
    manifest_by_id = {
        row["image_id"]: row
        for row in document["metadata"]["subset"]["image_manifest"]
    }
    decisions_by_image_class = defaultdict(dict)
    for row in decisions:
        decisions_by_image_class[(row["image_id"], row["class_id"])][
            row["pred_id"]] = row
    case_root = output / "cases"
    contact_root = output / "contact_sheets"
    case_root.mkdir()
    contact_root.mkdir()
    for group in AUDIT_GROUPS:
        group_dir = case_root / group
        group_dir.mkdir()
        files = []
        for case in [row for row in cases if row["category"] == group]:
            filename = (
                f"{case['case_id']}_{case['image_id']}_"
                f"{case['class_name']}_{case['gt_id']}.png"
            )
            destination = group_dir / filename
            image_path = Path(
                manifest_by_id[case["image_id"]]["image_file"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            render_audit_case(
                case,
                document_by_id[case["image_id"]],
                raw_by_id[case["image_id"]],
                image_path,
                decisions_by_image_class[
                    (case["image_id"], case["class_id"])],
                destination,
                confidence,
            )
            case["panel_file"] = str(
                destination.relative_to(output)).replace("\\", "/")
            files.append(destination)
        contact_sheet(group, files, contact_root / f"{group}.png")

    case_fields = [
        "case_id", "category", "rank", "panel_file", "selection_score",
        "image_id", "class_id", "class_name", "gt_id", "focus_pred_id",
        "focus_status", "focus_score", "source_gt_hit_count", "owner_iou",
        "fragment_precision", "marginal_unique_coverage", "detection_type",
        "detected", "assigned_pred_count", "accepted_pred_count",
        "support_pred_count", "rejected_pred_count", "group_coverage",
        "group_precision", "group_dice", "group_iou",
        "owned_prediction_roles", "owned_prediction_ids",
        "original_positive_CG_with_multigt_candidate",
        "single_gt_low_overlap_candidate", "BestIoGT_i",
        "CollectiveGain_i",
    ]
    _write_csv(output / "audit_queue.csv", cases, case_fields)
    review_rows = [dict(
        case_id=row["case_id"],
        category=row["category"],
        image_id=row["image_id"],
        class_name=row["class_name"],
        gt_id=row["gt_id"],
        focus_pred_id=row["focus_pred_id"],
        human_label="",
        keep_as_support="",
        confidence="",
        notes="",
    ) for row in cases]
    review_fields = [
        "case_id", "category", "image_id", "class_name", "gt_id",
        "focus_pred_id", "human_label", "keep_as_support", "confidence",
        "notes",
    ]
    _write_csv(output / "review_template.csv", review_rows, review_fields)
    (output / "README.md").write_text(
        _report_text(
            fragment_run,
            source_run,
            cases,
            pool_counts,
            role_counts,
            reference,
        ),
        encoding="utf-8",
        newline="\n",
    )
    selected_counts = {
        group: sum(row["category"] == group for row in cases)
        for group in AUDIT_GROUPS
    }
    _write_json(output / "analysis_complete.json", dict(
        status="complete",
        analysis_version=ANALYSIS_VERSION,
        fragment_run=str(fragment_run),
        source_run=str(source_run),
        counts=dict(
            selected_cases=len(cases),
            groups=selected_counts,
            source_images=len(document["images"]),
            source_gt=reference["n_gt"],
            source_predictions=reference["n_pred"],
            role_counts=dict(role_counts),
            candidate_pool_counts=pool_counts,
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
    print(json.dumps(dict(
        output=str(output),
        selected_cases=len(cases),
        groups=selected_counts,
        source_images=len(document["images"]),
        source_gt=reference["n_gt"],
        source_predictions=reference["n_pred"],
    ), ensure_ascii=False), flush=True)
    print(
        "SUCCESS: fragment audit package complete. "
        "No inference or training performed.",
        flush=True,
    )
    return output
