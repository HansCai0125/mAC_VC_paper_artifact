"""Offline case ranking and visualization for completed VisDrone diagnostic runs."""

from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
import statistics

from .geometry import area, intersection, union_area
from .offline import PROJECT_ROOT, evaluate_document


ANALYSIS_VERSION = "region-case-analysis-0.1.0"
SIZE_BINS = (("small", 0.0, 32.0 ** 2),
             ("medium", 32.0 ** 2, 96.0 ** 2),
             ("large", 96.0 ** 2, float("inf")))
CASE_GROUPS = ("high_cg_high_class_quality", "high_cg_class_tension",
               "single_gt_low_overlap_cg", "coverage_precision_gap",
               "small_missed", "threshold_tradeoff")


def harmonic(a, b):
    return 0.0 if a <= 0 or b <= 0 else 2.0 * a * b / (a + b)


def pair_vc(images):
    """Exploratory only: active-pair macro of mean_i H(AC_i, pair precision)."""
    values = []
    for image in images:
        gt_by_class = defaultdict(list)
        for row in image.per_gt:
            gt_by_class[row["class_id"]].append(row["AC"])
        for pair in image.per_class:
            coverage = gt_by_class.get(pair["class_id"], [])
            precision = pair["SpatialPrecision"] or 0.0
            values.append(statistics.fmean(harmonic(value, precision) for value in coverage)
                          if coverage else 0.0)
    return statistics.fmean(values) if values else None


def select_diverse(candidates, count):
    """Deterministically prefer different classes, then fill with different pairs."""
    ordered = sorted(candidates, key=lambda row: (-row["selection_score"], row["image_id"],
                                                   row["class_id"], row.get("gt_id") or ""))
    chosen, used_pairs, used_classes = [], set(), set()
    for prefer_new_class in (True, False):
        for row in ordered:
            pair = (row["image_id"], row["class_id"])
            if pair in used_pairs or row in chosen:
                continue
            if prefer_new_class and row["class_id"] in used_classes:
                continue
            chosen.append(dict(row))
            used_pairs.add(pair)
            used_classes.add(row["class_id"])
            if len(chosen) == count:
                return chosen
    return chosen


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"nonstandard JSON number: {value}")))


def _write_json(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as handle:
        handle.write(data)
    return data


def _write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if row.get(key) is None else row.get(key, "") for key in fields})


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(source_run):
    source_run = Path(source_run).resolve()
    required = ["run_complete.json", "raw_predictions.json", "filtered_predictions.json",
                "preprocessing_audit.json", "report/metadata.json"]
    missing = [name for name in required if not (source_run / name).is_file()]
    if missing:
        raise ValueError(f"incomplete source run; missing {missing}")
    if (source_run / "run_failed.json").exists():
        raise ValueError("source run contains run_failed.json")
    complete = _read_json(source_run / "run_complete.json")
    metadata = _read_json(source_run / "report/metadata.json")
    filtered_bytes = (source_run / "filtered_predictions.json").read_bytes()
    if complete.get("status") != "complete" or metadata.get("status") != "complete":
        raise ValueError("source run is not marked complete")
    if metadata.get("input_sha256") != sha256(filtered_bytes).hexdigest():
        raise ValueError("filtered prediction cache hash differs from report metadata")
    document = json.loads(filtered_bytes.decode("utf-8"))
    raw = _read_json(source_run / "raw_predictions.json")
    audit = _read_json(source_run / "preprocessing_audit.json")
    image_ids = [row["image_id"] for row in document.get("images", [])]
    if (len(image_ids) != len(set(image_ids)) or image_ids != [row["image_id"] for row in raw.get("images", [])]
            or image_ids != [row["image_id"] for row in audit]):
        raise ValueError("filtered, raw, and audit image order/IDs differ")
    names = document.get("metadata", {}).get("class_names")
    if not isinstance(names, list) or len(names) != 10:
        raise ValueError("ten class names are required")
    return source_run, document, raw, audit, complete, metadata


def _indexes(images):
    gt = {}
    by_pair = defaultdict(list)
    pairs = {}
    for image in images:
        for row in image.per_gt:
            key = (row["image_id"], row["gt_id"])
            gt[key] = row
            by_pair[(row["image_id"], row["class_id"])].append(row)
        for row in image.per_class:
            pairs[(row["image_id"], row["class_id"])] = row
    return gt, by_pair, pairs


def _pair_context(key, by_pair, pairs):
    rows = by_pair.get(key, [])
    pair = pairs[key]
    return dict(
        pair_mean_AC=statistics.fmean(row["AC"] for row in rows) if rows else 0.0,
        pair_mean_BestIoGT=statistics.fmean(row["BestIoGT_i"] for row in rows) if rows else 0.0,
        pair_mean_CollectiveGain=statistics.fmean(row["CollectiveGain_i"] for row in rows) if rows else 0.0,
        pair_Dice=pair["ClassUnionDice"], pair_IoU=pair["ClassUnionIoU"],
        pair_SpatialPrecision=pair["SpatialPrecision"] or 0.0,
        pair_n_gt=pair["n_gt"], pair_n_pred=pair["n_pred"],
        pair_orphan_pred_count=pair["orphan_pred_count"],
        pair_multigt_pred_count=pair["multigt_pred_count"])


def _case_from_gt(category, score, gt, context, class_names):
    return dict(category=category, selection_score=score, image_id=gt["image_id"],
                class_id=gt["class_id"], class_name=class_names[gt["class_id"]],
                gt_id=gt["gt_id"], gt_area=gt["gt_area"], AC=gt["AC"],
                BestIoGT_i=gt["BestIoGT_i"], CollectiveGain_i=gt["CollectiveGain_i"],
                intersecting_pred_count=gt["intersecting_pred_count"],
                best_pred_id=gt["best_pred_id"], **context)


def rank_cases(primary_images, low_images, class_names, count):
    gt, by_pair, pairs = _indexes(primary_images)
    low_gt, low_by_pair, low_pairs = _indexes(low_images)
    candidates = {name: [] for name in CASE_GROUPS}
    for row in gt.values():
        key = (row["image_id"], row["class_id"])
        context = _pair_context(key, by_pair, pairs)
        cg = row["CollectiveGain_i"]
        if cg > 0 and row["AC"] >= 0.5 and context["pair_Dice"] >= 0.7 and context["pair_SpatialPrecision"] >= 0.7:
            score = cg * (context["pair_Dice"] + context["pair_SpatialPrecision"]) / 2.0
            candidates["high_cg_high_class_quality"].append(
                _case_from_gt("high_cg_high_class_quality", score, row, context, class_names))
        if cg > 0.01 and (context["pair_Dice"] < 0.7 or context["pair_SpatialPrecision"] < 0.7):
            score = cg * (2.0 - context["pair_Dice"] - context["pair_SpatialPrecision"])
            candidates["high_cg_class_tension"].append(
                _case_from_gt("high_cg_class_tension", score, row, context, class_names))
        if row["gt_area"] < 32.0 ** 2 and row["AC"] < 0.25:
            score = (1.0 - row["AC"]) * (1.0 + (32.0 ** 2 - row["gt_area"]) / 32.0 ** 2)
            candidates["small_missed"].append(
                _case_from_gt("small_missed", score, row, context, class_names))
    for key, pair in pairs.items():
        rows = by_pair.get(key, [])
        if not rows:
            continue
        context = _pair_context(key, by_pair, pairs)
        if context["pair_mean_AC"] >= 0.5 and context["pair_mean_AC"] > context["pair_Dice"]:
            selected = max(rows, key=lambda row: (row["AC"], row["CollectiveGain_i"], row["gt_id"]))
            score = context["pair_mean_AC"] - context["pair_Dice"]
            candidates["coverage_precision_gap"].append(
                _case_from_gt("coverage_precision_gap", score, selected, context, class_names))
        if key in low_pairs and key in low_by_pair:
            low_context = _pair_context(key, low_by_pair, low_pairs)
            delta_ac = low_context["pair_mean_AC"] - context["pair_mean_AC"]
            delta_dice = low_context["pair_Dice"] - context["pair_Dice"]
            if delta_ac > 0.05:
                selected = max(rows, key=lambda row: (
                    low_gt.get((row["image_id"], row["gt_id"]), row)["AC"] - row["AC"], row["gt_id"]))
                score = delta_ac + max(0.0, -delta_dice) + min(
                    0.2, max(0, low_context["pair_orphan_pred_count"] - context["pair_orphan_pred_count"]) / 20.0)
                case = _case_from_gt("threshold_tradeoff", score, selected, context, class_names)
                case.update(low_pair_mean_AC=low_context["pair_mean_AC"],
                            low_pair_Dice=low_context["pair_Dice"],
                            low_pair_SpatialPrecision=low_context["pair_SpatialPrecision"],
                            low_pair_n_pred=low_context["pair_n_pred"],
                            low_pair_orphan_pred_count=low_context["pair_orphan_pred_count"],
                            delta_AC_low_minus_primary=delta_ac,
                            delta_Dice_low_minus_primary=delta_dice)
                candidates["threshold_tradeoff"].append(case)
    selected = []
    for group in CASE_GROUPS:
        rows = select_diverse(candidates[group], count)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
            row["case_id"] = f"{group}_{rank:02d}"
            selected.append(row)
    return selected


def rank_single_gt_low_overlap_cases(primary_images, local_rows, class_names, count):
    """Select potential within-GT complementary cases after removing direct neighbor pooling."""
    _, by_pair, pairs = _indexes(primary_images)
    candidates = []
    for row in local_rows:
        pairwise = row["max_contributing_pairwise_iou"]
        if (row["CollectiveGain_i"] <= 1e-9 or row["contributing_pred_count"] < 2
                or row["contributing_multigt_count"] != 0 or pairwise is None or pairwise >= 0.3):
            continue
        key = (row["image_id"], row["class_id"])
        context = _pair_context(key, by_pair, pairs)
        score = row["CollectiveGain_i"] * row["selected_gt_local_dice"] * (1.0 - pairwise)
        candidates.append(_case_from_gt(
            "single_gt_low_overlap_cg", score, row, context, class_names))
    selected = select_diverse(candidates, count)
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
        row["case_id"] = f"single_gt_low_overlap_cg_{rank:02d}"
    return selected


def class_summary(images, class_names):
    _, by_pair, pairs = _indexes(images)
    rows = []
    for class_id, class_name in enumerate(class_names):
        gt_rows = [row for key, values in by_pair.items() if key[1] == class_id for row in values]
        pair_rows = [row for key, row in pairs.items() if key[1] == class_id]
        rows.append(dict(
            class_id=class_id, class_name=class_name, n_gt=len(gt_rows), n_active_pairs=len(pair_rows),
            mAC=statistics.fmean(row["AC"] for row in gt_rows) if gt_rows else None,
            BestIoGT=statistics.fmean(row["BestIoGT_i"] for row in gt_rows) if gt_rows else None,
            CollectiveGain=statistics.fmean(row["CollectiveGain_i"] for row in gt_rows) if gt_rows else None,
            RecallACtau=statistics.fmean(int(row["coverage_hit"]) for row in gt_rows) if gt_rows else None,
            cg_positive_fraction=statistics.fmean(row["CollectiveGain_i"] > 1e-9 for row in gt_rows) if gt_rows else None,
            ClassUnionDice=statistics.fmean(row["ClassUnionDice"] for row in pair_rows) if pair_rows else None,
            ClassUnionIoU=statistics.fmean(row["ClassUnionIoU"] for row in pair_rows) if pair_rows else None))
    return rows


def size_summary(images):
    gt_rows = [row for image in images for row in image.per_gt]
    rows = []
    for name, low, high in SIZE_BINS:
        values = [row for row in gt_rows if low <= row["gt_area"] < high]
        rows.append(dict(size=name, lower_area=low, upper_area=None if high == float("inf") else high,
                         n_gt=len(values), fraction=len(values) / len(gt_rows) if gt_rows else None,
                         mAC=statistics.fmean(row["AC"] for row in values) if values else None,
                         BestIoGT=statistics.fmean(row["BestIoGT_i"] for row in values) if values else None,
                         CollectiveGain=statistics.fmean(row["CollectiveGain_i"] for row in values) if values else None,
                         RecallACtau=statistics.fmean(int(row["coverage_hit"]) for row in values) if values else None))
    return rows


def enrich_case_geometry(case, image_record, theta):
    """Add local-vs-class support evidence for the selected GT."""
    class_id = case["class_id"]
    gts = [row for row in image_record["gts"] if row["class_id"] == class_id]
    selected = next(row for row in gts if row["id"] == case["gt_id"])
    predictions = [row for row in image_record["predictions"]
                   if row["class_id"] == class_id and row["score"] >= theta]
    contributing = [row for row in predictions
                    if intersection(tuple(selected["xyxy"]), tuple(row["xyxy"])) is not None]
    pred_rects = [tuple(row["xyxy"]) for row in contributing]
    selected_intersections = [value for row in contributing
                              if (value := intersection(tuple(selected["xyxy"]),
                                                        tuple(row["xyxy"]))) is not None]
    class_intersections = [value for row in contributing for gt in gts
                           if (value := intersection(tuple(gt["xyxy"]),
                                                     tuple(row["xyxy"]))) is not None]
    pred_union = union_area(pred_rects)
    selected_intersection = union_area(selected_intersections)
    class_intersection = union_area(class_intersections)
    local_precision = selected_intersection / pred_union if pred_union else 0.0
    class_precision = min(1.0, class_intersection / pred_union) if pred_union else 0.0
    pairwise_iou = []
    for index, first in enumerate(pred_rects):
        for second in pred_rects[index + 1:]:
            overlap = intersection(first, second)
            overlap_area = area(overlap) if overlap is not None else 0.0
            denominator = area(first) + area(second) - overlap_area
            pairwise_iou.append(overlap_area / denominator if denominator else 0.0)
    hit_counts = [sum(intersection(tuple(row["xyxy"]), tuple(gt["xyxy"])) is not None
                      for gt in gts) for row in contributing]
    case.update(
        contributing_pred_count=len(contributing),
        contributing_multigt_count=sum(value >= 2 for value in hit_counts),
        contributing_single_gt_count=sum(value == 1 for value in hit_counts),
        contributing_pred_ids=";".join(row["id"] for row in contributing),
        contributing_scores=";".join(f"{row['score']:.6f}" for row in contributing),
        contributing_union_area=pred_union,
        selected_gt_local_precision=local_precision,
        contributing_class_precision=class_precision,
        neighboring_class_credit=max(0.0, class_precision - local_precision),
        selected_gt_local_dice=harmonic(case["AC"], local_precision),
        max_contributing_pairwise_iou=max(pairwise_iou) if pairwise_iou else None,
        mean_contributing_pairwise_iou=statistics.fmean(pairwise_iou) if pairwise_iou else None)
    return case


def local_gt_diagnostics(images, document, class_names, theta):
    """Compute per-GT local union support for every GT at one threshold."""
    image_records = {row["image_id"]: row for row in document["images"]}
    rows = []
    for image in images:
        for diagnostic in image.per_gt:
            row = dict(diagnostic)
            row["class_name"] = class_names[row["class_id"]]
            enrich_case_geometry(row, image_records[row["image_id"]], theta)
            rows.append(row)
    return sorted(rows, key=lambda row: (row["image_id"], row["class_id"], row["gt_id"]))


def local_support_summary(rows, epsilon=1e-9):
    positive = [row for row in rows if row["CollectiveGain_i"] > epsilon]
    def fraction(count):
        return count / len(positive) if positive else None
    any_multigt = sum(row["contributing_multigt_count"] > 0 for row in positive)
    neighbor = sum(row["neighboring_class_credit"] > 0.1 for row in positive)
    no_multigt = [row for row in positive if row["contributing_multigt_count"] == 0]
    duplicate_like = sum((row["max_contributing_pairwise_iou"] or 0.0) > 0.5
                         for row in no_multigt)
    low_overlap = sum((row["max_contributing_pairwise_iou"] or 0.0) < 0.3
                      for row in no_multigt)
    return dict(
        n_gt=len(rows),
        mean_selected_gt_local_precision=statistics.fmean(
            row["selected_gt_local_precision"] for row in rows) if rows else None,
        mean_selected_gt_local_union_dice=statistics.fmean(
            row["selected_gt_local_dice"] for row in rows) if rows else None,
        n_high_AC_ge_075=sum(row["AC"] >= 0.75 for row in rows),
        n_high_AC_ge_075_low_local_dice_lt_05=sum(
            row["AC"] >= 0.75 and row["selected_gt_local_dice"] < 0.5 for row in rows),
        n_positive_CG=len(positive),
        positive_CG_any_multigt_contributor=any_multigt,
        positive_CG_any_multigt_fraction=fraction(any_multigt),
        positive_CG_neighbor_credit_gt_01=neighbor,
        positive_CG_neighbor_credit_fraction=fraction(neighbor),
        positive_CG_no_multigt_contributor=len(no_multigt),
        positive_CG_no_multigt_fraction=fraction(len(no_multigt)),
        positive_CG_no_multigt_pairwise_iou_gt_05=duplicate_like,
        positive_CG_no_multigt_pairwise_iou_gt_05_fraction=fraction(duplicate_like),
        positive_CG_no_multigt_pairwise_iou_lt_03=low_overlap,
        positive_CG_no_multigt_pairwise_iou_lt_03_fraction=fraction(low_overlap),
        positive_CG_mean_neighbor_credit=statistics.fmean(
            row["neighboring_class_credit"] for row in positive) if positive else None,
        positive_CG_mean_local_union_dice=statistics.fmean(
            row["selected_gt_local_dice"] for row in positive) if positive else None,
        heuristic_note=("Multi-GT and pairwise-IoU partitions are diagnostic flags, not "
                        "semantic labels or proof of useful/duplicate fragmentation."))


def _font(size, bold=False):
    from PIL import ImageFont
    paths = ([Path(r"C:\Windows\Fonts\arialbd.ttf"), Path(r"C:\Windows\Fonts\segoeuib.ttf")]
             if bold else [Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\segoeui.ttf")])
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


COLORS = dict(gt=(0, 158, 115, 255), selected_gt=(0, 110, 80, 255),
              best=(0, 114, 178, 255), contributing=(230, 159, 0, 255),
              orphan=(213, 94, 0, 255), other=(204, 121, 167, 210),
              ignore=(100, 100, 100, 55), ignore_outline=(90, 90, 90, 150))


def _fit_crop(crop, width, height):
    left, top, right, bottom = crop
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(width), right), min(float(height), bottom)
    if right <= left or bottom <= top:
        return (0.0, 0.0, float(width), float(height))
    return (left, top, right, bottom)


def detail_crop(gt, predictions, width, height):
    rects = [gt["xyxy"]]
    for pred in predictions:
        if intersection(tuple(gt["xyxy"]), tuple(pred["xyxy"])) is not None:
            rects.append(pred["xyxy"])
    left, top = min(r[0] for r in rects), min(r[1] for r in rects)
    right, bottom = max(r[2] for r in rects), max(r[3] for r in rects)
    span = max(right - left, bottom - top, 120.0)
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    return _fit_crop((cx - span * 0.8, cy - span * 0.8, cx + span * 0.8, cy + span * 0.8),
                     width, height)


def _render_view(source, crop, slot_size, gts, predictions, ignore_rects, selected_gt_id, best_pred_id,
                 label_predictions):
    from PIL import Image, ImageDraw
    crop = _fit_crop(crop, source.width, source.height)
    cut = source.crop(tuple(int(round(v)) for v in crop))
    scale = min(slot_size[0] / cut.width, slot_size[1] / cut.height)
    size = (max(1, round(cut.width * scale)), max(1, round(cut.height * scale)))
    cut = cut.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", slot_size, (245, 245, 245))
    offset = ((slot_size[0] - size[0]) // 2, (slot_size[1] - size[1]) // 2)
    canvas.paste(cut, offset)
    overlay = Image.new("RGBA", slot_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def mapped(rect):
        return tuple(offset[i % 2] + (rect[i] - crop[i % 2]) * scale for i in range(4))

    for rect in ignore_rects:
        draw.rectangle(mapped(rect), fill=COLORS["ignore"], outline=COLORS["ignore_outline"], width=2)
    selected_gt = next((row for row in gts if row["id"] == selected_gt_id), None)
    gt_rects = [tuple(row["xyxy"]) for row in gts]
    for pred in predictions:
        rect = tuple(pred["xyxy"])
        gt_hit_count = sum(intersection(rect, gt_rect) is not None for gt_rect in gt_rects)
        orphan = gt_hit_count == 0
        contributes = selected_gt is not None and intersection(rect, tuple(selected_gt["xyxy"])) is not None
        if pred["id"] == best_pred_id:
            color, line_width = COLORS["best"], 5
        elif contributes:
            color, line_width = COLORS["contributing"], 4
        elif orphan:
            color, line_width = COLORS["orphan"], 3
        else:
            color, line_width = COLORS["other"], 2
        draw.rectangle(mapped(rect), outline=color, width=line_width)
        if label_predictions and (contributes or pred["id"] == best_pred_id):
            x, y, _, _ = mapped(rect)
            scope = "M" if gt_hit_count >= 2 else "S"
            draw.text((x + 2, max(0, y - 17)), f"{scope}{pred['id'][-3:]} {pred['score']:.2f}",
                      fill=color, font=_font(15, True), stroke_width=2, stroke_fill=(255, 255, 255, 230))
    for row in gts:
        selected = row["id"] == selected_gt_id
        draw.rectangle(mapped(row["xyxy"]), outline=COLORS["selected_gt" if selected else "gt"],
                       width=6 if selected else 3)
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def render_case(case, image_record, raw_record, image_path, output, primary_theta, low_theta):
    from PIL import Image, ImageDraw
    source = Image.open(image_path).convert("RGB")
    if source.size != (image_record["width"], image_record["height"]):
        raise ValueError(f"image dimensions differ from cache: {image_path}")
    class_id = case["class_id"]
    gts = [row for row in image_record["gts"] if row["class_id"] == class_id]
    selected_gt = next(row for row in gts if row["id"] == case["gt_id"])
    ignore_rects = [row["xyxy"] for row in raw_record["annotations"] if row["category"] == 0]
    primary = [row for row in image_record["predictions"]
               if row["class_id"] == class_id and row["score"] >= primary_theta]
    low = [row for row in image_record["predictions"]
           if row["class_id"] == class_id and row["score"] >= low_theta]
    canvas = Image.new("RGB", (1600, 900), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), f"{case['case_id']} | {case['image_id']} | {case['class_name']} | {case['gt_id']}",
              fill=(25, 25, 25), font=_font(28, True))
    if case["category"] == "threshold_tradeoff":
        views = [(low, f"theta={low_theta:.2f}: {len(low)} class predictions"),
                 (primary, f"theta={primary_theta:.2f}: {len(primary)} class predictions")]
        for column, (predictions, title) in enumerate(views):
            panel = _render_view(source, (0, 0, source.width, source.height), (755, 650), gts,
                                 predictions, ignore_rects, selected_gt["id"],
                                 case["best_pred_id"] if column else None, False)
            x = 30 + column * 785
            canvas.paste(panel, (x, 100))
            draw.text((x, 70), title, fill=(40, 40, 40), font=_font(20, True))
    else:
        crop = detail_crop(selected_gt, primary, source.width, source.height)
        panels = [(_render_view(source, (0, 0, source.width, source.height), (755, 650), gts,
                                primary, ignore_rects, selected_gt["id"], case["best_pred_id"], False),
                   "Full image-class context"),
                  (_render_view(source, crop, (755, 650), gts, primary, ignore_rects,
                                selected_gt["id"], case["best_pred_id"], True),
                   "Selected GT detail")]
        for column, (panel, title) in enumerate(panels):
            x = 30 + column * 785
            canvas.paste(panel, (x, 100))
            draw.text((x, 70), title, fill=(40, 40, 40), font=_font(20, True))
    metrics = (f"GT: AC={case['AC']:.3f}  BestIoGT={case['BestIoGT_i']:.3f}  "
               f"CG={case['CollectiveGain_i']:.3f}  area={case['gt_area']:.0f}  "
               f"intersecting predictions={case['intersecting_pred_count']}")
    pair = (f"Pair: mean AC={case['pair_mean_AC']:.3f}  Dice={case['pair_Dice']:.3f}  "
            f"spatial precision={case['pair_SpatialPrecision']:.3f}  GT={case['pair_n_gt']}  "
            f"pred={case['pair_n_pred']}  orphan={case['pair_orphan_pred_count']}  "
            f"multi-GT pred={case['pair_multigt_pred_count']}")
    local = (f"Selected support: local precision={case['selected_gt_local_precision']:.3f}  "
             f"class precision={case['contributing_class_precision']:.3f}  "
             f"neighbor credit={case['neighboring_class_credit']:.3f}  "
             f"local Dice={case['selected_gt_local_dice']:.3f}  "
             f"contributing multi-GT={case['contributing_multigt_count']}/{case['contributing_pred_count']}")
    draw.text((30, 775), metrics, fill=(25, 25, 25), font=_font(19))
    draw.text((30, 805), pair, fill=(25, 25, 25), font=_font(19))
    draw.text((30, 835), local, fill=(25, 25, 25), font=_font(17))
    draw.text((30, 865), "GT green (selected dark) | best pred blue | contributing orange | orphan red | M/S label = multi/single-GT scope | ignore gray",
              fill=(55, 55, 55), font=_font(16))
    canvas.save(output, format="PNG", optimize=True)


def contact_sheet(title, files, output):
    from PIL import Image, ImageDraw
    columns, thumb = 2, (760, 428)
    rows = (len(files) + columns - 1) // columns
    sheet = Image.new("RGB", (1580, 70 + rows * 448), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 20), title.replace("_", " ").title(), fill=(25, 25, 25), font=_font(28, True))
    for index, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb)
        x = 20 + (index % columns) * 780
        y = 65 + (index // columns) * 448
        sheet.paste(image, (x, y))
    sheet.save(output, format="PNG", optimize=True)


def overview_plot(threshold_rows, class_rows, size_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    theta = [row["theta"] for row in threshold_rows]
    for name in ("mAC", "BestIoGT", "ClassUnionDice", "ClassUnionIoU",
                 "mean_gt_local_union_dice"):
        axes[0, 0].plot(theta, [row[name] for row in threshold_rows], marker="o", label=name)
    axes[0, 0].set(title="Threshold sensitivity", xlabel="Confidence threshold", ylabel="Score")
    axes[0, 0].set_ylim(0, 0.65)
    axes[0, 0].grid(alpha=.25)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(theta, [row["n_pred"] for row in threshold_rows], marker="o", label="Predictions")
    axes[0, 1].plot(theta, [row["orphan_pred_count"] for row in threshold_rows], marker="s", label="Orphan predictions")
    axes[0, 1].set(title="Prediction volume", xlabel="Confidence threshold", ylabel="Count over 50 images")
    axes[0, 1].grid(alpha=.25)
    axes[0, 1].legend(frameon=False)
    x = np.arange(len(size_rows))
    width = .26
    for offset, name in zip((-width, 0, width), ("mAC", "BestIoGT", "RecallACtau")):
        axes[1, 0].bar(x + offset, [row[name] for row in size_rows], width, label=name)
    axes[1, 0].set(title="GT-scale diagnostic", xlabel="GT size (COCO area bins)", ylabel="Score",
                   xticks=x, xticklabels=[f"{row['size']}\n(n={row['n_gt']})" for row in size_rows])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(frameon=False)
    y = np.arange(len(class_rows))
    axes[1, 1].barh(y + .18, [row["mAC"] for row in class_rows], .34, label="mAC")
    axes[1, 1].barh(y - .18, [row["ClassUnionDice"] for row in class_rows], .34, label="ClassUnionDice")
    axes[1, 1].set(title="Class diagnostic", xlabel="Score", ylabel="Class",
                   yticks=y, yticklabels=[row["class_name"] for row in class_rows])
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].legend(frameon=False)
    figure.suptitle("VisDrone 50-image regional diagnostic", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _markdown(local_stats, threshold_rows, class_rows, size_rows, cases, source_run):
    primary = next(row for row in threshold_rows if row["theta"] == 0.25)
    low = next(row for row in threshold_rows if row["theta"] == 0.05)
    top_gap = next(row for row in cases if row["category"] == "coverage_precision_gap")
    top_good = next(row for row in cases if row["category"] == "high_cg_high_class_quality")
    lines = [
        "# 第三轮：真实案例诊断", "",
        f"来源运行：`{source_run}`。本轮只读取预测缓存，没有推理或训练。分析版本：`{ANALYSIS_VERSION}`。", "",
        "## 主要结果", "",
        f"在 theta=0.25 时，mAC={primary['mAC']:.6f}、BestIoGT={primary['BestIoGT']:.6f}、CollectiveGain={primary['CollectiveGain']:.6f}、Dice={primary['ClassUnionDice']:.6f}。",
        f"theta 从 0.25 降到 0.05 后，mAC 增加 {low['mAC'] - primary['mAC']:.6f}，orphan 框从 {primary['orphan_pred_count']} 增至 {low['orphan_pred_count']}，Dice 变化为 {low['ClassUnionDice'] - primary['ClassUnionDice']:.6f}。覆盖可以被低置信度候选明显抬高，但空间质量没有同步增加。",
        f"全部 {local_stats['n_positive_CG']} 个正 CG GT 中，{local_stats['positive_CG_any_multigt_contributor']} 个包含至少一个同时命中多个 GT 的预测，{local_stats['positive_CG_neighbor_credit_gt_01']} 个的相邻同类区域贡献超过 0.1。只有 {local_stats['positive_CG_no_multigt_contributor']} 个没有多 GT 贡献者；其中 {local_stats['positive_CG_no_multigt_pairwise_iou_gt_05']} 个预测间最大 IoU 超过 0.5，{local_stats['positive_CG_no_multigt_pairwise_iou_lt_03']} 个低于 0.3。后一组只是潜在单 GT 低重叠互补候选，仍需看图。", "",
        f"最大 AC-Dice 分歧案例为 `{top_gap['case_id']}`：{top_gap['image_id']} / {top_gap['class_name']}，pair mean AC={top_gap['pair_mean_AC']:.3f}，Dice={top_gap['pair_Dice']:.3f}，空间精度={top_gap['pair_SpatialPrecision']:.3f}。",
        f"高 CG + 高类别级空间质量组的首个案例为 `{top_good['case_id']}`：{top_good['image_id']} / {top_good['class_name']}，GT CG={top_good['CollectiveGain_i']:.3f}，pair Dice={top_good['pair_Dice']:.3f}。这些标签由数值规则选出，仍需看图确认，不能当作人工真值。", "",
        "## 如何看输出", "",
        "- region_gt_local.csv 与 local_support_summary.json：全部 GT 的当前目标局部精度、类别精度、相邻贡献、局部 UnionDice 和多 GT/框间重叠诊断。",
        "- `overview.png`：阈值、预测数量、目标尺度和类别差异。",
        "- `contact_sheets/`：六组候选案例总览；`cases/` 中是可放大的单图面板。",
        "- `case_index.csv`：选择分数及逐 GT、图像类别指标；selection_score 只用于排序，不是新指标。",
        "- `threshold_sensitivity.csv`：缓存可支持的五个阈值结果，以及三个探索性组合。",
        "- `review_template.csv`：人工查看后填写 `useful_collective`、`duplicate_or_neighbor_spill`、`oversized_or_background`、`miss`、`unclear` 等标签。", "",
        "## 解释边界", "",
        "`high_cg_high_class_quality` 只表示 CG、类别并集 Dice 和类别空间精度同时较高；`high_cg_class_tension` 表示 CG 与类别级空间质量出现张力。局部精度只看当前 GT，类别精度允许同一批预测覆盖相邻同类 GT；两者之差是 neighboring_class_credit。小目标采用 COCO 的面积分箱，只是诊断切片。", "",
        "现有探索性组合都没有定稿：`dataset_H_mAC_Dice` 混合了 GT macro 与 active-pair macro；`active_pair_mean_gt_H_AC_precision` 使用共享的图像类别空间精度，仍可能让大目标补贴小目标。逐 GT 局部 UnionDice 还会忽略远处 orphan。它们只用于观察失效方式，不进入正式 VC。", "",
        "## 看图后的下一步", "",
        "优先检查 single_gt_low_overlap_cg，再对其余候选做人工归类，确认 CG 中有多少是真实互补覆盖，以及 AC-Dice 分歧是否主要来自大框、orphan 或相邻目标。之后再决定 VC 是否需要逐 GT 的局部空间精度/责任范围。公式定稿后，才扩展至模型与 seed 对比并加入 AP、AR、ABO、LRP/oLRP、固定阈值 F1。现阶段不恢复 mAC loss 训练。", "",
    ]
    return "\n".join(lines)


def analyze_run(source_run, *, thresholds=(0.05, 0.10, 0.25, 0.50, 0.75),
                primary_theta=0.25, low_theta=0.05, cases_per_group=6, output=None):
    source_run, document, raw, audit, complete, source_metadata = validate_source(source_run)
    thresholds = tuple(float(value) for value in thresholds)
    if len(set(thresholds)) != len(thresholds) or any(value < document["metadata"]["postprocessing"]["confidence_floor"] or value > 1 for value in thresholds):
        raise ValueError("thresholds must be unique and within the exported confidence range")
    if primary_theta not in thresholds or low_theta not in thresholds or low_theta >= primary_theta:
        raise ValueError("low and primary thresholds must be included and ordered")
    if type(cases_per_group) is not int or cases_per_group < 1:
        raise ValueError("cases_per_group must be a positive integer")
    class_names = document["metadata"]["class_names"]
    evaluated, local_by_threshold, threshold_rows = {}, {}, []
    for theta in thresholds:
        _, images, summary = evaluate_document(document, theta=theta, tau=complete["summary"]["tau"])
        evaluated[theta] = images
        local_rows = local_gt_diagnostics(images, document, class_names, theta)
        local_by_threshold[theta] = local_rows
        threshold_rows.append(dict(
            summary,
            dataset_H_mAC_Dice=harmonic(summary["mAC"], summary["ClassUnionDice"]),
            active_pair_mean_gt_H_AC_precision=pair_vc(images),
            mean_gt_local_union_dice=statistics.fmean(
                row["selected_gt_local_dice"] for row in local_rows)))
    primary_summary = next(row for row in threshold_rows if row["theta"] == primary_theta)
    for key, value in complete["summary"].items():
        if key in primary_summary and primary_summary[key] != value:
            raise ValueError(f"recomputed primary summary differs at {key}")
    class_rows = class_summary(evaluated[primary_theta], class_names)
    size_rows = size_summary(evaluated[primary_theta])
    primary_local_rows = local_by_threshold[primary_theta]
    local_stats = local_support_summary(primary_local_rows)
    cases = rank_cases(evaluated[primary_theta], evaluated[low_theta], class_names, cases_per_group)
    cases.extend(rank_single_gt_low_overlap_cases(
        evaluated[primary_theta], primary_local_rows, class_names, cases_per_group))
    missing_groups = [name for name in CASE_GROUPS if sum(row["category"] == name for row in cases) < cases_per_group]
    if missing_groups:
        raise ValueError(f"not enough cases for {missing_groups}")
    output = (Path(output).resolve() if output else PROJECT_ROOT / "outputs" / "region_eval" /
              ("case_analysis_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    archive = (PROJECT_ROOT / "bin" / "old").resolve()
    if output.exists() or output == archive or output.is_relative_to(archive):
        raise ValueError("output must be a new directory outside bin/old")
    output.mkdir(parents=True)
    _write_json(output / "analysis_manifest.json", dict(
        status="started", analysis_version=ANALYSIS_VERSION, source_run=str(source_run),
        source_filtered_sha256=_file_hash(source_run / "filtered_predictions.json"),
        source_raw_sha256=_file_hash(source_run / "raw_predictions.json"),
        parameters=dict(thresholds=thresholds, primary_theta=primary_theta,
                        low_theta=low_theta, cases_per_group=cases_per_group),
        source_report_metadata_sha256=_file_hash(source_run / "report" / "metadata.json"),
        source_code_sha256={str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _file_hash(path)
                            for path in [PROJECT_ROOT / "region_eval" / "case_analysis.py",
                                         PROJECT_ROOT / "run_region_case_analysis.py"]}))
    threshold_fields = ["theta", "n_pred", "avg_preds", "mAC", "BestIoGT", "CollectiveGain",
                        "RecallACtau", "ClassUnionDice", "ClassUnionIoU", "orphan_pred_count",
                        "multigt_pred_count", "n_active_image_classes", "dataset_H_mAC_Dice",
                        "active_pair_mean_gt_H_AC_precision", "mean_gt_local_union_dice"]
    class_fields = list(class_rows[0])
    size_fields = list(size_rows[0])
    case_fields = ["case_id", "category", "rank", "panel_file", "selection_score", "image_id",
                   "class_id", "class_name", "gt_id", "gt_area", "AC", "BestIoGT_i",
                   "CollectiveGain_i", "intersecting_pred_count", "best_pred_id", "pair_mean_AC",
                   "pair_mean_BestIoGT", "pair_mean_CollectiveGain", "pair_Dice", "pair_IoU",
                   "pair_SpatialPrecision", "pair_n_gt", "pair_n_pred", "pair_orphan_pred_count",
                   "pair_multigt_pred_count", "low_pair_mean_AC", "low_pair_Dice",
                   "low_pair_SpatialPrecision", "low_pair_n_pred", "low_pair_orphan_pred_count",
                   "delta_AC_low_minus_primary", "delta_Dice_low_minus_primary",
                   "contributing_pred_count", "contributing_multigt_count",
                   "contributing_single_gt_count", "contributing_pred_ids", "contributing_scores",
                   "contributing_union_area", "selected_gt_local_precision",
                   "contributing_class_precision", "neighboring_class_credit",
                   "selected_gt_local_dice", "max_contributing_pairwise_iou",
                   "mean_contributing_pairwise_iou"]
    _write_csv(output / "threshold_sensitivity.csv", threshold_rows, threshold_fields)
    _write_csv(output / "class_summary.csv", class_rows, class_fields)
    _write_csv(output / "size_summary.csv", size_rows, size_fields)
    local_fields = ["image_id", "gt_id", "class_id", "class_name", "theta", "gt_area", "AC",
                    "BestIoGT_i", "CollectiveGain_i", "coverage_hit",
                    "intersecting_pred_count", "best_pred_id", "contributing_pred_count",
                    "contributing_multigt_count", "contributing_single_gt_count",
                    "contributing_pred_ids", "contributing_scores", "contributing_union_area",
                    "selected_gt_local_precision", "contributing_class_precision",
                    "neighboring_class_credit", "selected_gt_local_dice",
                    "max_contributing_pairwise_iou", "mean_contributing_pairwise_iou"]
    _write_csv(output / "region_gt_local.csv", primary_local_rows, local_fields)
    _write_json(output / "local_support_summary.json", local_stats)
    document_by_id = {row["image_id"]: row for row in document["images"]}
    raw_by_id = {row["image_id"]: row for row in raw["images"]}
    manifest_by_id = {row["image_id"]: row for row in document["metadata"]["subset"]["image_manifest"]}
    case_root, contact_root = output / "cases", output / "contact_sheets"
    case_root.mkdir()
    contact_root.mkdir()
    for group in CASE_GROUPS:
        group_dir = case_root / group
        group_dir.mkdir()
        files = []
        for case in [row for row in cases if row["category"] == group]:
            enrich_case_geometry(case, document_by_id[case["image_id"]], primary_theta)
            filename = f"{case['case_id']}_{case['image_id']}_{case['class_name']}_{case['gt_id']}.png"
            destination = group_dir / filename
            image_path = Path(manifest_by_id[case["image_id"]]["image_file"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            render_case(case, document_by_id[case["image_id"]], raw_by_id[case["image_id"]],
                        image_path, destination, primary_theta, low_theta)
            case["panel_file"] = str(destination.relative_to(output)).replace("\\", "/")
            files.append(destination)
        contact_sheet(group, files, contact_root / f"{group}.png")
    _write_csv(output / "case_index.csv", cases, case_fields)
    review_rows = [dict(case_id=row["case_id"], category=row["category"], image_id=row["image_id"],
                        class_name=row["class_name"], gt_id=row["gt_id"], human_label="", notes="")
                   for row in cases]
    _write_csv(output / "review_template.csv", review_rows,
               ["case_id", "category", "image_id", "class_name", "gt_id", "human_label", "notes"])
    overview_plot(threshold_rows, class_rows, size_rows, output / "overview.png")
    (output / "README.md").write_text(
        _markdown(local_stats, threshold_rows, class_rows, size_rows, cases, source_run),
        encoding="utf-8", newline="\n")
    counts = dict(selected_cases=len(cases), groups={name: sum(row["category"] == name for row in cases)
                                                    for name in CASE_GROUPS},
                  source_images=len(document["images"]), source_gt=primary_summary["n_gt"],
                  source_predictions_at_primary=primary_summary["n_pred"],
                  local_gt_rows=len(primary_local_rows))
    _write_json(output / "analysis_complete.json", dict(
        status="complete", analysis_version=ANALYSIS_VERSION, source_run=str(source_run), counts=counts,
        primary_summary={key: primary_summary[key] for key in (
            "theta", "mAC", "BestIoGT", "CollectiveGain", "RecallACtau",
            "ClassUnionDice", "ClassUnionIoU", "orphan_pred_count", "multigt_pred_count")},
        local_support_summary=local_stats,
        output_sha256={str(path.relative_to(output)).replace("\\", "/"): _file_hash(path)
                       for path in sorted(output.rglob("*")) if path.is_file()
                       and path.name not in ("analysis_manifest.json", "analysis_complete.json")}))
    print(json.dumps(dict(output=str(output), **counts), ensure_ascii=False), flush=True)
    print("SUCCESS: offline case analysis complete. No inference or training performed.", flush=True)
    return output
