"""Calibration-only threshold selection for the frozen SAM experiment.

Every confidence-filtered, globally capped prediction set is a prefix of the
same score order.  We therefore calculate exact regional and instance sufficient
statistics once per image/prefix and reuse them across thresholds and budgets.
No evaluation/holdout images or downstream masks enter this module.
"""

from __future__ import annotations

from collections import defaultdict
from math import fsum, isfinite

from region_eval.geometry import area, intersection, union_area


POLICIES = ("fixed_025", "F1", "LRP", "BestIoGT", "mAC", "VC")
TIE_ATOL = 1e-12
MATCH_IOU = 0.5


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_budget(value):
    if type(value) is not int or value <= 0:
        raise ValueError("budget must be a positive integer")
    return value


def _score_order(predictions):
    rows = list(predictions)
    seen = set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("prediction IDs must be unique nonempty strings per image")
        seen.add(identifier)
        score = _number(row.get("score"), "prediction score")
        if not 0 <= score <= 1:
            raise ValueError("prediction score must be in [0, 1]")
    return sorted(rows, key=lambda row: (-row["score"], row["id"]))


def retain_predictions(predictions, theta, budget):
    """Filter and globally cap without changing records or class labels."""
    theta = _number(theta, "theta")
    if not 0 <= theta <= 1:
        raise ValueError("theta must be in [0, 1]")
    budget = _validate_budget(budget)
    return [row for row in _score_order(predictions) if row["score"] >= theta][:budget]


def _validate_image(image, class_ids):
    width = _number(image.get("width"), "image width")
    height = _number(image.get("height"), "image height")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    for field in ("gts", "predictions"):
        rows = image.get(field)
        if not isinstance(rows, list):
            raise ValueError(f"image {field} must be a list")
        seen = set()
        for row in rows:
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError(f"{field} IDs must be unique nonempty strings per image")
            seen.add(identifier)
            cls = row.get("class_id")
            if type(cls) is not int or cls not in class_ids:
                raise ValueError("undeclared or noninteger class_id")
            rect = row.get("xyxy")
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                raise ValueError("boxes require four xyxy coordinates")
            x1, y1, x2, y2 = [_number(x, "box coordinate") for x in rect]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("boxes must have positive area and already be clipped")


def _prefix_statistics(image, predictions, wanted):
    """Return additive statistics for requested score-order prefix lengths."""
    gts = sorted(image["gts"], key=lambda row: row["id"])
    by_class = defaultdict(list)
    for index, gt in enumerate(gts):
        by_class[gt["class_id"]].append(index)
    gt_rects = [tuple(gt["xyxy"]) for gt in gts]
    gt_areas = [area(rect) for rect in gt_rects]
    coverage_rects = [[] for _ in gts]
    owned_rects = [[] for _ in gts]
    owned_clips = [[] for _ in gts]
    ac = [0.0] * len(gts)
    best = [0.0] * len(gts)
    dice = [0.0] * len(gts)
    matched = set()
    local_errors = []
    tp = 0
    orphan = 0
    output = {}

    def snapshot(count):
        return dict(
            n_gt=len(gts), n_pred=count, TP=tp, FP=count - tp,
            FN=len(gts) - tp, orphan_pred_count=orphan,
            localization_error_sum=fsum(local_errors),
            ac_sum=fsum(ac), best_iogt_sum=fsum(best), local_vc_sum=fsum(dice),
        )

    if 0 in wanted:
        output[0] = snapshot(0)
    for count, prediction in enumerate(predictions, start=1):
        rect = tuple(prediction["xyxy"])
        pred_area = area(rect)
        hits = []
        for index in by_class[prediction["class_id"]]:
            overlap = intersection(rect, gt_rects[index])
            if overlap is None:
                continue
            overlap_area = area(overlap)
            iou = overlap_area / (pred_area + gt_areas[index] - overlap_area)
            hits.append((index, overlap, overlap_area, iou))
            coverage_rects[index].append(overlap)
            best[index] = min(1.0, max(best[index], overlap_area / gt_areas[index]))
            ac[index] = min(1.0, max(best[index], union_area(coverage_rects[index]) / gt_areas[index]))

        if hits:
            owner, overlap, _, _ = min(
                hits, key=lambda hit: (-hit[3], -hit[2], gts[hit[0]]["id"]))
            owned_rects[owner].append(rect)
            owned_clips[owner].append(overlap)
            support = union_area(owned_rects[owner])
            covered = min(gt_areas[owner], union_area(owned_clips[owner]))
            coverage = covered / gt_areas[owner]
            precision = covered / support if support else 0.0
            dice[owner] = (
                2.0 * coverage * precision / (coverage + precision)
                if coverage + precision else 0.0
            )
        else:
            orphan += 1

        matchable = [hit for hit in hits if hit[0] not in matched and hit[3] >= MATCH_IOU]
        if matchable:
            match = min(matchable, key=lambda hit: (-hit[3], -hit[2], gts[hit[0]]["id"]))
            matched.add(match[0])
            tp += 1
            local_errors.append((1.0 - match[3]) / (1.0 - MATCH_IOU))
        if count in wanted:
            output[count] = snapshot(count)
    return output


def calibrate(images, metadata, thresholds, budgets, progress=None):
    """Calculate all calibration rows; progress, when supplied, takes (done,total).

    Inputs must already follow the existing cache's clipping and crowd policy.
    The caller is responsible for passing *only* calibration images.
    F1/LRP are micro aggregated at IoU 0.5; mAC/BestIoGT are GT means;
    VC is sum(local owned-union Dice)/(GT count + orphan prediction count).
    """
    thresholds = tuple(sorted(_number(value, "theta") for value in thresholds))
    if not thresholds or len(set(thresholds)) != len(thresholds) or not all(0 <= t <= 1 for t in thresholds):
        raise ValueError("thresholds must be unique, nonempty and within [0,1]")
    budgets = tuple(sorted(_validate_budget(value) for value in budgets))
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be nonempty and unique")
    class_ids = metadata.get("class_ids")
    if not isinstance(class_ids, list) or class_ids != list(range(len(class_ids))) or not class_ids:
        raise ValueError("metadata.class_ids must be contiguous zero-based IDs")
    if len(metadata.get("class_names", [])) != len(class_ids):
        raise ValueError("metadata.class_names must align with class_ids")
    processing = metadata.get("postprocessing", {})
    floor = _number(processing.get("confidence_floor"), "confidence floor")
    if not 0 <= floor <= thresholds[0]:
        raise ValueError("cannot calibrate below the exported confidence floor")
    images = list(images)
    if not images:
        raise ValueError("calibration requires at least one image")
    identifiers = [image.get("image_id") for image in images]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("image IDs must be nonempty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("calibration image IDs must be unique")

    totals = {(budget, theta): defaultdict(float) for budget in budgets for theta in thresholds}
    int_fields = ("n_gt", "n_pred", "TP", "FP", "FN", "orphan_pred_count")
    for done, image in enumerate(images, start=1):
        _validate_image(image, set(class_ids))
        ranked = [p for p in _score_order(image["predictions"]) if p["score"] >= thresholds[0]][:max(budgets)]
        counts = {theta: sum(p["score"] >= theta for p in ranked) for theta in thresholds}
        wanted = {min(budget, counts[theta]) for budget in budgets for theta in thresholds}
        prefix_stats = _prefix_statistics(image, ranked[:max(wanted)], wanted)
        for (budget, theta), total in totals.items():
            stats = prefix_stats[min(budget, counts[theta])]
            for key, value in stats.items():
                total[key] += value
        if progress is not None:
            progress(done, len(images))

    output = []
    for (budget, theta), total in totals.items():
        ngt, npr = int(total["n_gt"]), int(total["n_pred"])
        tp, fp, fn = (int(total[key]) for key in ("TP", "FP", "FN"))
        vc_denom = ngt + int(total["orphan_pred_count"])
        lrp_denom = tp + fp + fn
        output.append(dict(
            budget=budget, theta=theta, n_images=len(images),
            **{key: int(total[key]) for key in int_fields},
            mean_prompts=npr / len(images),
            F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            LRP=(total["localization_error_sum"] + fp + fn) / lrp_denom if lrp_denom else None,
            BestIoGT=total["best_iogt_sum"] / ngt if ngt else None,
            mAC=total["ac_sum"] / ngt if ngt else None,
            VC=total["local_vc_sum"] / vc_denom if vc_denom else None,
            localization_error_sum=total["localization_error_sum"],
            local_vc_sum=total["local_vc_sum"],
        ))
    return output


def choose_policies(rows, fixed_theta=0.25):
    """Choose a common threshold per budget/policy using calibration rows only.

    Metric ties within absolute 1e-12 select the highest threshold; with nested
    sets this cannot retain more predictions. A fixed-threshold row must exist.
    """
    fixed_theta = _number(fixed_theta, "fixed_theta")
    by_budget = defaultdict(list)
    for row in rows:
        by_budget[_validate_budget(row["budget"])].append(dict(row))
    if not by_budget:
        raise ValueError("cannot select policies without calibration rows")
    output = []
    for budget in sorted(by_budget):
        candidates = by_budget[budget]
        if len({row["theta"] for row in candidates}) != len(candidates):
            raise ValueError("duplicate theta rows within a budget")
        fixed = [row for row in candidates if abs(row["theta"] - fixed_theta) <= TIE_ATOL]
        if len(fixed) != 1:
            raise ValueError("exactly one fixed-threshold calibration row is required")
        for policy in POLICIES:
            if policy == "fixed_025":
                chosen = fixed[0]
                selection_metric = "fixed_threshold"
                value = fixed_theta
            else:
                available = [row for row in candidates if row.get(policy) is not None and isfinite(row[policy])]
                if not available:
                    raise ValueError(f"{policy} undefined for every threshold at budget={budget}")
                best_value = (min if policy == "LRP" else max)(row[policy] for row in available)
                tied = [row for row in available if abs(row[policy] - best_value) <= TIE_ATOL]
                chosen = max(tied, key=lambda row: row["theta"])
                selection_metric = policy
                value = chosen[policy]
            output.append(dict(chosen, policy=policy, selection_metric=selection_metric,
                               calibration_value=value))
    return output
