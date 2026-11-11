"""VC v1.0 candidate metric for cached, post-NMS object detections.

The metric is deliberately evaluation-only.  Predictions are first filtered by
confidence and class.  Each prediction with positive same-class overlap is
assigned to exactly one GT using the deterministic hard-IoU rule.  The union of
the predictions assigned to a GT defines its collective support region.

The dataset score is::

    VC(theta) = sum_i Dice(union(assigned predictions_i), GT_i)
                / (N_GT + N_orphan_predictions)

Standard one-to-one F1/LRP and AP remain separate diagnostics.
"""

from math import isfinite
from pathlib import Path
import statistics

from .case_analysis import _file_hash, _write_csv, _write_json
from .matching_analysis import ap_metrics, fixed_threshold_metrics
from .offline import evaluate_document
from .responsibility_analysis import score_document


VC_VERSION = "vc-candidate-1.0.0"
PRIMARY_THETA = 0.25
THETA_GRID = tuple(round(index * 0.05, 2) for index in range(1, 20))
TAU_GRID = (0.25, 0.50, 0.75)
MATCHING_IOU = 0.50


def _mean(rows, field):
    return statistics.fmean(row[field] for row in rows) if rows else None


def _validate_canonical_document(document):
    """Require the cache to contain clipped, finite, positive-area boxes."""
    metadata = document.get("metadata", {})
    class_ids = metadata.get("class_ids")
    class_names = metadata.get("class_names")
    if class_ids != list(range(len(class_ids or []))):
        raise ValueError("VC v1 requires contiguous zero-based class_ids")
    if not isinstance(class_names, list) or len(class_names) != len(class_ids):
        raise ValueError("metadata.class_names must align with class_ids")
    for image in document.get("images", []):
        width, height = image.get("width"), image.get("height")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and isfinite(value) and value > 0 for value in (width, height)):
            raise ValueError("images require positive finite width and height")
        for kind in ("gts", "predictions"):
            for box in image.get(kind, []):
                rect = box.get("xyxy")
                if (not isinstance(rect, list) or len(rect) != 4
                        or not all(isinstance(value, (int, float))
                                   and not isinstance(value, bool)
                                   and isfinite(value) for value in rect)):
                    raise ValueError("VC v1 requires finite xyxy boxes")
                x1, y1, x2, y2 = rect
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError("VC v1 cache boxes must be clipped to the image")


def evaluate_vc(document, *, theta=PRIMARY_THETA, tau=0.50):
    """Evaluate final-candidate VC and separate fixed-threshold diagnostics."""
    if not 0.0 <= theta <= 1.0 or not 0.0 <= tau <= 1.0:
        raise ValueError("theta and tau must be in [0, 1]")
    _validate_canonical_document(document)
    _, _, region = evaluate_document(document, theta=theta, tau=tau)
    gt_rows, pair_rows = score_document(document, theta, "hard_iou", tau)

    n_gt = len(gt_rows)
    n_orphan = sum(row["orphan_pred_count"] for row in pair_rows)
    local_vc_sum = sum(row["responsibility_dice"] for row in gt_rows)
    denominator = n_gt + n_orphan
    vc = local_vc_sum / denominator if denominator else None
    if vc is not None and not 0.0 <= vc <= 1.0:
        raise AssertionError("VC must lie in [0, 1]")

    class_names = document["metadata"]["class_names"]
    fixed_rows, _ = fixed_threshold_metrics(
        document,
        class_names,
        confidence_thresholds=(theta,),
        iou_thresholds=(MATCHING_IOU,),
    )
    standard = fixed_rows[0]
    n_pred = region["n_pred"]
    result = dict(
        metric_version=VC_VERSION,
        confidence_threshold=theta,
        matching_iou=MATCHING_IOU,
        n_images=region["n_images"],
        n_gt=n_gt,
        n_pred=n_pred,
        assigned_pred_count=sum(row["assigned_pred_count"] for row in pair_rows),
        orphan_pred_count=n_orphan,
        orphan_rate=n_orphan / n_pred if n_pred else 0.0,
        mAC=region["mAC"],
        mean_responsibility_coverage=_mean(gt_rows, "responsibility_coverage"),
        mean_spatial_precision=_mean(gt_rows, "responsibility_precision"),
        mean_local_VC=_mean(gt_rows, "responsibility_dice"),
        VC=vc,
        ClassUnionDice=region["ClassUnionDice"],
        standard_TP=standard["micro_TP"],
        standard_FP=standard["micro_FP"],
        standard_FN=standard["micro_FN"],
        standard_precision=standard["micro_precision"],
        standard_recall=standard["micro_recall"],
        standard_F1=standard["micro_F1"],
        standard_LRP=standard["micro_LRP"],
    )
    return result, gt_rows, pair_rows


def analyze_vc_document(
        document,
        *,
        confidence_thresholds=THETA_GRID,
        primary_theta=PRIMARY_THETA,
        coverage_thresholds=TAU_GRID):
    """Compute the complete offline smoke analysis from one prediction cache."""
    thresholds = tuple(float(value) for value in confidence_thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("confidence thresholds must be unique and nonempty")
    if primary_theta not in thresholds:
        raise ValueError("primary theta must be present in the threshold grid")
    floor = document["metadata"]["postprocessing"]["confidence_floor"]
    if min(thresholds) < floor:
        raise ValueError("threshold grid is below the exported confidence floor")

    summaries = []
    primary_gt_rows = primary_pair_rows = None
    for theta in thresholds:
        summary, gt_rows, pair_rows = evaluate_vc(
            document, theta=theta, tau=0.50)
        summaries.append(summary)
        if theta == primary_theta:
            primary_gt_rows, primary_pair_rows = gt_rows, pair_rows

    recall_rows = []
    for tau in coverage_thresholds:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("coverage thresholds must be in [0, 1]")
        recall_rows.append(dict(
            confidence_threshold=primary_theta,
            coverage_threshold=float(tau),
            n_gt=len(primary_gt_rows),
            RecallACtau=(statistics.fmean(
                row["responsibility_coverage"] >= tau
                for row in primary_gt_rows) if primary_gt_rows else None),
        ))

    ap_summary, ap_class_rows = ap_metrics(
        document, document["metadata"]["class_names"])
    ap_summary["max_detections_note"] = (
        "diagnostic AP from the shared post-NMS VC cache; cache max_det="
        f"{document['metadata']['postprocessing']['max_det']}; use the official "
        "dataset evaluator for the paper's standard AP table"
    )
    return dict(
        threshold_rows=summaries,
        primary_gt_rows=primary_gt_rows,
        primary_pair_rows=primary_pair_rows,
        recall_rows=recall_rows,
        diagnostic_ap_summary=ap_summary,
        diagnostic_ap_class_rows=ap_class_rows,
    )


def write_vc_analysis(output, *, document, source_file, analysis):
    """Write a fresh, self-describing VC analysis directory."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"analysis output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    threshold_rows = analysis["threshold_rows"]
    gt_rows = analysis["primary_gt_rows"]
    pair_rows = analysis["primary_pair_rows"]
    ap_rows = analysis["diagnostic_ap_class_rows"]
    _write_csv(output / "vc_thresholds.csv", threshold_rows, list(threshold_rows[0]))
    _write_csv(output / "vc_gt_primary.csv", gt_rows, list(gt_rows[0]) if gt_rows else ["image_id"])
    _write_csv(output / "vc_pairs_primary.csv", pair_rows, list(pair_rows[0]) if pair_rows else ["image_id"])
    _write_csv(output / "recall_ac_tau.csv", analysis["recall_rows"], list(analysis["recall_rows"][0]))
    _write_csv(output / "diagnostic_ap_by_class.csv", ap_rows, list(ap_rows[0]))
    _write_json(output / "diagnostic_ap_summary.json", analysis["diagnostic_ap_summary"])
    manifest = dict(
        status="complete",
        metric_version=VC_VERSION,
        formula=("sum_i 2*area(U_i intersect G_i)/(area(U_i)+area(G_i)) "
                 "/ (N_GT + N_orphan_predictions)"),
        assignment=("class-aware hard IoU; positive-overlap predictions have exactly "
                    "one GT owner; IoU, intersection area, then GT id break ties"),
        source_file=str(Path(source_file).resolve()),
        source_sha256=_file_hash(source_file),
        primary_theta=PRIMARY_THETA,
        confidence_thresholds=list(THETA_GRID),
        coverage_thresholds=list(TAU_GRID),
        standard_metrics_role="reported separately; not a VC formula component",
        EPI_role="excluded",
    )
    _write_json(output / "analysis_complete.json", manifest)
    return output
