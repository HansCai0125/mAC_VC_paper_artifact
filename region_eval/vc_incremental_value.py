"""Controlled incremental-value tests for VC v1.

The suite isolates three properties that are not supplied by coverage alone:

1. spatial-specificity under equal mAC,
2. dataset-level penalties for orphan predictions, and
3. single-owner responsibility when one prediction supports multiple GTs.

Canonical cases are evaluated by the production VC implementation.  The
real-run orphan analysis is an exact algebraic projection from the frozen
18-run formal matrix; it does not modify predictions, train a model, or run
inference.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import statistics

from .case_analysis import _file_hash
from .gaming_formal import (
    MATRIX_ROOT,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _fingerprint,
    _source_records,
)
from .offline import PROJECT_ROOT
from .vc_v1 import PRIMARY_THETA, VC_VERSION, evaluate_vc


INCREMENTAL_VALUE_VERSION = "vc-v1-incremental-value-0.1.0"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "vc_components"
)

SCALE_GRID = (0.50, 0.71, 0.80, 0.90, 1.00, 1.10, 1.25, 1.41, 1.50, 2.00, 3.00)
ORPHAN_COUNTS = (0, 1, 2, 4, 8, 16)
ORPHAN_RATIOS = (0.00, 0.25, 0.50, 1.00, 2.00)
GAP_RATIOS = (0.00, 0.25, 0.50, 1.00, 2.00, 4.00)
MODELS = ("yolov8n", "yolo11n", "yolo11s")
MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolo11n": "YOLO11n",
    "yolo11s": "YOLO11s",
}

ABS_TOLERANCE = 1e-10


def _document(gts, predictions, *, class_names=("object",)):
    """Return a canonical synthetic document accepted by VC v1."""
    return {
        "schema_version": 1,
        "metadata": {
            "dataset": "vc_incremental_value_synthetic",
            "split": "controlled",
            "model_id": "deterministic_geometry",
            "checkpoint_id": "none",
            "coordinate_frame": "original_image_xyxy",
            "ignore_crowd_policy": "already_filtered",
            "class_ids": list(range(len(class_names))),
            "class_names": list(class_names),
            "postprocessing": {
                "stage": "synthetic",
                "confidence_floor": 0.0,
                "nms_iou": None,
                "max_det": 100,
            },
        },
        "images": [{
            "image_id": "controlled_image",
            "width": 1000,
            "height": 1000,
            "gts": gts,
            "predictions": predictions,
        }],
    }


def _gt(identifier, xyxy, class_id=0):
    return {"id": identifier, "class_id": class_id, "xyxy": list(xyxy)}


def _prediction(identifier, xyxy, class_id=0, score=0.99):
    return {
        "id": identifier,
        "class_id": class_id,
        "score": float(score),
        "xyxy": list(xyxy),
    }


def _close(actual, expected, label):
    if not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=ABS_TOLERANCE
    ):
        raise AssertionError(
            f"{label}: expected {expected:.16g}, received {actual:.16g}"
        )


def _metric_row(experiment, variant, level, result, **extra):
    return {
        "experiment": experiment,
        "variant": variant,
        "level": level,
        **extra,
        "mAC": float(result["mAC"]),
        "VC": float(result["VC"]),
        "mean_local_VC": float(result["mean_local_VC"]),
        "mean_responsibility_coverage": float(
            result["mean_responsibility_coverage"]
        ),
        "mean_spatial_precision": float(result["mean_spatial_precision"]),
        "F1": float(result["standard_F1"]),
        "LRP": float(result["standard_LRP"]),
        "one_minus_LRP": 1.0 - float(result["standard_LRP"]),
        "n_gt": int(result["n_gt"]),
        "n_pred": int(result["n_pred"]),
        "assigned_pred_count": int(result["assigned_pred_count"]),
        "orphan_pred_count": int(result["orphan_pred_count"]),
        "TP": int(result["standard_TP"]),
        "FP": int(result["standard_FP"]),
        "FN": int(result["standard_FN"]),
    }


def equal_mac_expansion_rows():
    """Score centered contraction/expansion using the production evaluator."""
    gt_box = (400.0, 400.0, 600.0, 600.0)
    cx = cy = 500.0
    half_side = 100.0
    rows = []
    for scale in SCALE_GRID:
        radius = half_side * scale
        pred_box = (cx - radius, cy - radius, cx + radius, cy + radius)
        document = _document(
            [_gt("target", gt_box)],
            [_prediction("scaled_prediction", pred_box)],
        )
        result, _, _ = evaluate_vc(document, theta=PRIMARY_THETA)
        area_ratio = scale * scale
        if scale <= 1.0:
            expected_mac = area_ratio
            expected_precision = 1.0
            expected_vc = 2.0 * area_ratio / (1.0 + area_ratio)
        else:
            expected_mac = 1.0
            expected_precision = 1.0 / area_ratio
            expected_vc = 2.0 / (1.0 + area_ratio)
        _close(result["mAC"], expected_mac, "equal-mAC scaling mAC")
        _close(result["VC"], expected_vc, "equal-mAC scaling VC")
        _close(
            result["mean_spatial_precision"],
            expected_precision,
            "equal-mAC scaling precision",
        )
        rows.append(_metric_row(
            "equal_mac_expansion",
            "center_scale",
            scale,
            result,
            scale=scale,
            prediction_to_gt_area_ratio=area_ratio,
            expected_mAC=expected_mac,
            expected_VC=expected_vc,
            absolute_formula_error=abs(float(result["VC"]) - expected_vc),
        ))
    return rows


def orphan_injection_rows():
    """Hold a perfect local union fixed while adding deterministic orphans."""
    target = _gt("target", (400.0, 400.0, 600.0, 600.0))
    rows = []
    for count in ORPHAN_COUNTS:
        predictions = [
            _prediction("perfect_prediction", (400.0, 400.0, 600.0, 600.0))
        ]
        for index in range(count):
            column = index % 4
            row = index // 4
            x1 = 40.0 + 25.0 * column
            y1 = 40.0 + 25.0 * row
            predictions.append(_prediction(
                f"orphan_{index:02d}", (x1, y1, x1 + 10.0, y1 + 10.0)
            ))
        result, _, _ = evaluate_vc(
            _document([target], predictions), theta=PRIMARY_THETA
        )
        expected_vc = 1.0 / (1.0 + count)
        expected_f1 = 2.0 / (2.0 + count)
        _close(result["mAC"], 1.0, "orphan-injection mAC")
        _close(result["mean_local_VC"], 1.0, "orphan-injection local VC")
        _close(result["VC"], expected_vc, "orphan-injection dataset VC")
        _close(result["standard_F1"], expected_f1, "orphan-injection F1")
        if int(result["orphan_pred_count"]) != count:
            raise AssertionError("orphan-injection count audit failed")
        rows.append(_metric_row(
            "orphan_injection",
            "fixed_perfect_union",
            count,
            result,
            added_orphan_count=count,
            added_orphans_per_gt=float(count),
            expected_mAC=1.0,
            expected_VC=expected_vc,
            absolute_formula_error=abs(float(result["VC"]) - expected_vc),
        ))
    return rows


def responsibility_competition_rows():
    """Compare target-specific predictions with one shared prediction."""
    rows = []
    for gap_ratio in GAP_RATIOS:
        first = (300.0, 450.0, 400.0, 550.0)
        second_x1 = 400.0 + 100.0 * gap_ratio
        second = (second_x1, 450.0, second_x1 + 100.0, 550.0)
        gts = [_gt("target_a", first), _gt("target_b", second)]

        separate_predictions = [
            _prediction("prediction_a", first),
            _prediction("prediction_b", second),
        ]
        separate, _, _ = evaluate_vc(
            _document(gts, separate_predictions), theta=PRIMARY_THETA
        )
        _close(separate["mAC"], 1.0, "separate-owner mAC")
        _close(separate["VC"], 1.0, "separate-owner VC")
        rows.append(_metric_row(
            "responsibility_competition",
            "separate_owners",
            gap_ratio,
            separate,
            gap_ratio=gap_ratio,
            expected_mAC=1.0,
            expected_VC=1.0,
            absolute_formula_error=abs(float(separate["VC"]) - 1.0),
        ))

        shared_box = (first[0], first[1], second[2], first[3])
        shared, gt_rows, _ = evaluate_vc(
            _document(gts, [_prediction("shared_prediction", shared_box)]),
            theta=PRIMARY_THETA,
        )
        expected_vc = 1.0 / (3.0 + gap_ratio)
        _close(shared["mAC"], 1.0, "shared-owner mAC")
        _close(shared["VC"], expected_vc, "shared-owner VC")
        if sum(row["assigned_pred_count"] for row in gt_rows) != 1:
            raise AssertionError("shared prediction must have exactly one owner")
        if sum(row["responsibility_coverage"] == 0.0 for row in gt_rows) != 1:
            raise AssertionError("one target must remain unsupported after ownership")
        rows.append(_metric_row(
            "responsibility_competition",
            "single_shared_owner",
            gap_ratio,
            shared,
            gap_ratio=gap_ratio,
            expected_mAC=1.0,
            expected_VC=expected_vc,
            absolute_formula_error=abs(float(shared["VC"]) - expected_vc),
        ))
    return rows


def canonical_property_rows():
    rows = (
        equal_mac_expansion_rows()
        + orphan_injection_rows()
        + responsibility_competition_rows()
    )
    if max(row["absolute_formula_error"] for row in rows) > ABS_TOLERANCE:
        raise AssertionError("canonical formula audit exceeded tolerance")
    # The controlled experiments have different level descriptors. Use one
    # deterministic superset so they can share a single auditable CSV.
    ordered_fields = [
        "experiment", "variant", "level",
        "scale", "prediction_to_gt_area_ratio",
        "added_orphan_count", "added_orphans_per_gt", "gap_ratio",
        "mAC", "VC", "mean_local_VC",
        "mean_responsibility_coverage", "mean_spatial_precision",
        "F1", "LRP", "one_minus_LRP",
        "n_gt", "n_pred", "assigned_pred_count", "orphan_pred_count",
        "TP", "FP", "FN",
        "expected_mAC", "expected_VC", "absolute_formula_error",
    ]
    unknown = sorted(set().union(*(row.keys() for row in rows)) - set(ordered_fields))
    if unknown:
        raise AssertionError(f"unregistered canonical output fields: {unknown}")
    return [{field: row.get(field) for field in ordered_fields} for row in rows]


def project_real_orphan_rows(records):
    """Apply the dataset VC denominator exactly to the frozen 18 runs."""
    rows = []
    for record in records:
        n_gt = int(record["n_gt"])
        n_pred = int(record["n_pred"])
        n_orphan = int(record["orphan_pred_count"])
        tp = int(record["standard_TP"])
        fp = int(record["standard_FP"])
        fn = int(record["standard_FN"])
        baseline_vc = float(record["VC"])
        baseline_mac = float(record["mAC"])
        baseline_f1 = float(record["standard_F1"])
        numerator = baseline_vc * (n_gt + n_orphan)
        for ratio in ORPHAN_RATIOS:
            added = int(round(ratio * n_gt))
            projected_vc = numerator / (n_gt + n_orphan + added)
            projected_f1 = (
                2.0 * tp / (2.0 * tp + fp + added + fn)
                if (2 * tp + fp + added + fn) else 0.0
            )
            expected_normalized_vc = (
                (n_gt + n_orphan) / (n_gt + n_orphan + added)
            )
            normalized_vc = projected_vc / baseline_vc
            _close(
                normalized_vc,
                expected_normalized_vc,
                "real-run orphan projection",
            )
            rows.append({
                "dataset": record["dataset"],
                "model": record["model"],
                "seed": int(record["seed"]),
                "model_id": record["model_id"],
                "source_run_fingerprint": record["source_run_fingerprint"],
                "added_orphans_per_gt": ratio,
                "added_orphan_count": added,
                "n_gt": n_gt,
                "baseline_n_pred": n_pred,
                "baseline_orphan_count": n_orphan,
                "projected_orphan_count": n_orphan + added,
                "projected_orphan_rate": (
                    (n_orphan + added) / (n_pred + added)
                    if (n_pred + added) else 0.0
                ),
                "baseline_mAC": baseline_mac,
                "projected_mAC": baseline_mac,
                "normalized_mAC": 1.0,
                "baseline_VC": baseline_vc,
                "projected_VC": projected_vc,
                "normalized_VC": normalized_vc,
                "baseline_F1": baseline_f1,
                "projected_F1": projected_f1,
                "normalized_F1": (
                    projected_f1 / baseline_f1 if baseline_f1 else None
                ),
            })
    return rows


def aggregate_real_orphan_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["model"], row["added_orphans_per_gt"]
        )
        groups[key].append(row)
    output = []
    fields = (
        "projected_orphan_rate", "projected_mAC", "normalized_mAC",
        "projected_VC", "normalized_VC", "projected_F1", "normalized_F1",
    )
    for key in sorted(groups):
        dataset, model, ratio = key
        group = groups[key]
        aggregate = {
            "dataset": dataset,
            "model": model,
            "added_orphans_per_gt": ratio,
            "n_runs": len(group),
            "uncertainty": (
                "single fixed checkpoint"
                if len(group) == 1
                else "sample standard deviation over five runs"
            ),
        }
        for field in fields:
            values = [float(row[field]) for row in group]
            aggregate[f"{field}_mean"] = statistics.fmean(values)
            aggregate[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(aggregate)
    return output


def _protocol(matrix_root, records):
    value = {
        "incremental_value_version": INCREMENTAL_VALUE_VERSION,
        "vc_version": VC_VERSION,
        "primary_theta": PRIMARY_THETA,
        "source_matrix": str(Path(matrix_root).resolve()),
        "source_matrix_fingerprint": _fingerprint([
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "run_fingerprint": row["source_run_fingerprint"],
                "cache_sha256": row["source_cache_sha256"],
            }
            for row in records
        ]),
        "scale_grid": list(SCALE_GRID),
        "orphan_counts": list(ORPHAN_COUNTS),
        "orphan_ratios": list(ORPHAN_RATIOS),
        "gap_ratios": list(GAP_RATIOS),
        "geometry": "continuous axis-aligned rectangles",
        "source_code": {
            "vc_incremental_value.py": _file_hash(Path(__file__)),
            "vc_v1.py": _file_hash(Path(__file__).with_name("vc_v1.py")),
            "responsibility_analysis.py": _file_hash(
                Path(__file__).with_name("responsibility_analysis.py")
            ),
        },
        "training": False,
        "inference": False,
    }
    value["protocol_fingerprint"] = _fingerprint(value)
    return value


def preflight_vc_incremental_value(*, matrix_root=MATRIX_ROOT, output=DEFAULT_OUTPUT):
    matrix_root = Path(matrix_root).resolve()
    records = _source_records(matrix_root)
    if len(records) != 18:
        raise ValueError(f"expected 18 frozen source runs, found {len(records)}")
    if sum(row["dataset"] == "mscoco" for row in records) != 3:
        raise ValueError("expected three fixed MS COCO runs")
    if sum(row["dataset"] == "visdrone" for row in records) != 15:
        raise ValueError("expected fifteen VisDrone runs")
    protocol = _protocol(matrix_root, records)
    canonical = canonical_property_rows()
    summary = {
        "status": "passed",
        "incremental_value_version": INCREMENTAL_VALUE_VERSION,
        "vc_version": VC_VERSION,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "source_runs": len(records),
        "canonical_rows": len(canonical),
        "real_projection_rows": len(records) * len(ORPHAN_RATIOS),
        "maximum_formula_error": max(
            row["absolute_formula_error"] for row in canonical
        ),
        "output": str(Path(output).resolve()),
        "training": False,
        "inference": False,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(
        "PREFLIGHT SUCCESS: VC incremental-value protocol is complete; "
        "no training or inference will be performed.",
        flush=True,
    )
    return summary


def _plot_canonical(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "mAC": "#0072B2",
        "VC": "#D55E00",
        "F1": "#009E73",
        "one_minus_LRP": "#6F6F6F",
    }
    labels = {
        "mAC": r"$mAC$",
        "VC": r"$VC$",
        "F1": "F1",
        "one_minus_LRP": r"$1-\mathrm{LRP}$",
    }
    markers = {"mAC": "o", "VC": "s", "F1": "^", "one_minus_LRP": "D"}
    fields = ("mAC", "VC", "F1", "one_minus_LRP")

    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))

    scaling = sorted(
        (row for row in rows if row["experiment"] == "equal_mac_expansion"),
        key=lambda row: row["scale"],
    )
    for field in fields:
        axes[0].plot(
            [row["scale"] for row in scaling],
            [row[field] for row in scaling],
            color=colors[field], marker=markers[field], linewidth=2,
            markersize=4, label=labels[field],
        )
    axes[0].axvline(1.0, color="0.4", linestyle=":", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_xticks((0.5, 0.71, 1.0, 1.41, 2.0, 3.0))
    axes[0].set_xticklabels(("0.50", "0.71", "1", "1.41", "2", "3"))
    axes[0].set(
        title="(a) Equal-coverage expansion",
        xlabel=r"Prediction side-length scale $s$",
        ylabel="Score",
        ylim=(-0.03, 1.05),
    )

    orphan = sorted(
        (row for row in rows if row["experiment"] == "orphan_injection"),
        key=lambda row: row["added_orphan_count"],
    )
    for field in fields:
        axes[1].plot(
            [row["added_orphan_count"] for row in orphan],
            [row[field] for row in orphan],
            color=colors[field], marker=markers[field], linewidth=2,
            markersize=4, label=labels[field],
        )
    axes[1].set_xscale("symlog", linthresh=1.0, base=2)
    axes[1].set_xticks(ORPHAN_COUNTS)
    axes[1].set_xticklabels([str(value) for value in ORPHAN_COUNTS])
    axes[1].set(
        title="(b) Fixed-union orphan injection",
        xlabel="Added orphan predictions per GT",
        ylabel="Score",
        ylim=(-0.03, 1.05),
    )

    competition = sorted(
        (
            row for row in rows
            if row["experiment"] == "responsibility_competition"
            and row["variant"] == "single_shared_owner"
        ),
        key=lambda row: row["gap_ratio"],
    )
    for field in fields:
        axes[2].plot(
            [row["gap_ratio"] for row in competition],
            [row[field] for row in competition],
            color=colors[field], marker=markers[field], linewidth=2,
            markersize=4, label=labels[field],
        )
    axes[2].axhline(
        1.0, color="#D55E00", linestyle=":", linewidth=1.0,
        label="VC with separate owners",
    )
    axes[2].set(
        title="(c) Shared-prediction ownership",
        xlabel=r"Inter-target gap / target width",
        ylabel="Score",
        ylim=(-0.03, 1.05),
    )

    for axis in axes:
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc="upper center", ncol=4, frameon=False
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    output = Path(output)
    figure.savefig(output / "vc_incremental_value_response.pdf", bbox_inches="tight")
    figure.savefig(
        output / "vc_incremental_value_response.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_real_projection(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    colors = ("#0072B2", "#009E73", "#D55E00")
    for axis, dataset, title in zip(
        axes, ("mscoco", "visdrone"), ("MS COCO", "VisDrone")
    ):
        for model, color in zip(MODELS, colors):
            selected = sorted(
                (
                    row for row in rows
                    if row["dataset"] == dataset and row["model"] == model
                ),
                key=lambda row: row["added_orphans_per_gt"],
            )
            x = [row["added_orphans_per_gt"] for row in selected]
            vc = [row["normalized_VC_mean"] for row in selected]
            vc_sd = [row["normalized_VC_std"] for row in selected]
            f1 = [row["normalized_F1_mean"] for row in selected]
            axis.plot(
                x, vc, color=color, marker="o", linewidth=2,
                label=f"{MODEL_LABELS[model]} VC",
            )
            if max(vc_sd, default=0.0) > 0:
                axis.fill_between(
                    x,
                    [max(0.0, value - error) for value, error in zip(vc, vc_sd)],
                    [value + error for value, error in zip(vc, vc_sd)],
                    color=color,
                    alpha=0.12,
                )
            axis.plot(
                x, f1, color=color, marker="x", linestyle="--", linewidth=1.2,
                alpha=0.75, label=f"{MODEL_LABELS[model]} F1",
            )
        axis.axhline(1.0, color="0.5", linestyle=":", linewidth=1.0)
        axis.set(
            title=title,
            xlabel="Added orphan predictions per GT",
            ylim=(0.0, 1.05),
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Score / unmodified score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    output = Path(output)
    figure.savefig(output / "real_orphan_projection.pdf", bbox_inches="tight")
    figure.savefig(
        output / "real_orphan_projection.png", dpi=260, bbox_inches="tight"
    )
    plt.close(figure)


def _anchor(rows, experiment, variant, field, value):
    return next(
        row for row in rows
        if row["experiment"] == experiment
        and row["variant"] == variant
        and math.isclose(float(row[field]), float(value), abs_tol=1e-12)
    )


def _latex_table(rows, output):
    perfect = _anchor(rows, "equal_mac_expansion", "center_scale", "scale", 1.0)
    expand_141 = _anchor(
        rows, "equal_mac_expansion", "center_scale", "scale", 1.41
    )
    expand_200 = _anchor(
        rows, "equal_mac_expansion", "center_scale", "scale", 2.0
    )
    orphan_4 = _anchor(
        rows, "orphan_injection", "fixed_perfect_union",
        "added_orphan_count", 4,
    )
    shared = _anchor(
        rows, "responsibility_competition", "single_shared_owner",
        "gap_ratio", 0.0,
    )
    table_rows = (
        ("Perfect support", perfect),
        (r"Equal-$mAC$ expansion ($s=1.41$)", expand_141),
        (r"Equal-$mAC$ expansion ($s=2$)", expand_200),
        ("Four added orphans", orphan_4),
        ("One prediction shared by two targets", shared),
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controlled incremental-value tests for VC. All cases are "
        r"scored by the production evaluator at $\theta=0.25$.}",
        r"\label{tab:vc_incremental_value}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Condition & $mAC$ & $VC$ & F1 & $1-\mathrm{LRP}$ \\",
        r"\hline",
    ]
    for label, row in table_rows:
        lines.append(
            f"{label} & {row['mAC']:.3f} & {row['VC']:.3f} & "
            f"{row['F1']:.3f} & {row['one_minus_LRP']:.3f} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    _atomic_text(Path(output) / "vc_incremental_value_table.tex", "\n".join(lines))


def _property_matrix(output):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Expected and observed roles of the complementary metrics.}",
        r"\label{tab:metric_property_matrix}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Controlled change & $mAC$ & $VC$ & Instance metrics \\",
        r"\hline",
        r"Reduce retained target support & Decreases & Decreases & May saturate after matching fails \\",
        r"Fragment a fixed prediction union & Invariant & Invariant & Penalize fragment multiplicity \\",
        r"Add area outside a fully covered target & Invariant & Decreases & Threshold-dependent \\",
        r"Add predictions with no valid owner & Invariant & Decreases & Penalize false positives \\",
        r"Duplicate an owned box exactly & Invariant & Invariant & Penalize duplicates \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    _atomic_text(Path(output) / "metric_property_matrix.tex", "\n".join(lines))


def _report(protocol, rows, projection, output):
    expand = _anchor(
        rows, "equal_mac_expansion", "center_scale", "scale", 1.41
    )
    shared = _anchor(
        rows, "responsibility_competition", "single_shared_owner",
        "gap_ratio", 0.0,
    )
    orphan = _anchor(
        rows, "orphan_injection", "fixed_perfect_union",
        "added_orphan_count", 4,
    )
    lines = [
        "# VC incremental-value experiment", "",
        "This controlled suite isolates what VC contributes beyond collective "
        "coverage. All canonical cases are scored by the production VC v1 "
        "implementation. The real-data orphan curves are exact projections "
        "from the frozen 18-run formal matrix.", "",
        "No training or model inference is performed.", "",
        "## Key anchors", "",
        f"- Equal-mAC expansion at s=1.41: mAC={expand['mAC']:.3f}, "
        f"VC={expand['VC']:.3f}, F1={expand['F1']:.3f}.",
        f"- Four added orphans with a fixed perfect union: "
        f"mAC={orphan['mAC']:.3f}, local VC={orphan['mean_local_VC']:.3f}, "
        f"dataset VC={orphan['VC']:.3f}.",
        f"- One prediction shared by two adjacent targets: "
        f"mAC={shared['mAC']:.3f}, VC={shared['VC']:.3f}, "
        f"F1={shared['F1']:.3f}.", "",
        "## Interpretation", "",
        "mAC supplies a coverage signal. VC does not inherit validity from mAC; "
        "it adds spatial specificity, single-owner responsibility, and an "
        "orphan-aware dataset denominator. Exact duplicates remain a known "
        "union-metric invariance and must be exposed by the reported instance "
        "metrics.", "",
        "## Outputs", "",
        "- vc_incremental_value_response.pdf: canonical three-property figure.",
        "- real_orphan_projection.pdf: exact projection over 18 formal runs.",
        "- vc_incremental_value_table.tex: compact paper table.",
        "- metric_property_matrix.tex: metric-role summary.",
        "- canonical_response.csv: auditable controlled values.",
        "- orphan_projection_by_run.csv and orphan_projection_by_model.csv: "
        "real-run projections.", "",
        f"Protocol fingerprint: `{protocol['protocol_fingerprint']}`", "",
        f"Real projection rows: {len(projection)}", "",
    ]
    _atomic_text(Path(output) / "README.md", "\n".join(lines))


def run_vc_incremental_value(*, matrix_root=MATRIX_ROOT, output=DEFAULT_OUTPUT):
    matrix_root = Path(matrix_root).resolve()
    output = Path(output).resolve()
    records = _source_records(matrix_root)
    protocol = _protocol(matrix_root, records)
    complete_path = output / "analysis_complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise ValueError(
                "completed output uses another protocol; choose a new RUN_NAME"
            )
        print(f"SKIP: VC incremental-value analysis already complete: {output}")
        return output

    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise ValueError(
                "partial output uses another protocol; choose a new RUN_NAME"
            )
    _atomic_json(protocol_path, protocol)

    print("VC-INCREMENTAL 1/4: scoring canonical property cases...", flush=True)
    canonical = canonical_property_rows()
    _atomic_csv(output / "canonical_response.csv", canonical)

    print("VC-INCREMENTAL 2/4: projecting orphan response over 18 runs...", flush=True)
    projection = project_real_orphan_rows(records)
    aggregate = aggregate_real_orphan_rows(projection)
    _atomic_csv(output / "orphan_projection_by_run.csv", projection)
    _atomic_csv(output / "orphan_projection_by_model.csv", aggregate)

    print("VC-INCREMENTAL 3/4: generating figures and LaTeX tables...", flush=True)
    _plot_canonical(canonical, output)
    _plot_real_projection(aggregate, output)
    _latex_table(canonical, output)
    _property_matrix(output)
    _report(protocol, canonical, projection, output)

    print("VC-INCREMENTAL 4/4: writing completion audit...", flush=True)
    complete = {
        "status": "complete",
        "incremental_value_version": INCREMENTAL_VALUE_VERSION,
        "vc_version": VC_VERSION,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "source_runs": len(records),
        "canonical_rows": len(canonical),
        "real_projection_rows": len(projection),
        "aggregate_projection_rows": len(aggregate),
        "maximum_formula_error": max(
            row["absolute_formula_error"] for row in canonical
        ),
        "training": False,
        "inference": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
    }
    _atomic_json(complete_path, complete)
    print(
        json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False),
        flush=True,
    )
    return output

