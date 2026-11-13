"""Fixed-checkpoint COCO bootstrap and retained VisDrone five-seed report."""

from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import yaml

from .case_analysis import _file_hash, _write_csv, _write_json
from .geometry import area, intersection
from .offline import PROJECT_ROOT
from .responsibility_analysis import rectangle_iou
from .vc_v1 import PRIMARY_THETA, VC_VERSION


ANALYSIS_VERSION = "vc-fixed-checkpoint-bootstrap-0.1.0"
EXPECTED_MATRIX_FINGERPRINT = (
    "916baa3f128416d7f82a8bd680511a11e8586944669af951b5811aa67deb5141"
)
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_BATCH = 50
IOU_THRESHOLD = 0.50

MATRIX_ROOT = PROJECT_ROOT / "results" / "detection"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "fixed_checkpoint_analysis_v1"
COCO_ROOT = Path(r"E:\Project\Database\MS-COCO")

SELECTIONS = (
    dict(
        model="yolov8n",
        seed=0,
        weights=COCO_ROOT / "yolo8n" / "yolov8n_s0" / "weights" / "best.pt",
        args=COCO_ROOT / "yolo8n" / "yolov8n_s0" / "args.yaml",
        batch=128,
    ),
    dict(
        model="yolo11n",
        seed=1,
        weights=COCO_ROOT / "yolo11n" / "yolo11n_s1" / "weights" / "best.pt",
        args=COCO_ROOT / "yolo11n" / "yolo11n_s1" / "args.yaml",
        batch=128,
    ),
    dict(
        model="yolo11s",
        seed=0,
        weights=COCO_ROOT / "yolo11s" / "yolo11s_s0" / "weights" / "best.pt",
        args=COCO_ROOT / "yolo11s" / "yolo11s_s0" / "args.yaml",
        batch=96,
    ),
)

COMMON_TRAINING_ARGS = dict(
    epochs=300,
    patience=100,
    imgsz=640,
    pretrained=True,
    optimizer="SGD",
    cos_lr=True,
    close_mosaic=10,
    amp=True,
    rect=False,
    workers=8,
    freeze=0,
    val=True,
    iou=0.7,
    max_det=500,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    translate=0.05,
    scale=0.3,
    fliplr=0.5,
    flipud=0.0,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.1,
    copy_paste_mode="flip",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    nbs=64,
)

CONTRIBUTION_FIELDS = (
    "n_gt",
    "ac_sum",
    "responsibility_coverage_sum",
    "responsibility_precision_sum",
    "responsibility_dice_sum",
    "coverage_hit_050",
    "n_pred",
    "n_orphan",
    "standard_tp",
    "standard_fp",
    "standard_fn",
    "lrp_numerator",
)

BOOTSTRAP_METRICS = (
    "VC",
    "mAC",
    "mean_responsibility_coverage",
    "mean_spatial_precision",
    "mean_local_VC",
    "orphan_rate",
    "RecallAC050",
    "standard_precision",
    "standard_recall",
    "standard_F1",
    "standard_LRP",
)

POINT_FIELDS = (
    "VC", "mAC", "mean_responsibility_coverage",
    "mean_spatial_precision", "mean_local_VC", "orphan_rate",
    "RecallAC050", "standard_precision", "standard_recall",
    "standard_F1", "standard_LRP",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fingerprint(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _as_float(value, name):
    if value in {None, "", "NA"}:
        raise ValueError(f"missing numeric value for {name}")
    return float(value)


def _as_int(value, name):
    result = _as_float(value, name)
    if not result.is_integer():
        raise ValueError(f"{name} must be integral")
    return int(result)


def _validate_training_args(selection):
    args_path = Path(selection["args"]).resolve()
    weights_path = Path(selection["weights"]).resolve()
    if not args_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"selected training artifact is missing: {selection['model']}")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected = dict(
        COMMON_TRAINING_ARGS,
        seed=selection["seed"],
        batch=selection["batch"],
    )
    mismatches = {
        name: dict(expected=value, found=args.get(name))
        for name, value in expected.items()
        if args.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"selected checkpoint training args changed for "
            f"{selection['model']}: {mismatches}")
    return dict(
        model=selection["model"],
        selected_seed=selection["seed"],
        selection_rule=(
            "lowest-numbered available run in the shared augmented, "
            "patience-100 protocol group"
        ),
        weights=str(weights_path),
        weights_sha256=_file_hash(weights_path),
        args_yaml=str(args_path),
        args_yaml_sha256=_file_hash(args_path),
        verified_training_args=expected,
    )


def _matrix_inputs(matrix_root):
    matrix_root = Path(matrix_root).resolve()
    required = (
        "matrix_complete.json", "matrix_manifest.json", "matrix_runs.csv",
        "matrix_thresholds.csv", "aggregate_primary.csv",
    )
    missing = [name for name in required if not (matrix_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"formal matrix is incomplete: {missing}")
    complete = _read_json(matrix_root / "matrix_complete.json")
    if (complete.get("status") != "complete"
            or complete.get("matrix_fingerprint") != EXPECTED_MATRIX_FINGERPRINT
            or complete.get("completed_combinations") != 30):
        raise ValueError("formal matrix completion marker changed")
    return matrix_root, {
        name: _file_hash(matrix_root / name) for name in required
    }


def _selected_matrix_rows(matrix_root):
    rows = _read_csv(matrix_root / "matrix_runs.csv")
    selected = {}
    for item in SELECTIONS:
        matches = [
            row for row in rows
            if row["dataset"] == "mscoco"
            and row["model"] == item["model"]
            and _as_int(row["seed"], "seed") == item["seed"]
        ]
        if len(matches) != 1:
            raise ValueError(f"missing selected matrix row: {item['model']}")
        selected[item["model"]] = matches[0]
    return selected


def _validate_manifest_selection(matrix_root, audits):
    manifest = _read_json(matrix_root / "matrix_manifest.json")
    specifications = manifest.get("specifications", [])
    for audit in audits:
        matches = [
            row for row in specifications
            if row.get("kind") == "mscoco"
            and row.get("model") == audit["model"]
            and row.get("seed") == audit["selected_seed"]
        ]
        if len(matches) != 1:
            raise ValueError(f"selected manifest row missing: {audit['model']}")
        row = matches[0]
        if (Path(row["weights"]).resolve() != Path(audit["weights"])
                or row["weights_sha256"] != audit["weights_sha256"]):
            raise ValueError(f"selected checkpoint changed: {audit['model']}")


def _new_contributions(image_ids):
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate image IDs in fixed-checkpoint cache")
    return {
        image_id: {field: 0.0 for field in CONTRIBUTION_FIELDS}
        for image_id in image_ids
    }


def _region_contributions(contributions, gt_rows, pair_rows):
    for row in gt_rows:
        image_id = row["image_id"]
        if image_id not in contributions:
            raise ValueError(f"unknown GT image ID: {image_id}")
        target = contributions[image_id]
        target["n_gt"] += 1.0
        target["ac_sum"] += _as_float(row["original_AC"], "original_AC")
        coverage = _as_float(
            row["responsibility_coverage"], "responsibility_coverage")
        target["responsibility_coverage_sum"] += coverage
        target["responsibility_precision_sum"] += _as_float(
            row["responsibility_precision"], "responsibility_precision")
        target["responsibility_dice_sum"] += _as_float(
            row["responsibility_dice"], "responsibility_dice")
        target["coverage_hit_050"] += float(coverage >= 0.50)
    for row in pair_rows:
        image_id = row["image_id"]
        if image_id not in contributions:
            raise ValueError(f"unknown pair image ID: {image_id}")
        target = contributions[image_id]
        target["n_pred"] += _as_int(row["n_pred"], "n_pred")
        target["n_orphan"] += _as_int(
            row["orphan_pred_count"], "orphan_pred_count")


def _matching_contributions(contributions, document):
    """Add exact one-to-one matching sufficient statistics per image."""
    if set(contributions) != {
            image["image_id"] for image in document["images"]}:
        raise ValueError("prediction document image IDs changed")
    for image in document["images"]:
        target = contributions[image["image_id"]]
        gts_by_class = defaultdict(list)
        predictions_by_class = defaultdict(list)
        for gt in image["gts"]:
            gts_by_class[gt["class_id"]].append(gt)
        for prediction in image["predictions"]:
            if prediction["score"] >= PRIMARY_THETA:
                predictions_by_class[prediction["class_id"]].append(prediction)
        if sum(map(len, gts_by_class.values())) != int(target["n_gt"]):
            raise ValueError(f"GT count mismatch: {image['image_id']}")
        for class_id in sorted(set(gts_by_class) | set(predictions_by_class)):
            unmatched = {
                row["id"]: row
                for row in sorted(
                    gts_by_class[class_id], key=lambda value: value["id"])
            }
            predictions = sorted(
                predictions_by_class[class_id],
                key=lambda value: (-value["score"], value["id"]),
            )
            for prediction in predictions:
                candidates = []
                for gt_id, gt in unmatched.items():
                    iou = rectangle_iou(prediction["xyxy"], gt["xyxy"])
                    if iou >= IOU_THRESHOLD:
                        overlap = intersection(
                            tuple(prediction["xyxy"]), tuple(gt["xyxy"]))
                        candidates.append((iou, area(overlap), gt_id))
                if candidates:
                    iou, _, gt_id = sorted(
                        candidates,
                        key=lambda value: (-value[0], -value[1], value[2]),
                    )[0]
                    unmatched.pop(gt_id)
                    target["standard_tp"] += 1.0
                    target["lrp_numerator"] += (
                        (1.0 - iou) / (1.0 - IOU_THRESHOLD))
                else:
                    target["standard_fp"] += 1.0
            target["standard_fn"] += len(unmatched)
            target["lrp_numerator"] += (
                len(unmatched)
                + max(0, len(predictions) - (
                    len(gts_by_class[class_id]) - len(unmatched)))
            )


def _contribution_matrix(contributions, ordered_ids):
    return np.asarray([
        [contributions[image_id][field] for field in CONTRIBUTION_FIELDS]
        for image_id in ordered_ids
    ], dtype=np.float64)


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(
        numerator.shape, denominator.shape), np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=result, where=denominator != 0)


def metrics_from_totals(totals):
    """Convert additive image contributions to the reported estimands."""
    totals = np.asarray(totals, dtype=np.float64)
    if totals.shape[-1] != len(CONTRIBUTION_FIELDS):
        raise ValueError("invalid contribution vector")
    values = {
        field: totals[..., index]
        for index, field in enumerate(CONTRIBUTION_FIELDS)
    }
    tp, fp, fn = (
        values["standard_tp"], values["standard_fp"],
        values["standard_fn"],
    )
    return {
        "VC": _safe_divide(
            values["responsibility_dice_sum"],
            values["n_gt"] + values["n_orphan"]),
        "mAC": _safe_divide(values["ac_sum"], values["n_gt"]),
        "mean_responsibility_coverage": _safe_divide(
            values["responsibility_coverage_sum"], values["n_gt"]),
        "mean_spatial_precision": _safe_divide(
            values["responsibility_precision_sum"], values["n_gt"]),
        "mean_local_VC": _safe_divide(
            values["responsibility_dice_sum"], values["n_gt"]),
        "orphan_rate": _safe_divide(
            values["n_orphan"], values["n_pred"]),
        "RecallAC050": _safe_divide(
            values["coverage_hit_050"], values["n_gt"]),
        "standard_precision": _safe_divide(tp, tp + fp),
        "standard_recall": _safe_divide(tp, tp + fn),
        "standard_F1": _safe_divide(2.0 * tp, 2.0 * tp + fp + fn),
        "standard_LRP": _safe_divide(
            values["lrp_numerator"], tp + fp + fn),
    }


def _validate_point_estimate(model, matrix, row):
    totals = matrix.sum(axis=0)
    point = metrics_from_totals(totals)
    count_fields = {
        "n_gt": "n_gt",
        "n_pred": "n_pred",
        "n_orphan": "orphan_pred_count",
        "standard_tp": "standard_TP",
        "standard_fp": "standard_FP",
        "standard_fn": "standard_FN",
    }
    values = {
        field: totals[index]
        for index, field in enumerate(CONTRIBUTION_FIELDS)
    }
    for internal, external in count_fields.items():
        if int(round(values[internal])) != _as_int(row[external], external):
            raise AssertionError(f"{model} count mismatch: {external}")
    for field in POINT_FIELDS:
        expected = _as_float(row[field], field)
        actual = float(point[field])
        if abs(actual - expected) > 1e-12:
            raise AssertionError(
                f"{model} point mismatch for {field}: {actual} != {expected}")
    return {field: float(point[field]) for field in BOOTSTRAP_METRICS}


def bootstrap_models(
        matrices, *, replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED, batch=BOOTSTRAP_BATCH):
    """Paired image bootstrap using identical draws for every model."""
    if type(replicates) is not int or replicates < 100:
        raise ValueError("bootstrap replicates must be an integer >= 100")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    shapes = {matrix.shape for matrix in matrices.values()}
    if len(shapes) != 1:
        raise ValueError("paired bootstrap matrices must have one shape")
    n_images, width = next(iter(shapes))
    if n_images < 2 or width != len(CONTRIBUTION_FIELDS):
        raise ValueError("invalid paired bootstrap input")
    rng = np.random.default_rng(seed)
    samples = {
        model: {
            metric: np.empty(replicates, dtype=np.float64)
            for metric in BOOTSTRAP_METRICS
        }
        for model in matrices
    }
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        indices = rng.integers(
            0, n_images, size=(stop - start, n_images), endpoint=False)
        for model, matrix in matrices.items():
            totals = matrix[indices].sum(axis=1)
            metrics = metrics_from_totals(totals)
            for metric in BOOTSTRAP_METRICS:
                samples[model][metric][start:stop] = metrics[metric]
    return samples


def _interval_rows(point_estimates, samples):
    rows = []
    for selection in SELECTIONS:
        model = selection["model"]
        for metric in BOOTSTRAP_METRICS:
            values = samples[model][metric]
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite bootstrap value: {model} {metric}")
            lower, upper = np.quantile(values, (0.025, 0.975))
            rows.append(dict(
                dataset="mscoco",
                model=model,
                selected_seed=selection["seed"],
                metric=metric,
                estimate=point_estimates[model][metric],
                bootstrap_mean=float(values.mean()),
                bootstrap_standard_error=float(values.std(ddof=1)),
                ci_method="image-level percentile bootstrap",
                ci_level=0.95,
                ci_lower=float(lower),
                ci_upper=float(upper),
                n_images=5000,
                n_bootstrap=len(values),
                bootstrap_seed=BOOTSTRAP_SEED,
            ))
    return rows


def _pairwise_rows(point_estimates, samples):
    contrasts = (
        ("yolo11n", "yolov8n"),
        ("yolo11s", "yolo11n"),
        ("yolo11s", "yolov8n"),
    )
    rows = []
    for first, second in contrasts:
        for metric in ("VC", "mAC", "standard_F1", "standard_LRP"):
            values = samples[first][metric] - samples[second][metric]
            lower, upper = np.quantile(values, (0.025, 0.975))
            rows.append(dict(
                dataset="mscoco",
                contrast=f"{first} - {second}",
                metric=metric,
                estimate_difference=(
                    point_estimates[first][metric]
                    - point_estimates[second][metric]),
                bootstrap_mean_difference=float(values.mean()),
                bootstrap_standard_error=float(values.std(ddof=1)),
                ci_method="paired image-level percentile bootstrap",
                ci_level=0.95,
                ci_lower=float(lower),
                ci_upper=float(upper),
                probability_difference_gt_zero=float((values > 0).mean()),
                n_images=5000,
                n_bootstrap=len(values),
                bootstrap_seed=BOOTSTRAP_SEED,
            ))
    return rows


def _paper_rows(matrix_rows, intervals, visdrone_rows):
    lookup = {
        (row["model"], row["metric"]): row for row in intervals
    }
    rows = []
    for selection in SELECTIONS:
        model = selection["model"]
        source = matrix_rows[model]
        def ci(metric):
            row = lookup[(model, metric)]
            return (
                f"{row['estimate']:.4f} "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            )
        rows.append(dict(
            dataset="MSCOCO2017",
            model=model,
            uncertainty="fixed checkpoint; image-bootstrap 95% CI",
            checkpoint=f"seed {selection['seed']}",
            VC=ci("VC"),
            mAC=ci("mAC"),
            standard_F1=ci("standard_F1"),
            standard_LRP=ci("standard_LRP"),
            diagnostic_AP50_95=f"{_as_float(source['diagnostic_AP50_95'], 'AP'):.4f}",
        ))
    for source in sorted(visdrone_rows, key=lambda row: row["model"]):
        def pm(field):
            return (
                f"{_as_float(source[field + '_mean'], field):.4f} ± "
                f"{_as_float(source[field + '_std'], field):.4f}"
            )
        rows.append(dict(
            dataset="VisDrone2019-DET",
            model=source["model"],
            uncertainty="five training seeds; sample standard deviation",
            checkpoint="seeds 0-4",
            VC=pm("VC"),
            mAC=pm("mAC"),
            standard_F1=pm("standard_F1"),
            standard_LRP=pm("standard_LRP"),
            diagnostic_AP50_95=pm("diagnostic_AP50_95"),
        ))
    return rows


def _write_plots(output, intervals, theta_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vc_rows = [row for row in intervals if row["metric"] == "VC"]
    labels = [row["model"] for row in vc_rows]
    centers = np.asarray([row["estimate"] for row in vc_rows])
    lower = centers - np.asarray([row["ci_lower"] for row in vc_rows])
    upper = np.asarray([row["ci_upper"] for row in vc_rows]) - centers
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.errorbar(
        np.arange(len(labels)), centers, yerr=np.vstack([lower, upper]),
        fmt="o", color="#1f5a94", capsize=5, linewidth=1.6,
    )
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel("VC at theta=0.25")
    axis.set_title("MSCOCO fixed checkpoints: image-bootstrap 95% CI")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "coco_fixed_vc_bootstrap.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for model in ("yolov8n", "yolo11n", "yolo11s"):
        rows = sorted(
            (row for row in theta_rows if row["model"] == model),
            key=lambda row: _as_float(
                row["confidence_threshold"], "confidence_threshold"),
        )
        axis.plot(
            [_as_float(row["confidence_threshold"], "theta") for row in rows],
            [_as_float(row["VC"], "VC") for row in rows],
            marker="o", markersize=3, label=model,
        )
    axis.axvline(PRIMARY_THETA, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Confidence threshold theta")
    axis.set_ylabel("VC")
    axis.set_title("MSCOCO fixed-checkpoint theta sensitivity")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "coco_fixed_theta_sensitivity.png", dpi=220)
    plt.close(figure)


def _markdown_summary(paper_rows, pairwise_rows):
    lines = [
        "# Fixed-checkpoint COCO and five-seed VisDrone summary",
        "",
        "COCO uses one predeclared checkpoint per architecture and 10,000 "
        "paired image-level bootstrap resamples. Brackets are percentile "
        "95% confidence intervals. VisDrone reports mean ± sample standard "
        "deviation over five training seeds.",
        "",
        "| Dataset | Model | VC | mAC | F1 | LRP | Diagnostic AP50:95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in paper_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['VC']} | "
            f"{row['mAC']} | {row['standard_F1']} | "
            f"{row['standard_LRP']} | {row['diagnostic_AP50_95']} |"
        )
    lines.extend([
        "",
        "## Paired COCO VC differences",
        "",
        "| Contrast | Difference | 95% CI | P(difference > 0) |",
        "|---|---:|---:|---:|",
    ])
    for row in pairwise_rows:
        if row["metric"] != "VC":
            continue
        lines.append(
            f"| {row['contrast']} | {row['estimate_difference']:.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
            f"{row['probability_difference_gt_zero']:.4f} |"
        )
    lines.extend([
        "",
        "Diagnostic AP is taken from the shared post-NMS cache and is a point "
        "estimate. It is not presented as an official COCOeval or VisDrone AP.",
        "",
    ])
    return "\n".join(lines)


def run_fixed_checkpoint_analysis(
        *, matrix_root=MATRIX_ROOT, output=DEFAULT_OUTPUT,
        replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    matrix_root, matrix_hashes = _matrix_inputs(matrix_root)
    if replicates != BOOTSTRAP_REPLICATES or seed != BOOTSTRAP_SEED:
        raise ValueError("paper bootstrap settings are frozen")
    audits = [_validate_training_args(row) for row in SELECTIONS]
    _validate_manifest_selection(matrix_root, audits)
    matrix_rows = _selected_matrix_rows(matrix_root)

    selected_input_hashes = {}
    matrices = {}
    point_estimates = {}
    ordered_ids = None
    for selection in SELECTIONS:
        model = selection["model"]
        run_dir = Path(matrix_rows[model]["output"]).resolve()
        analysis_dir = run_dir / "analysis"
        gt_path = analysis_dir / "vc_gt_primary.csv"
        pair_path = analysis_dir / "vc_pairs_primary.csv"
        prediction_path = run_dir / "filtered_predictions.json"
        required = (gt_path, pair_path, prediction_path)
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"selected run is incomplete: {run_dir}")
        selected_input_hashes[model] = {
            str(path): _file_hash(path) for path in required
        }
        print(f"[{model}] loading fixed prediction cache...", flush=True)
        document = _read_json(prediction_path)
        image_ids = [image["image_id"] for image in document["images"]]
        if len(image_ids) != 5000:
            raise ValueError(f"{model} must contain 5000 COCO images")
        current_order = sorted(image_ids)
        if ordered_ids is None:
            ordered_ids = current_order
        elif current_order != ordered_ids:
            raise ValueError("selected COCO runs do not share image IDs")
        contributions = _new_contributions(image_ids)
        _region_contributions(
            contributions, _read_csv(gt_path), _read_csv(pair_path))
        _matching_contributions(contributions, document)
        matrix = _contribution_matrix(contributions, ordered_ids)
        point_estimates[model] = _validate_point_estimate(
            model, matrix, matrix_rows[model])
        matrices[model] = matrix
        print(f"[{model}] exact point-estimate audit passed.", flush=True)
        del document, contributions

    plan = dict(
        analysis_version=ANALYSIS_VERSION,
        vc_version=VC_VERSION,
        source_matrix_fingerprint=EXPECTED_MATRIX_FINGERPRINT,
        matrix_hashes=matrix_hashes,
        selected_inputs=selected_input_hashes,
        selections=audits,
        primary_theta=PRIMARY_THETA,
        matching_iou=IOU_THRESHOLD,
        bootstrap=dict(
            unit="validation image",
            scheme="nonparametric paired resampling with replacement",
            n_images=len(ordered_ids),
            replicates=replicates,
            seed=seed,
            interval="2.5th and 97.5th percentiles",
        ),
        source_sha256=_file_hash(Path(__file__)),
    )
    plan["analysis_fingerprint"] = _fingerprint(plan)
    output = Path(output).resolve()
    complete_path = output / "analysis_complete.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("analysis_fingerprint") != plan["analysis_fingerprint"]:
            raise ValueError("completed fixed-checkpoint output is stale")
        print(f"SKIP: fixed-checkpoint analysis already complete: {output}")
        return output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"incomplete output already exists: {output}")

    print(
        f"Running {replicates} paired image bootstrap replicates...",
        flush=True,
    )
    samples = bootstrap_models(matrices, replicates=replicates, seed=seed)
    intervals = _interval_rows(point_estimates, samples)
    pairwise = _pairwise_rows(point_estimates, samples)

    threshold_rows = _read_csv(matrix_root / "matrix_thresholds.csv")
    chosen = {(row["model"], str(row["selected_seed"])) for row in audits}
    theta_rows = [
        row for row in threshold_rows
        if row["dataset"] == "mscoco"
        and (row["model"], row["seed"]) in chosen
    ]
    if len(theta_rows) != 3 * 19:
        raise ValueError("selected theta grid is incomplete")
    visdrone_rows = [
        row for row in _read_csv(matrix_root / "aggregate_primary.csv")
        if row["dataset"] == "visdrone"
    ]
    if len(visdrone_rows) != 3 or any(
            _as_int(row["n_seeds"], "n_seeds") != 5
            for row in visdrone_rows):
        raise ValueError("VisDrone five-seed aggregate is incomplete")
    paper_rows = _paper_rows(matrix_rows, intervals, visdrone_rows)

    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(temporary / "protocol.json", plan)
        _write_csv(
            temporary / "coco_fixed_bootstrap.csv", intervals,
            list(intervals[0]),
        )
        _write_csv(
            temporary / "coco_fixed_pairwise.csv", pairwise,
            list(pairwise[0]),
        )
        _write_csv(
            temporary / "coco_fixed_theta.csv", theta_rows,
            list(theta_rows[0]),
        )
        _write_csv(
            temporary / "visdrone_five_seed.csv", visdrone_rows,
            list(visdrone_rows[0]),
        )
        _write_csv(
            temporary / "paper_main_table.csv", paper_rows,
            list(paper_rows[0]),
        )
        (temporary / "README.md").write_text(
            _markdown_summary(paper_rows, pairwise), encoding="utf-8")
        _write_plots(temporary, intervals, theta_rows)
        output_files = sorted(
            path for path in temporary.iterdir() if path.is_file())
        complete = dict(
            status="complete",
            **plan,
            result_files={
                path.name: _file_hash(path) for path in output_files
            },
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(temporary / "analysis_complete.json", complete)
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps(dict(
        status="complete",
        output=str(output),
        analysis_fingerprint=plan["analysis_fingerprint"],
        selected_coco_models=len(SELECTIONS),
        bootstrap_replicates=replicates,
        retained_visdrone_seed_runs=15,
    ), ensure_ascii=False), flush=True)
    print(
        "SUCCESS: fixed-checkpoint COCO bootstrap and VisDrone five-seed "
        f"analysis complete: {output}",
        flush=True,
    )
    return output