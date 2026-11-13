"""Merge completed RT-DETR runs into the frozen VC v1 paper reports.

This is an offline reporting layer. It performs no training and no inference.
The original YOLO-only paper output remains unchanged.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
from uuid import uuid4

import numpy as np

from .case_analysis import _file_hash
from .fixed_checkpoint_analysis import (
    BOOTSTRAP_METRICS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    _contribution_matrix,
    _matching_contributions,
    _new_contributions,
    _region_contributions,
    _validate_point_estimate,
    bootstrap_models,
)
from .offline import PROJECT_ROOT
from .vc_v1 import PRIMARY_THETA, THETA_GRID, VC_VERSION


REPORT_VERSION = "vc-v1-rtdetr-paper-results-0.1.0"
MODEL = "rtdetr_l"
MODEL_LABEL = "RT-DETR-L"
MODEL_ORDER = ("yolov8n", "yolo11n", "yolo11s", MODEL)
MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolo11n": "YOLO11n",
    "yolo11s": "YOLO11s",
    MODEL: MODEL_LABEL,
}
DATASET_LABELS = {"mscoco": "MS COCO", "visdrone": "VisDrone"}

RTDETR_ROOT = PROJECT_ROOT / "results" / "rtdetr"
YOLO_PAPER_ROOT = PROJECT_ROOT / "results" / "paper_results_v1"
YOLO_THETA = PROJECT_ROOT / "results" / "gaming_main" / "theta_sensitivity.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "paper_tables"

PRIMARY_FIELDS = (
    "VC", "mAC", "mean_responsibility_coverage", "mean_spatial_precision",
    "mean_local_VC", "orphan_rate", "standard_precision", "standard_recall",
    "standard_F1", "standard_LRP", "diagnostic_AP50", "diagnostic_AP75",
    "diagnostic_AP50_95", "diagnostic_AR50_95", "RecallAC025",
    "RecallAC050", "RecallAC075", "best_VC_theta", "best_F1_theta",
    "best_LRP_theta",
)
THRESHOLD_FIELDS = (
    "VC", "mAC", "mean_responsibility_coverage", "mean_spatial_precision",
    "mean_local_VC", "orphan_rate", "standard_F1", "standard_LRP",
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fingerprint(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}")
    return value


def _threshold_rows(path: Path):
    rows = _read_csv(path)
    numeric = set(rows[0]) - {"metric_version"}
    converted = []
    for row in rows:
        item = {"metric_version": row["metric_version"]}
        for field in numeric:
            value = _number(row, field)
            item[field] = int(value) if field.startswith("n_") or field in {
                "assigned_pred_count", "orphan_pred_count", "standard_TP",
                "standard_FP", "standard_FN",
            } else value
        converted.append(item)
    if [row["confidence_threshold"] for row in converted] != list(THETA_GRID):
        raise ValueError(f"incomplete theta grid: {path}")
    for row in converted:
        if row["assigned_pred_count"] + row["orphan_pred_count"] != row["n_pred"]:
            raise AssertionError("RT-DETR assignment counts do not conserve predictions")
        if row["standard_TP"] + row["standard_FP"] != row["n_pred"]:
            raise AssertionError("RT-DETR matching counts do not conserve predictions")
        if row["standard_TP"] + row["standard_FN"] != row["n_gt"]:
            raise AssertionError("RT-DETR matching counts do not conserve GT")
    return converted


def _recall(path: Path):
    return {
        float(row["coverage_threshold"]): float(row["RecallACtau"])
        for row in _read_csv(path)
    }


def _completed_sources(root: Path):
    suite_path = root / "suite_complete.json"
    if not suite_path.is_file():
        raise FileNotFoundError(suite_path)
    suite = _read_json(suite_path)
    if suite.get("status") != "complete" or suite.get("completed_runs") != 6:
        raise ValueError("RT-DETR suite is not complete")
    expected = {("mscoco", 0), *(("visdrone", seed) for seed in range(5))}
    found = set()
    sources = []
    for result in suite["results"]:
        run = Path(result["output"]).resolve()
        complete_path = run / "run_complete.json"
        analysis_complete = run / "analysis" / "analysis_complete.json"
        required = (
            complete_path,
            run / "run_manifest.json",
            run / "cache_complete.json",
            analysis_complete,
            run / "analysis" / "vc_thresholds.csv",
            run / "analysis" / "recall_ac_tau.csv",
            run / "analysis" / "diagnostic_ap_summary.json",
        )
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"incomplete RT-DETR result: {run}")
        complete = _read_json(complete_path)
        if complete.get("status") != "complete":
            raise ValueError(f"RT-DETR run is not complete: {run}")
        if complete.get("max_det_reached_images") != 0:
            raise ValueError(f"RT-DETR max_det truncation detected: {run}")
        if complete.get("invalid_prediction_count") != 0:
            raise ValueError(f"invalid RT-DETR predictions detected: {run}")
        kind = "mscoco" if complete["dataset"] == "MSCOCO2017" else "visdrone"
        seed = int(complete["training_seed"])
        found.add((kind, seed))
        sources.append(dict(kind=kind, seed=seed, run=run, complete=complete))
    if found != expected:
        raise ValueError(f"unexpected RT-DETR run set: {found}")
    return suite_path, suite, sorted(sources, key=lambda row: (row["kind"], row["seed"]))


def _run_tables(sources):
    run_rows = []
    threshold_rows = []
    for source in sources:
        analysis = source["run"] / "analysis"
        thresholds = _threshold_rows(analysis / "vc_thresholds.csv")
        primary = next(
            row for row in thresholds
            if row["confidence_threshold"] == PRIMARY_THETA
        )
        if primary != source["complete"]["primary"]:
            raise ValueError(f"primary result mismatch: {source['run']}")
        ap = _read_json(analysis / "diagnostic_ap_summary.json")
        recall = _recall(analysis / "recall_ac_tau.csv")
        best_vc = max(thresholds, key=lambda row: row["VC"])
        best_f1 = max(thresholds, key=lambda row: row["standard_F1"])
        best_lrp = min(thresholds, key=lambda row: row["standard_LRP"])
        run_row = dict(
            dataset=source["kind"], model=MODEL, seed=source["seed"],
            model_id=source["complete"]["model_id"], output=str(source["run"]),
            diagnostic_AP50=float(ap["AP50"]),
            diagnostic_AP75=float(ap["AP75"]),
            diagnostic_AP50_95=float(ap["AP50_95"]),
            diagnostic_AR50_95=float(ap["AR50_95"]),
            RecallAC025=recall[0.25], RecallAC050=recall[0.5],
            RecallAC075=recall[0.75],
            best_VC_theta=best_vc["confidence_threshold"],
            best_F1_theta=best_f1["confidence_threshold"],
            best_LRP_theta=best_lrp["confidence_threshold"],
            **primary,
        )
        run_rows.append(run_row)
        threshold_rows.extend(dict(
            dataset=source["kind"], model=MODEL, seed=source["seed"],
            model_id=source["complete"]["model_id"], **row,
        ) for row in thresholds)
    if len(run_rows) != 6 or len(threshold_rows) != 6 * len(THETA_GRID):
        raise AssertionError("RT-DETR reporting row count changed")
    return run_rows, threshold_rows


def _aggregate(rows, group_fields, metric_fields):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        item = dict(zip(group_fields, key))
        item["n_runs"] = len(group)
        for field in metric_fields:
            values = [float(row[field]) for row in group]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _bootstrap_coco(coco_source, coco_run_row):
    run = coco_source["run"]
    analysis = run / "analysis"
    gt_path = analysis / "vc_gt_primary.csv"
    pair_path = analysis / "vc_pairs_primary.csv"
    prediction_path = run / "filtered_predictions.json"
    print("[RT-DETR-L] loading COCO cache for 10,000 image-bootstrap replicates...", flush=True)
    document = _read_json(prediction_path)
    image_ids = [image["image_id"] for image in document["images"]]
    if len(image_ids) != 5000 or len(set(image_ids)) != 5000:
        raise ValueError("RT-DETR COCO cache must contain 5000 unique images")
    ordered_ids = sorted(image_ids)
    contributions = _new_contributions(image_ids)
    _region_contributions(contributions, _read_csv(gt_path), _read_csv(pair_path))
    _matching_contributions(contributions, document)
    matrix = _contribution_matrix(contributions, ordered_ids)
    point = _validate_point_estimate(MODEL, matrix, coco_run_row)
    del document, contributions
    samples = bootstrap_models(
        {MODEL: matrix}, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    rows = []
    for metric in BOOTSTRAP_METRICS:
        values = samples[MODEL][metric]
        lower, upper = np.quantile(values, (0.025, 0.975))
        rows.append(dict(
            dataset="mscoco", model=MODEL, selected_seed=0, metric=metric,
            estimate=point[metric], bootstrap_mean=float(values.mean()),
            bootstrap_standard_error=float(values.std(ddof=1)),
            ci_method="image-level percentile bootstrap", ci_level=0.95,
            ci_lower=float(lower), ci_upper=float(upper), n_images=5000,
            n_bootstrap=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        ))
    return rows


def _main_table(yolo_rows, run_rows, aggregate_primary, bootstrap_rows):
    lookup = {row["metric"]: row for row in bootstrap_rows}
    coco = next(row for row in run_rows if row["dataset"] == "mscoco")
    vis = next(row for row in aggregate_primary if row["dataset"] == "visdrone")

    def ci(metric):
        row = lookup[metric]
        return f"{row['estimate']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"

    def pm(metric):
        return f"{vis[metric + '_mean']:.4f} ± {vis[metric + '_std']:.4f}"

    combined = list(yolo_rows)
    combined.append(dict(
        dataset="MSCOCO2017", model=MODEL,
        uncertainty="fixed checkpoint; image-bootstrap 95% CI",
        checkpoint="seed 0", VC=ci("VC"), mAC=ci("mAC"),
        standard_F1=ci("standard_F1"), standard_LRP=ci("standard_LRP"),
        diagnostic_AP50_95=f"{coco['diagnostic_AP50_95']:.4f}",
    ))
    combined.append(dict(
        dataset="VisDrone2019-DET", model=MODEL,
        uncertainty="five training seeds; sample standard deviation",
        checkpoint="seeds 0-4", VC=pm("VC"), mAC=pm("mAC"),
        standard_F1=pm("standard_F1"), standard_LRP=pm("standard_LRP"),
        diagnostic_AP50_95=pm("diagnostic_AP50_95"),
    ))
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    combined.sort(key=lambda row: (
        0 if row["dataset"] == "MSCOCO2017" else 1, order[row["model"]]))
    if len(combined) != 8:
        raise AssertionError("four-model main table must contain eight rows")
    return combined


def _combined_theta(yolo_rows, rtdetr_aggregate):
    rtdetr_rows = []
    for row in rtdetr_aggregate:
        rtdetr_rows.append(dict(
            dataset=row["dataset"], model=MODEL,
            confidence_threshold=row["confidence_threshold"],
            uncertainty=(
                "fixed checkpoint point estimate" if row["dataset"] == "mscoco"
                else "five training seeds; sample standard deviation"),
            n_runs=row["n_runs"],
            VC_mean=row["VC_mean"], VC_std=row["VC_std"],
            mAC_mean=row["mAC_mean"], mAC_std=row["mAC_std"],
            standard_F1_mean=row["standard_F1_mean"],
            standard_F1_std=row["standard_F1_std"],
            standard_LRP_mean=row["standard_LRP_mean"],
            standard_LRP_std=row["standard_LRP_std"],
            orphan_rate_mean=row["orphan_rate_mean"],
            orphan_rate_std=row["orphan_rate_std"],
        ))
    combined = [dict(row) for row in yolo_rows] + rtdetr_rows
    if len(combined) != 8 * len(THETA_GRID):
        raise AssertionError("four-model theta table must contain 152 rows")
    return sorted(combined, key=lambda row: (
        row["dataset"], MODEL_ORDER.index(row["model"]),
        float(row["confidence_threshold"])))


def _theta_optima(theta_rows):
    rows = []
    for dataset in ("mscoco", "visdrone"):
        for model in MODEL_ORDER:
            curve = [
                row for row in theta_rows
                if row["dataset"] == dataset and row["model"] == model
            ]
            if len(curve) != len(THETA_GRID):
                raise ValueError(f"incomplete theta curve: {dataset}/{model}")
            primary = next(
                row for row in curve
                if float(row["confidence_threshold"]) == PRIMARY_THETA)
            best_vc = max(curve, key=lambda row: float(row["VC_mean"]))
            best_f1 = max(curve, key=lambda row: float(row["standard_F1_mean"]))
            best_lrp = min(curve, key=lambda row: float(row["standard_LRP_mean"]))
            best_value = float(best_vc["VC_mean"])
            primary_value = float(primary["VC_mean"])
            item = dict(
                dataset=DATASET_LABELS[dataset], model=MODEL_LABELS[model],
                best_VC_theta=float(best_vc["confidence_threshold"]),
                best_VC=best_value,
                relative_gap_from_best=(best_value - primary_value) / best_value,
                best_F1_theta=float(best_f1["confidence_threshold"]),
                best_LRP_theta=float(best_lrp["confidence_threshold"]),
                statistical_unit=primary["uncertainty"],
            )
            item["VC_at_theta_0.25"] = primary_value
            rows.append(item)
    return rows


def _plot_theta(theta_rows, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = (
        ("VC_mean", "VC", "#0072B2", "-"),
        ("mAC_mean", "mAC", "#D55E00", "--"),
        ("standard_F1_mean", "F1", "#009E73", "-."),
        ("one_minus_LRP", "1-LRP", "#CC79A7", ":"),
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.4,
        "axes.titlesize": 8.0, "axes.labelsize": 8.0,
        "legend.fontsize": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 400,
    })
    figure, axes = plt.subplots(2, 4, figsize=(7.16, 4.55), sharex=True, sharey=True)
    panel = 0
    for row_index, dataset in enumerate(("mscoco", "visdrone")):
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            curve = sorted(
                (row for row in theta_rows
                 if row["dataset"] == dataset and row["model"] == model),
                key=lambda row: float(row["confidence_threshold"]),
            )
            x = [float(row["confidence_threshold"]) for row in curve]
            for field, label, colour, linestyle in fields:
                y = [
                    1.0 - float(row["standard_LRP_mean"])
                    if field == "one_minus_LRP" else float(row[field])
                    for row in curve
                ]
                axis.plot(x, y, color=colour, linestyle=linestyle,
                          marker="o", markersize=1.8, linewidth=1.2, label=label)
            axis.axvline(PRIMARY_THETA, color="#444444", linestyle=(0, (3, 2)),
                         linewidth=0.75)
            axis.set_title(
                f"({chr(ord('a') + panel)}) {DATASET_LABELS[dataset]}\n{MODEL_LABELS[model]}",
                loc="left", pad=3)
            panel += 1
            axis.set_xlim(0.05, 0.95)
            axis.set_ylim(0.0, 0.85)
            axis.set_xticks([0.05, 0.25, 0.45, 0.65, 0.85])
            axis.grid(axis="y", alpha=0.22, linewidth=0.55)
    for axis in axes[1, :]:
        axis.set_xlabel("Confidence θ")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, 1.01))
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.9, w_pad=0.65)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"fig_theta_sensitivity_four_models.{extension}",
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _summary_text(main_rows, optima, bootstrap_rows, vis_primary):
    coco = next(row for row in main_rows
                if row["dataset"] == "MSCOCO2017" and row["model"] == MODEL)
    vis = next(row for row in main_rows
               if row["dataset"] == "VisDrone2019-DET" and row["model"] == MODEL)
    coco_theta = next(row for row in optima
                      if row["dataset"] == "MS COCO" and row["model"] == MODEL_LABEL)
    vis_theta = next(row for row in optima
                     if row["dataset"] == "VisDrone" and row["model"] == MODEL_LABEL)
    return f"""# Four-model VC v1 result update

## Experimental setup draft

Experiments were conducted on MS COCO 2017 and VisDrone2019-DET using their official training and validation splits. All detectors were trained on the corresponding training sets and evaluated on the complete validation sets, comprising 5,000 images for MS COCO and 548 images for VisDrone. Four detector architectures were considered: YOLOv8n, YOLO11n, YOLO11s, and RT-DETR-L. For MS COCO, uncertainty for each reported trained model was estimated using 10,000 image-level bootstrap replicates over the complete validation set. For VisDrone, each architecture was trained with five independent random seeds, and the results are reported as the mean and sample standard deviation. VC, mAC, F1, and LRP were evaluated at the common primary confidence threshold $\\theta=0.25$ and matching IoU 0.5. To examine operating-point sensitivity, all metrics were additionally reported at 19 confidence thresholds from 0.05 to 0.95. YOLO uses its standard post-NMS detections, whereas RT-DETR uses its native set-based postprocessing.

## RT-DETR-L primary results

On MS COCO, RT-DETR-L obtained VC {coco['VC']}, mAC {coco['mAC']}, F1 {coco['standard_F1']}, and LRP {coco['standard_LRP']}. On VisDrone, its five-seed results were VC {vis['VC']}, mAC {vis['mAC']}, F1 {vis['standard_F1']}, and LRP {vis['standard_LRP']}. The VisDrone VC mean and sample standard deviation were {vis_primary['VC_mean']:.4f} and {vis_primary['VC_std']:.4f}, respectively.

The VC-maximizing RT-DETR-L threshold was {coco_theta['best_VC_theta']:.2f} on MS COCO and {vis_theta['best_VC_theta']:.2f} on VisDrone. At the common $\\theta=0.25$ operating point, the relative gaps from the per-dataset VC maxima were {100*coco_theta['relative_gap_from_best']:.2f}% and {100*vis_theta['relative_gap_from_best']:.2f}%. This shift toward higher thresholds shows that confidence calibration differs across detector families. We therefore retain $\\theta=0.25$ only as the predeclared common operating point for the primary comparison, without claiming that it is optimal for every architecture, and report the complete curves and model-specific optima as sensitivity results.

## Reporting notes

- `table_main_performance_four_models.csv` is the updated eight-row primary table.
- `table_theta_optima_four_models.csv` and the corresponding figure include all four architectures on both datasets.
- MS COCO uncertainty is estimated by image-level bootstrap over the complete validation set for each reported trained model.
- VisDrone uncertainty reflects variation across five independent training seeds and is reported as the sample standard deviation.
- `diagnostic_AP50_95` is computed from the shared prediction cache and is retained for audit only. Use official dataset-evaluator AP as the paper's standard AP result.
- The existing controlled Gaming analysis remains a separate three-YOLO-architecture experiment until RT-DETR Gaming is explicitly added.
"""


def build_rtdetr_paper_results(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    suite_path, suite, sources = _completed_sources(RTDETR_ROOT)
    yolo_main_path = YOLO_PAPER_ROOT / "table_main_performance.csv"
    if not yolo_main_path.is_file() or not YOLO_THETA.is_file():
        raise FileNotFoundError("frozen YOLO paper results are incomplete")
    source_paths = [suite_path, yolo_main_path, YOLO_THETA]
    for source in sources:
        run = source["run"]
        source_paths.extend((
            run / "run_complete.json", run / "run_manifest.json",
            run / "cache_complete.json", run / "analysis" / "analysis_complete.json",
            run / "analysis" / "vc_thresholds.csv",
            run / "analysis" / "recall_ac_tau.csv",
            run / "analysis" / "diagnostic_ap_summary.json",
        ))
    coco_source = next(row for row in sources if row["kind"] == "mscoco")
    source_paths.extend((
        coco_source["run"] / "filtered_predictions.json",
        coco_source["run"] / "analysis" / "vc_gt_primary.csv",
        coco_source["run"] / "analysis" / "vc_pairs_primary.csv",
    ))
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): _file_hash(path)
                     for path in source_paths}
    plan = dict(
        report_version=REPORT_VERSION, vc_version=VC_VERSION,
        primary_theta=PRIMARY_THETA, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED, source_files=source_hashes,
        source_suite_fingerprint=suite["suite_fingerprint"],
        source_sha256=_file_hash(Path(__file__)),
    )
    plan["report_fingerprint"] = _fingerprint(plan)
    complete_path = output / "report_complete.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("report_fingerprint") != plan["report_fingerprint"]:
            raise ValueError("completed RT-DETR paper report is stale")
        print(f"SKIP: RT-DETR paper results already complete: {output}")
        return output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"incomplete output directory already exists: {output}")

    run_rows, threshold_rows = _run_tables(sources)
    aggregate_primary = _aggregate(
        run_rows, ("dataset", "model"), PRIMARY_FIELDS)
    aggregate_thresholds = _aggregate(
        threshold_rows, ("dataset", "model", "confidence_threshold"),
        THRESHOLD_FIELDS)
    coco_run = next(row for row in run_rows if row["dataset"] == "mscoco")
    bootstrap_rows = _bootstrap_coco(coco_source, coco_run)
    main_rows = _main_table(
        _read_csv(yolo_main_path), run_rows, aggregate_primary, bootstrap_rows)
    theta_rows = _combined_theta(_read_csv(YOLO_THETA), aggregate_thresholds)
    optima = _theta_optima(theta_rows)
    vis_primary = next(row for row in aggregate_primary
                       if row["dataset"] == "visdrone")

    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_csv(temporary / "rtdetr_run_primary.csv", run_rows)
        _write_csv(temporary / "rtdetr_run_thresholds.csv", threshold_rows)
        _write_csv(temporary / "rtdetr_aggregate_primary.csv", aggregate_primary)
        _write_csv(temporary / "rtdetr_aggregate_thresholds.csv", aggregate_thresholds)
        _write_csv(temporary / "coco_rtdetr_bootstrap.csv", bootstrap_rows)
        _write_csv(temporary / "table_main_performance_four_models.csv", main_rows)
        _write_csv(temporary / "theta_sensitivity_four_models.csv", theta_rows)
        _write_csv(temporary / "table_theta_optima_four_models.csv", optima)
        _plot_theta(theta_rows, temporary)
        (temporary / "RESULTS_UPDATE.md").write_text(
            _summary_text(main_rows, optima, bootstrap_rows, vis_primary),
            encoding="utf-8")
        (temporary / "README.md").write_text(
            "# RT-DETR paper-results extension\n\n"
            "Offline merge of the completed RT-DETR-L suite with the frozen "
            "YOLO paper tables. No training or inference is performed. The "
            "original paper_results_v1 directory is unchanged.\n",
            encoding="utf-8")
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        complete = dict(
            status="complete", **plan, training=False, inference=False,
            row_counts=dict(
                rtdetr_runs=len(run_rows), rtdetr_thresholds=len(threshold_rows),
                aggregate_primary=len(aggregate_primary),
                aggregate_thresholds=len(aggregate_thresholds),
                main_table=len(main_rows), combined_theta=len(theta_rows),
                theta_optima=len(optima), bootstrap=len(bootstrap_rows)),
            output_files={path.name: _file_hash(path) for path in files},
            completed_at=datetime.now(timezone.utc).isoformat())
        _write_json(temporary / "report_complete.json", complete)
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps(dict(
        status="complete", output=str(output), four_model_rows=len(main_rows),
        theta_rows=len(theta_rows), bootstrap_replicates=BOOTSTRAP_REPLICATES,
        training=False, inference=False), ensure_ascii=False), flush=True)
    print(f"SUCCESS: four-model RT-DETR paper results complete: {output}", flush=True)
    return output
