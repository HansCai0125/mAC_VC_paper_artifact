"""Build uniform five-seed paper tables after the COCO RT-DETR rerun."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

from .case_analysis import _file_hash
from .offline import PROJECT_ROOT
from .rtdetr_paper_results import (
    DATASET_LABELS,
    MODEL,
    MODEL_LABELS,
    MODEL_ORDER,
    PRIMARY_FIELDS,
    THRESHOLD_FIELDS,
    _aggregate,
    _plot_theta,
    _recall,
    _theta_optima,
    _threshold_rows,
)
from .vc_v1 import PRIMARY_THETA, THETA_GRID, VC_VERSION


FIVE_SEED_REPORT_VERSION = "vc-v1-five-seed-paper-results-0.1.0"
YOLO_MATRIX = PROJECT_ROOT / "results" / "detection"
YOLO_COCO = (
    PROJECT_ROOT / "results" / "yolo_coco_5seed_eval_v1"
)
RTDETR_COCO = (
    PROJECT_ROOT / "results" / "rtdetr_coco_5seed_eval_v1"
)
RTDETR_OLD_REPORT = (
    PROJECT_ROOT / "results" / "paper_tables"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "paper_results_five_seed_v1"
)


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rtdetr_coco_sources(root):
    suite_path = Path(root) / "suite_complete.json"
    suite = _read_json(suite_path)
    if suite.get("status") != "complete" or suite.get("completed_runs") != 5:
        raise ValueError("five-seed RT-DETR COCO evaluation is incomplete")
    sources = []
    for result in suite["results"]:
        run = Path(result["output"]).resolve()
        complete = _read_json(run / "run_complete.json")
        if complete.get("dataset") != "MSCOCO2017":
            raise ValueError(f"unexpected RT-DETR dataset: {run}")
        if complete.get("max_det_reached_images") != 0:
            raise ValueError(f"RT-DETR max_det truncation detected: {run}")
        if complete.get("invalid_prediction_count") != 0:
            raise ValueError(f"invalid RT-DETR predictions detected: {run}")
        sources.append(dict(
            kind="mscoco",
            seed=int(complete["training_seed"]),
            run=run,
            complete=complete,
        ))
    if {row["seed"] for row in sources} != set(range(5)):
        raise ValueError("RT-DETR COCO results do not contain seeds 0--4")
    return suite_path, sorted(sources, key=lambda row: row["seed"])


def _yolo_coco_sources(root):
    suite_path = Path(root) / "suite_complete.json"
    suite = _read_json(suite_path)
    if suite.get("status") != "complete" or suite.get("completed_runs") != 15:
        raise ValueError("five-seed YOLO COCO evaluation is incomplete")
    sources = []
    for result in suite["results"]:
        run = Path(result["output"]).resolve()
        complete = _read_json(run / "run_complete.json")
        if complete.get("dataset") != "MSCOCO2017":
            raise ValueError(f"unexpected YOLO dataset: {run}")
        sources.append(dict(
            kind="mscoco",
            model=result["model"],
            seed=int(result["seed"]),
            run=run,
            complete=complete,
        ))
    expected = {
        (model, seed)
        for model in ("yolov8n", "yolo11n", "yolo11s")
        for seed in range(5)
    }
    if {(row["model"], row["seed"]) for row in sources} != expected:
        raise ValueError("YOLO COCO results do not contain 3 models x 5 seeds")
    return suite_path, sorted(
        sources, key=lambda row: (MODEL_ORDER.index(row["model"]), row["seed"]))


def _detector_run_tables(sources, default_model=None):
    run_rows = []
    threshold_rows = []
    for source in sources:
        analysis = source["run"] / "analysis"
        thresholds = _threshold_rows(analysis / "vc_thresholds.csv")
        primary = next(
            row for row in thresholds
            if row["confidence_threshold"] == PRIMARY_THETA)
        if primary != source["complete"]["primary"]:
            raise ValueError(f"primary result mismatch: {source['run']}")
        ap = _read_json(analysis / "diagnostic_ap_summary.json")
        recall = _recall(analysis / "recall_ac_tau.csv")
        best_vc = max(thresholds, key=lambda row: row["VC"])
        best_f1 = max(thresholds, key=lambda row: row["standard_F1"])
        best_lrp = min(thresholds, key=lambda row: row["standard_LRP"])
        model = source.get("model", default_model)
        if model is None:
            raise ValueError("detector source has no model identifier")
        run_rows.append(dict(
            dataset="mscoco",
            model=model,
            seed=source["seed"],
            model_id=source["complete"]["model_id"],
            output=str(source["run"]),
            diagnostic_AP50=float(ap["AP50"]),
            diagnostic_AP75=float(ap["AP75"]),
            diagnostic_AP50_95=float(ap["AP50_95"]),
            diagnostic_AR50_95=float(ap["AR50_95"]),
            RecallAC025=recall[0.25],
            RecallAC050=recall[0.5],
            RecallAC075=recall[0.75],
            best_VC_theta=best_vc["confidence_threshold"],
            best_F1_theta=best_f1["confidence_threshold"],
            best_LRP_theta=best_lrp["confidence_threshold"],
            **primary,
        ))
        threshold_rows.extend(dict(
            dataset="mscoco",
            model=model,
            seed=source["seed"],
            model_id=source["complete"]["model_id"],
            **row,
        ) for row in thresholds)
    if len(threshold_rows) != len(run_rows) * len(THETA_GRID):
        raise AssertionError("detector threshold row count changed")
    return run_rows, threshold_rows


def _main_rows(yolo_vis, yolo_coco, rtdetr_coco, rtdetr_vis):
    rows = []
    source_rows = [
        *yolo_coco,
        *yolo_vis,
        rtdetr_coco,
        rtdetr_vis,
    ]
    for row in source_rows:
        if int(row.get("n_seeds", row.get("n_runs", 0))) != 5:
            raise ValueError(
                f"main-table group is not five-seed: "
                f"{row['dataset']}/{row['model']}")
        rows.append(dict(
            dataset=DATASET_LABELS[row["dataset"]],
            model=MODEL_LABELS[row["model"]],
            n_seeds=5,
            mAC_mean=float(row["mAC_mean"]),
            mAC_std=float(row["mAC_std"]),
            VC_mean=float(row["VC_mean"]),
            VC_std=float(row["VC_std"]),
            F1_mean=float(row["standard_F1_mean"]),
            F1_std=float(row["standard_F1_std"]),
            LRP_mean=float(row["standard_LRP_mean"]),
            LRP_std=float(row["standard_LRP_std"]),
            diagnostic_AP50_95_mean=float(row["diagnostic_AP50_95_mean"]),
            diagnostic_AP50_95_std=float(row["diagnostic_AP50_95_std"]),
        ))
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    rows.sort(key=lambda row: (
        0 if row["dataset"] == "MS COCO" else 1,
        order[next(key for key, value in MODEL_LABELS.items()
                   if value == row["model"])],
    ))
    if len(rows) != 8:
        raise AssertionError("uniform five-seed table must contain eight rows")
    return rows


def _formatted_rows(rows):
    def pm(row, mean, std):
        return f"{row[mean]:.4f} ± {row[std]:.4f}"

    return [dict(
        dataset=row["dataset"],
        model=row["model"],
        n_seeds=5,
        mAC=pm(row, "mAC_mean", "mAC_std"),
        VC=pm(row, "VC_mean", "VC_std"),
        F1=pm(row, "F1_mean", "F1_std"),
        LRP=pm(row, "LRP_mean", "LRP_std"),
        diagnostic_AP50_95=pm(
            row, "diagnostic_AP50_95_mean", "diagnostic_AP50_95_std"),
    ) for row in rows]


def _normalise_theta(rows):
    output = []
    for row in rows:
        if int(row.get("n_seeds", row.get("n_runs", 0))) != 5:
            raise ValueError(
                f"theta group is not five-seed: "
                f"{row['dataset']}/{row['model']}")
        output.append(dict(
            dataset=row["dataset"],
            model=row["model"],
            confidence_threshold=float(row["confidence_threshold"]),
            uncertainty="five training seeds; sample standard deviation",
            n_runs=5,
            VC_mean=float(row["VC_mean"]),
            VC_std=float(row["VC_std"]),
            mAC_mean=float(row["mAC_mean"]),
            mAC_std=float(row["mAC_std"]),
            standard_F1_mean=float(row["standard_F1_mean"]),
            standard_F1_std=float(row["standard_F1_std"]),
            standard_LRP_mean=float(row["standard_LRP_mean"]),
            standard_LRP_std=float(row["standard_LRP_std"]),
            orphan_rate_mean=float(row["orphan_rate_mean"]),
            orphan_rate_std=float(row["orphan_rate_std"]),
        ))
    return output


def build_five_seed_paper_results(output=DEFAULT_OUTPUT):
    """Create the final uniform five-seed tables and threshold figure."""
    output = Path(output).resolve()
    yolo_primary_path = YOLO_MATRIX / "aggregate_primary.csv"
    yolo_theta_path = YOLO_MATRIX / "aggregate_thresholds.csv"
    vis_primary_path = RTDETR_OLD_REPORT / "rtdetr_aggregate_primary.csv"
    vis_theta_path = RTDETR_OLD_REPORT / "rtdetr_aggregate_thresholds.csv"
    yolo_suite_path, yolo_sources = _yolo_coco_sources(YOLO_COCO)
    yolo_run_rows, yolo_threshold_rows = _detector_run_tables(yolo_sources)
    yolo_coco_primary = _aggregate(
        yolo_run_rows, ("dataset", "model"), PRIMARY_FIELDS)
    yolo_coco_theta = _aggregate(
        yolo_threshold_rows,
        ("dataset", "model", "confidence_threshold"),
        THRESHOLD_FIELDS)
    suite_path, sources = _rtdetr_coco_sources(RTDETR_COCO)
    run_rows, threshold_rows = _detector_run_tables(sources, MODEL)
    rtdetr_coco_primary = _aggregate(
        run_rows, ("dataset", "model"), PRIMARY_FIELDS)[0]
    rtdetr_coco_theta = _aggregate(
        threshold_rows,
        ("dataset", "model", "confidence_threshold"),
        THRESHOLD_FIELDS,
    )
    rtdetr_vis_primary = next(
        row for row in _read_csv(vis_primary_path)
        if row["dataset"] == "visdrone")
    rtdetr_vis_theta = [
        row for row in _read_csv(vis_theta_path)
        if row["dataset"] == "visdrone"]
    yolo_vis_primary = [
        row for row in _read_csv(yolo_primary_path)
        if row["dataset"] == "visdrone"]
    yolo_vis_theta = [
        row for row in _read_csv(yolo_theta_path)
        if row["dataset"] == "visdrone"]
    main = _main_rows(
        yolo_vis_primary,
        yolo_coco_primary,
        rtdetr_coco_primary,
        rtdetr_vis_primary,
    )
    theta = _normalise_theta([
        *yolo_coco_theta,
        *yolo_vis_theta,
        *rtdetr_coco_theta,
        *rtdetr_vis_theta,
    ])
    if len(theta) != 8 * len(THETA_GRID):
        raise AssertionError("combined theta table must contain 152 rows")
    theta.sort(key=lambda row: (
        row["dataset"], MODEL_ORDER.index(row["model"]),
        row["confidence_threshold"]))
    optima = _theta_optima(theta)

    source_paths = (
        yolo_primary_path, yolo_theta_path, vis_primary_path, vis_theta_path,
        yolo_suite_path, suite_path,
    )
    plan = dict(
        report_version=FIVE_SEED_REPORT_VERSION,
        vc_version=VC_VERSION,
        source_sha256={
            str(path.relative_to(PROJECT_ROOT)): _file_hash(path)
            for path in source_paths
        },
    )
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    if output.exists():
        complete_path = output / "report_complete.json"
        if complete_path.is_file():
            existing = _read_json(complete_path)
            if existing.get("source_sha256") == plan["source_sha256"]:
                print(f"SKIP: uniform five-seed report already complete: {output}")
                return output
        raise ValueError(f"stale or incomplete report output exists: {output}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_csv(temporary / "table_main_five_seed_numeric.csv", main)
        _write_csv(temporary / "table_main_five_seed_formatted.csv",
                   _formatted_rows(main))
        _write_csv(temporary / "theta_sensitivity_four_models.csv", theta)
        _write_csv(temporary / "table_theta_optima_four_models.csv", optima)
        _plot_theta(theta, temporary)
        (temporary / "README.md").write_text(
            "# Uniform five-seed paper results\n\n"
            "All eight dataset/model rows use five independently trained "
            "checkpoints and report the sample mean and sample standard "
            "deviation. The diagnostic AP column is an internal audit value; "
            "use official evaluator AP in the paper.\n",
            encoding="utf-8",
        )
        complete = dict(
            status="complete",
            **plan,
            n_main_rows=len(main),
            n_theta_rows=len(theta),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        (temporary / "report_complete.json").write_text(
            json.dumps(complete, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(f"SUCCESS: uniform five-seed paper results complete: {output}")
    return output
