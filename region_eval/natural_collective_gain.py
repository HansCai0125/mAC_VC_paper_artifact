"""Natural collective-coverage analysis on saved detector predictions.

For every ground-truth box this module computes:

* AC: coverage from the union of all retained same-class predictions;
* Best-IoGT: the best single ground-truth-normalized IoA;
* collective gain: AC - Best-IoGT;
* ABO: best single-box IoU;
* best prediction-normalized IoA;
* deterministic marginal contributor counts.

The analysis is offline: it never trains a model or performs inference.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import io
import json
from math import fsum
from pathlib import Path
import statistics
from time import sleep
from uuid import uuid4

from .cache_registry import (
    MODEL_ORDER,
    REGISTRY_VERSION,
    CachedRun,
    main_natural_sources,
    registry_audit,
)
from .geometry import area, intersection, union_area
from .offline import PROJECT_ROOT
from .vc_v1 import PRIMARY_THETA, VC_VERSION


ANALYSIS_VERSION = "vc-v1-natural-collective-gain-0.1.0"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "collective_gain"
)
GAIN_EPSILON = 1e-9
SUBSTANTIAL_MARGINAL_FRACTION = 0.01
GAIN_THRESHOLDS = (0.0, 0.05, 0.10, 0.20)
EXCEEDANCE_GRID = tuple(
    [index / 200.0 for index in range(41)]
    + [0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
)
SIZE_BINS = (
    ("small", 0.0, 32.0 ** 2),
    ("medium", 32.0 ** 2, 96.0 ** 2),
    ("large", 96.0 ** 2, float("inf")),
)
MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolo11n": "YOLO11n",
    "yolo11s": "YOLO11s",
    "rtdetr_l": "RT-DETR-L",
}
DATASET_LABELS = {"mscoco": "MS COCO", "visdrone": "VisDrone"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_text(path, value, attempts=20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        for attempt in range(attempts):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                sleep(min(0.05 * (2 ** attempt), 2.0))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path, value):
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _atomic_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(fields or rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, "\ufeff" + buffer.getvalue())


def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def _mean(values):
    return fsum(values) / len(values) if values else None


def _fraction(rows, predicate):
    return _mean([int(predicate(row)) for row in rows])


def _size_label(gt_area):
    for label, lower, upper in SIZE_BINS:
        if lower <= gt_area < upper:
            return label
    raise AssertionError("unreachable size bin")


def _density_labels(images):
    """Balanced, deterministic within-dataset image-density terciles."""
    ranked = sorted((len(image["gts"]), str(image["image_id"])) for image in images)
    labels = {}
    names = ("low", "medium", "high")
    for rank, (count, image_id) in enumerate(ranked):
        group = min(2, (rank * 3) // max(1, len(ranked)))
        labels[image_id] = (names[group], count)
    return labels


def _rectangle_iou(first, second, overlap_area=None):
    overlap = intersection(first, second)
    if overlap is None:
        return 0.0
    overlap_area = area(overlap) if overlap_area is None else overlap_area
    denominator = area(first) + area(second) - overlap_area
    return overlap_area / denominator if denominator else 0.0


def _existing_gt_lookup(path):
    rows = _read_csv(path)
    output = {}
    for row in rows:
        key = (str(row["image_id"]), str(row["gt_id"]))
        if key in output:
            raise ValueError(f"duplicate existing per-GT row: {key}")
        output[key] = {
            "AC": float(row["original_AC"]),
            "BestIoGT": float(row["original_BestIoGT"]),
            "CollectiveGain": float(row["original_CollectiveGain"]),
            "local_VC": float(row["responsibility_dice"]),
            "assigned_pred_count": int(float(row["assigned_pred_count"])),
        }
    return output


def _gt_metrics(gt, predictions):
    gt_rect = tuple(float(value) for value in gt["xyxy"])
    gt_area = area(gt_rect)
    if gt_area <= 0:
        raise ValueError(f"degenerate GT: {gt['id']}")
    candidates = []
    for prediction in predictions:
        pred_rect = tuple(float(value) for value in prediction["xyxy"])
        overlap = intersection(gt_rect, pred_rect)
        if overlap is None:
            continue
        overlap_area = area(overlap)
        pred_area = area(pred_rect)
        candidates.append({
            "id": str(prediction["id"]),
            "score": float(prediction["score"]),
            "overlap": overlap,
            "overlap_area": overlap_area,
            "pred_area": pred_area,
            "IoU": _rectangle_iou(gt_rect, pred_rect, overlap_area),
            "pred_IoA": overlap_area / pred_area if pred_area else 0.0,
        })
    intersections = [row["overlap"] for row in candidates]
    covered_area = min(gt_area, union_area(intersections))
    best_area = max((row["overlap_area"] for row in candidates), default=0.0)
    ac = covered_area / gt_area
    best_iogt = best_area / gt_area
    ordered = sorted(
        candidates,
        key=lambda row: (-row["overlap_area"], -row["score"], row["id"]),
    )
    retained = []
    current_area = 0.0
    marginal = []
    marginal_ids = []
    for candidate in ordered:
        updated = union_area([*retained, candidate["overlap"]])
        gain = max(0.0, updated - current_area)
        if gain > GAIN_EPSILON * gt_area:
            retained.append(candidate["overlap"])
            current_area = updated
            marginal.append(gain / gt_area)
            marginal_ids.append(candidate["id"])
    collective_gain = max(0.0, ac - best_iogt)
    if collective_gain > GAIN_EPSILON and len(marginal) < 2:
        raise AssertionError("positive collective gain requires multiple contributors")
    return {
        "AC": ac,
        "BestIoGT": best_iogt,
        "CollectiveGain": collective_gain,
        "ABO": max((row["IoU"] for row in candidates), default=0.0),
        "BestPredIoA": max((row["pred_IoA"] for row in candidates), default=0.0),
        "intersecting_pred_count": len(candidates),
        "marginal_contributor_count": len(marginal),
        "substantial_marginal_contributor_count": sum(
            value >= SUBSTANTIAL_MARGINAL_FRACTION for value in marginal
        ),
        "marginal_pred_ids": ";".join(marginal_ids),
        "best_pred_id": ordered[0]["id"] if ordered else "",
    }


def analyze_document(document, *, theta=PRIMARY_THETA, existing_rows=None):
    """Return per-GT natural collective-gain rows for one cached run."""
    density = _density_labels(document["images"])
    results = []
    for image in document["images"]:
        image_id = str(image["image_id"])
        predictions_by_class = defaultdict(list)
        for prediction in image["predictions"]:
            if float(prediction["score"]) >= theta:
                predictions_by_class[int(prediction["class_id"])].append(prediction)
        density_label, image_gt_count = density[image_id]
        for gt in image["gts"]:
            class_id = int(gt["class_id"])
            metrics = _gt_metrics(gt, predictions_by_class[class_id])
            gt_area = area(tuple(float(value) for value in gt["xyxy"]))
            row = {
                "image_id": image_id,
                "gt_id": str(gt["id"]),
                "class_id": class_id,
                "gt_area": gt_area,
                "size_bin": _size_label(gt_area),
                "image_gt_count": image_gt_count,
                "density_bin": density_label,
                **metrics,
            }
            if existing_rows is not None:
                reference = existing_rows.pop((image_id, str(gt["id"])), None)
                if reference is None:
                    raise ValueError(f"existing analysis missed GT: {image_id}/{gt['id']}")
                for field in ("AC", "BestIoGT", "CollectiveGain"):
                    if abs(row[field] - reference[field]) > 1e-10:
                        raise ValueError(
                            f"{field} mismatch at {image_id}/{gt['id']}: "
                            f"{row[field]} != {reference[field]}"
                        )
                row["local_VC"] = reference["local_VC"]
                row["assigned_pred_count"] = reference["assigned_pred_count"]
            results.append(row)
    if existing_rows:
        raise ValueError(f"existing analysis contains {len(existing_rows)} extra GT rows")
    return results


def _summarize(rows):
    gains = [row["CollectiveGain"] for row in rows]
    return {
        "n_gt": len(rows),
        "mAC": _mean([row["AC"] for row in rows]),
        "BestIoGT": _mean([row["BestIoGT"] for row in rows]),
        "ABO": _mean([row["ABO"] for row in rows]),
        "BestPredIoA": _mean([row["BestPredIoA"] for row in rows]),
        "CollectiveGain_mean": _mean(gains),
        "CollectiveGain_median": _quantile(gains, 0.5),
        "CollectiveGain_p90": _quantile(gains, 0.9),
        "CollectiveGain_p95": _quantile(gains, 0.95),
        "P_gain_gt_0": _fraction(rows, lambda row: row["CollectiveGain"] > GAIN_EPSILON),
        "P_gain_gt_005": _fraction(rows, lambda row: row["CollectiveGain"] > 0.05),
        "P_gain_gt_010": _fraction(rows, lambda row: row["CollectiveGain"] > 0.10),
        "P_gain_gt_020": _fraction(rows, lambda row: row["CollectiveGain"] > 0.20),
        "multi_marginal_fraction": _fraction(
            rows, lambda row: row["marginal_contributor_count"] >= 2
        ),
        "substantial_multi_marginal_fraction": _fraction(
            rows, lambda row: row["substantial_marginal_contributor_count"] >= 2
        ),
        "mean_intersecting_pred_count": _mean(
            [row["intersecting_pred_count"] for row in rows]
        ),
        "mean_marginal_contributor_count": _mean(
            [row["marginal_contributor_count"] for row in rows]
        ),
    }


def _breakdown(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    order = {
        "small": 0, "medium": 1, "large": 2,
        "low": 0, "medium": 1, "high": 2,
    }
    return [
        {field: label, **_summarize(values)}
        for label, values in sorted(groups.items(), key=lambda item: order[item[0]])
    ]


def _exceedance(rows):
    return [
        {
            "gain_threshold": threshold,
            "fraction_above": _fraction(
                rows, lambda row, threshold=threshold:
                row["CollectiveGain"] > threshold
            ),
        }
        for threshold in EXCEEDANCE_GRID
    ]


def _source_output(root, source):
    return root / "runs" / source.dataset / source.model / f"seed_{source.seed}"


def _run_complete_matches(path, source):
    if not path.is_file():
        return False
    complete = _read_json(path)
    return (
        complete.get("status") == "complete"
        and complete.get("analysis_version") == ANALYSIS_VERSION
        and complete.get("source_run_fingerprint") == source.run_fingerprint
        and complete.get("theta") == PRIMARY_THETA
    )


def _process_source(source: CachedRun, root: Path):
    output = _source_output(root, source)
    complete_path = output / "complete.json"
    if _run_complete_matches(complete_path, source):
        return {
            "summary": _read_json(output / "run_summary.json"),
            "size": _read_csv(output / "size_breakdown.csv"),
            "density": _read_csv(output / "density_breakdown.csv"),
            "exceedance": _read_csv(output / "gain_exceedance.csv"),
            "cases": _read_csv(output / "top_cases.csv"),
        }
    output.mkdir(parents=True, exist_ok=True)
    document = _read_json(source.prediction_file)
    metadata = document.get("metadata", {})
    if metadata.get("run_fingerprint") != source.run_fingerprint:
        raise ValueError(f"prediction metadata mismatch: {source.prediction_file}")
    existing = _existing_gt_lookup(
        source.analysis_directory / "vc_gt_primary.csv"
    )
    rows = analyze_document(document, theta=PRIMARY_THETA, existing_rows=existing)
    summary = {
        "dataset": source.dataset,
        "model": source.model,
        "seed": source.seed,
        "model_id": source.model_id,
        "source_run_fingerprint": source.run_fingerprint,
        **_summarize(rows),
    }
    primary = _read_json(source.run_directory / "run_complete.json")["primary"]
    if abs(summary["mAC"] - float(primary["mAC"])) > 1e-10:
        raise ValueError(f"mAC audit failed: {source.run_directory}")
    summary.update(
        VC=float(primary["VC"]),
        F1=float(primary["standard_F1"]),
        LRP=float(primary["standard_LRP"]),
    )
    ap_path = source.analysis_directory / "diagnostic_ap_summary.json"
    if ap_path.is_file():
        ap = _read_json(ap_path)
        summary.update(
            diagnostic_AP50_95=float(ap["AP50_95"]),
            diagnostic_AR50_95=float(ap["AR50_95"]),
        )
    size_rows = [
        {"dataset": source.dataset, "model": source.model, "seed": source.seed, **row}
        for row in _breakdown(rows, "size_bin")
    ]
    density_rows = [
        {"dataset": source.dataset, "model": source.model, "seed": source.seed, **row}
        for row in _breakdown(rows, "density_bin")
    ]
    exceedance_rows = [
        {"dataset": source.dataset, "model": source.model, "seed": source.seed, **row}
        for row in _exceedance(rows)
    ]
    case_rows = []
    for row in sorted(
            rows,
            key=lambda value: (
                -value["CollectiveGain"], -value["AC"],
                value["image_id"], value["gt_id"],
            ))[:20]:
        case_rows.append({
            "dataset": source.dataset,
            "model": source.model,
            "seed": source.seed,
            **row,
        })
    per_gt_fields = list(rows[0])
    _atomic_csv(output / "per_gt_collective_gain.csv", rows, per_gt_fields)
    _atomic_json(output / "run_summary.json", summary)
    _atomic_csv(output / "size_breakdown.csv", size_rows)
    _atomic_csv(output / "density_breakdown.csv", density_rows)
    _atomic_csv(output / "gain_exceedance.csv", exceedance_rows)
    _atomic_csv(output / "top_cases.csv", case_rows)
    _atomic_json(
        complete_path,
        {
            "status": "complete",
            "analysis_version": ANALYSIS_VERSION,
            "source_run_fingerprint": source.run_fingerprint,
            "theta": PRIMARY_THETA,
            "n_gt": len(rows),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "summary": summary,
        "size": size_rows,
        "density": density_rows,
        "exceedance": exceedance_rows,
        "cases": case_rows,
    }


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_seed_rows(rows, group_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    excluded = set(group_fields) | {"seed", "model_id", "source_run_fingerprint"}
    for key in sorted(groups):
        group = groups[key]
        if len(group) != 5 or {int(row["seed"]) for row in group} != set(range(5)):
            raise ValueError(f"five-seed group is incomplete: {key}")
        aggregate = {field: value for field, value in zip(group_fields, key)}
        aggregate.update(n_seeds=5)
        for field in group[0]:
            if field in excluded:
                continue
            values = [_number(row.get(field)) for row in group]
            if any(value is None for value in values):
                continue
            aggregate[f"{field}_mean"] = statistics.fmean(values)
            aggregate[f"{field}_std"] = statistics.stdev(values)
        output.append(aggregate)
    return output


def _plot_exceedance(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharey=True)
    for axis, dataset in zip(axes, ("mscoco", "visdrone")):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for model in MODEL_ORDER:
            selected = [row for row in dataset_rows if row["model"] == model]
            if not selected:
                continue
            selected.sort(key=lambda row: float(row["gain_threshold"]))
            x = [float(row["gain_threshold"]) for row in selected]
            y = [float(row["fraction_above_mean"]) for row in selected]
            sd = [float(row["fraction_above_std"]) for row in selected]
            axis.plot(x, y, marker="o", markersize=3, label=MODEL_LABELS[model])
            axis.fill_between(
                x,
                [max(0.0, a - b) for a, b in zip(y, sd)],
                [min(1.0, a + b) for a, b in zip(y, sd)],
                alpha=0.15,
            )
        axis.set(
            title=DATASET_LABELS[dataset],
            xlabel=r"Collective-gain threshold $\delta$",
            xlim=(0.0, 0.30),
            ylim=(0.0, None),
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"Fraction of GT with $\Delta_{collective}>\delta$")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels))
    figure.suptitle("Natural collective gain in saved detector outputs", y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output / "collective_gain_exceedance.pdf", bbox_inches="tight")
    figure.savefig(output / "collective_gain_exceedance.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_breakdowns(size_rows, density_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
    for column, dataset in enumerate(("mscoco", "visdrone")):
        for row_index, (rows, field, categories, title) in enumerate((
                (size_rows, "size_bin", ("small", "medium", "large"), "Object size"),
                (density_rows, "density_bin", ("low", "medium", "high"), "Image density"),
        )):
            axis = axes[row_index, column]
            models = [
                model for model in MODEL_ORDER
                if any(row["dataset"] == dataset and row["model"] == model for row in rows)
            ]
            x = np.arange(len(categories))
            width = 0.8 / max(1, len(models))
            for index, model in enumerate(models):
                selected = {
                    row[field]: row for row in rows
                    if row["dataset"] == dataset and row["model"] == model
                }
                values = [float(selected[name]["P_gain_gt_005_mean"]) for name in categories]
                errors = [float(selected[name]["P_gain_gt_005_std"]) for name in categories]
                axis.bar(
                    x + (index - (len(models) - 1) / 2) * width,
                    values,
                    width,
                    yerr=errors,
                    capsize=2,
                    label=MODEL_LABELS[model],
                )
            axis.set(
                title=f"{DATASET_LABELS[dataset]}: {title}",
                xticks=x,
                xticklabels=categories,
                ylabel=r"$P(\Delta_{collective}>0.05)$" if column == 0 else "",
                ylim=(0.0, None),
            )
            axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels))
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "collective_gain_breakdowns.pdf", bbox_inches="tight")
    figure.savefig(output / "collective_gain_breakdowns.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _report(output, model_rows):
    lines = [
        "# Natural Collective Gain Analysis", "",
        "This report is generated from saved full-validation prediction caches.",
        "No model training or inference is performed.", "",
        f"Primary confidence threshold: `{PRIMARY_THETA}`", "",
        "## Main outputs", "",
        "- `model_summary.csv`: five-seed model summaries.",
        "- `run_summary.csv`: one row per model seed.",
        "- `collective_gain_exceedance.pdf`: distribution of natural gain.",
        "- `collective_gain_breakdowns.pdf`: object-size and density breakdowns.",
        "- `top_collective_gain_cases.csv`: candidates for qualitative review.",
        "- `runs/.../per_gt_collective_gain.csv`: auditable per-GT values.", "",
        "## Interpretation rule", "",
        "`BestIoGT` is the best single ground-truth-normalized IoA. "
        "`CollectiveGain = AC - BestIoGT` isolates coverage supplied by the "
        "prediction union beyond the best single prediction.", "",
        "## Five-seed groups", "",
    ]
    for row in model_rows:
        lines.append(
            f"- {DATASET_LABELS[row['dataset']]} / {MODEL_LABELS[row['model']]}: "
            f"mean gain={float(row['CollectiveGain_mean_mean']):.6f}, "
            f"P(gain>0.05)={float(row['P_gain_gt_005_mean']):.4f}."
        )
    _atomic_text(output / "README.md", "\n".join(lines) + "\n")


def preflight_natural_collective_gain(output=DEFAULT_OUTPUT):
    sources = main_natural_sources()
    audit = registry_audit(sources)
    audit.update(
        status="passed",
        analysis_version=ANALYSIS_VERSION,
        vc_version=VC_VERSION,
        theta=PRIMARY_THETA,
        output=str(Path(output).resolve()),
        training=False,
        inference=False,
    )
    return audit


def run_natural_collective_gain(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = preflight_natural_collective_gain(output)
    _atomic_json(output / "preflight.json", preflight)
    sources = main_natural_sources()
    run_rows, size_rows, density_rows, exceedance_rows, case_rows = [], [], [], [], []
    for index, source in enumerate(sources, 1):
        print(
            f"COLLECTIVE {index:02d}/{len(sources):02d} START: "
            f"{source.dataset}:{source.model}:seed{source.seed}",
            flush=True,
        )
        result = _process_source(source, output)
        run_rows.append(result["summary"])
        size_rows.extend(result["size"])
        density_rows.extend(result["density"])
        exceedance_rows.extend(result["exceedance"])
        case_rows.extend(result["cases"])
        print(
            f"COLLECTIVE {index:02d}/{len(sources):02d} COMPLETE: "
            f"{source.dataset}:{source.model}:seed{source.seed}",
            flush=True,
        )
    model_rows = _aggregate_seed_rows(run_rows, ("dataset", "model"))
    size_model_rows = _aggregate_seed_rows(
        size_rows, ("dataset", "model", "size_bin")
    )
    density_model_rows = _aggregate_seed_rows(
        density_rows, ("dataset", "model", "density_bin")
    )
    exceedance_model_rows = _aggregate_seed_rows(
        exceedance_rows, ("dataset", "model", "gain_threshold")
    )
    case_rows.sort(key=lambda row: (
        row["dataset"], row["model"], -float(row["CollectiveGain"]),
        int(row["seed"]), row["image_id"], row["gt_id"],
    ))
    _atomic_csv(output / "run_summary.csv", run_rows)
    _atomic_csv(output / "model_summary.csv", model_rows)
    _atomic_csv(output / "size_breakdown_by_run.csv", size_rows)
    _atomic_csv(output / "size_breakdown_by_model.csv", size_model_rows)
    _atomic_csv(output / "density_breakdown_by_run.csv", density_rows)
    _atomic_csv(output / "density_breakdown_by_model.csv", density_model_rows)
    _atomic_csv(output / "gain_exceedance_by_run.csv", exceedance_rows)
    _atomic_csv(output / "gain_exceedance_by_model.csv", exceedance_model_rows)
    _atomic_csv(output / "top_collective_gain_cases.csv", case_rows)
    _plot_exceedance(exceedance_model_rows, output)
    _plot_breakdowns(size_model_rows, density_model_rows, output)
    _report(output, model_rows)
    complete = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "registry_version": REGISTRY_VERSION,
        "vc_version": VC_VERSION,
        "theta": PRIMARY_THETA,
        "n_runs": len(run_rows),
        "n_model_groups": len(model_rows),
        "training": False,
        "inference": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
    }
    _atomic_json(output / "analysis_complete.json", complete)
    return output
