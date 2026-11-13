"""Direct paired comparison of mAC with existing overlap baselines.

This module consumes the completed natural collective-gain analysis.  It does
not train detectors and does not run model inference.  Every comparison is
paired at the ground-truth level and model summaries use five training seeds.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np

from .cache_registry import DATASETS, MODEL_ORDER, CachedRun, main_natural_sources
from .natural_collective_gain import (
    ANALYSIS_VERSION as NATURAL_ANALYSIS_VERSION,
    DEFAULT_OUTPUT as NATURAL_OUTPUT,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _read_csv,
    _read_json,
)
from .offline import PROJECT_ROOT
from .vc_v1 import PRIMARY_THETA, VC_VERSION


COMPARISON_VERSION = "vc-v1-direct-metric-comparison-0.1.2"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "metric_comparison"
MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolo11n": "YOLO11n",
    "yolo11s": "YOLO11s",
    "rtdetr_l": "RT-DETR-L",
}
DATASET_LABELS = {"mscoco": "MS COCO", "visdrone": "VisDrone"}
IMAGE_ROOTS = {
    "mscoco": PROJECT_ROOT.parent / "Database" / "MS-COCO" / "images" / "val",
    "visdrone": PROJECT_ROOT.parent / "Database" / "VisDrone" / "images" / "val",
}
HISTOGRAM_BINS = np.linspace(0.0, 1.0, 61)
EPSILON = 1e-12


def _source_input(root: Path, source: CachedRun) -> Path:
    return root / "runs" / source.dataset / source.model / f"seed_{source.seed}"


def preflight_direct_metric_comparison(
        output=DEFAULT_OUTPUT, natural_output=NATURAL_OUTPUT):
    natural_output = Path(natural_output).resolve()
    complete_path = natural_output / "analysis_complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(
            "natural collective-gain analysis is missing; run "
            "experiments/collective_gain.py first"
        )
    complete = _read_json(complete_path)
    if (
            complete.get("status") != "complete"
            or complete.get("analysis_version") != NATURAL_ANALYSIS_VERSION
            or int(complete.get("n_runs", 0)) != 35):
        raise ValueError("natural collective-gain analysis is incomplete or changed")
    sources = main_natural_sources()
    total_bytes = 0
    for source in sources:
        run_root = _source_input(natural_output, source)
        per_gt = run_root / "per_gt_collective_gain.csv"
        run_complete = run_root / "complete.json"
        for path in (per_gt, run_complete):
            if not path.is_file():
                raise FileNotFoundError(f"missing first-round result: {path}")
        marker = _read_json(run_complete)
        if (
                marker.get("status") != "complete"
                or marker.get("source_run_fingerprint") != source.run_fingerprint):
            raise ValueError(f"stale first-round result: {run_root}")
        total_bytes += per_gt.stat().st_size
    for dataset, root in IMAGE_ROOTS.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{dataset} validation images not found: {root}")
    return {
        "status": "passed",
        "comparison_version": COMPARISON_VERSION,
        "natural_analysis_version": NATURAL_ANALYSIS_VERSION,
        "vc_version": VC_VERSION,
        "theta": PRIMARY_THETA,
        "n_runs": len(sources),
        "n_model_groups": 7,
        "n_per_gt_bytes": total_bytes,
        "output": str(Path(output).resolve()),
        "training": False,
        "inference": False,
    }


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) <= EPSILON or np.std(second) <= EPSILON:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def summarize_paired_rows(rows: list[dict]) -> dict:
    """Summarize paired per-GT comparisons for one model seed."""
    ac = np.asarray([float(row["AC"]) for row in rows], dtype=float)
    best_iogt = np.asarray([float(row["BestIoGT"]) for row in rows], dtype=float)
    abo = np.asarray([float(row["ABO"]) for row in rows], dtype=float)
    pred_ioa = np.asarray([float(row["BestPredIoA"]) for row in rows], dtype=float)
    vc = np.asarray([float(row["local_VC"]) for row in rows], dtype=float)
    gain = ac - best_iogt
    substantial = gain > 0.05
    positive_ac = ac > EPSILON
    collective_share = gain[positive_ac] / ac[positive_ac]
    return {
        "n_gt": len(rows),
        "AC_mean": float(np.mean(ac)),
        "BestIoGT_mean": float(np.mean(best_iogt)),
        "ABO_mean": float(np.mean(abo)),
        "BestPredIoA_mean": float(np.mean(pred_ioa)),
        "local_VC_mean": float(np.mean(vc)),
        "gain_mean": float(np.mean(gain)),
        "collective_share_given_AC_positive_mean": (
            float(np.mean(collective_share)) if len(collective_share) else 0.0
        ),
        "P_gain_gt_005": _fraction(substantial),
        "P_union050_single_below050": _fraction((ac >= 0.50) & (best_iogt < 0.50)),
        "P_union075_single_below050": _fraction((ac >= 0.75) & (best_iogt < 0.50)),
        "P_AC050_ABO_below050": _fraction((ac >= 0.50) & (abo < 0.50)),
        "P_BestIoGT050_ABO_below050": _fraction(
            (best_iogt >= 0.50) & (abo < 0.50)
        ),
        "P_AC050_BestPredIoA_below050": _fraction(
            (ac >= 0.50) & (pred_ioa < 0.50)
        ),
        "P_gain005_and_VC_below050": _fraction(substantial & (vc < 0.50)),
        "P_VC_below050_given_gain005": (
            _fraction(vc[substantial] < 0.50) if np.any(substantial) else 0.0
        ),
        "r_AC_BestIoGT": _pearson(ac, best_iogt),
        "r_AC_ABO": _pearson(ac, abo),
        "r_AC_BestPredIoA": _pearson(ac, pred_ioa),
        "r_AC_local_VC": _pearson(ac, vc),
    }


def _aggregate_five_seeds(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"])].append(row)
    output = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            DATASETS.index(item[0][0]), MODEL_ORDER.index(item[0][1])
        ),
    )
    for (dataset, model), group in ordered_groups:
        seeds = {int(row["seed"]) for row in group}
        if len(group) != 5 or seeds != set(range(5)):
            raise ValueError(f"incomplete five-seed group: {dataset}/{model}")
        summary = {"dataset": dataset, "model": model, "n_seeds": 5}
        excluded = {"dataset", "model", "seed", "model_id", "run_fingerprint"}
        for field in group[0]:
            if field in excluded:
                continue
            values = [float(row[field]) for row in group]
            summary[f"{field}_mean"] = statistics.fmean(values)
            summary[f"{field}_std"] = statistics.stdev(values)
        output.append(summary)
    return output


def _histograms(rows: list[dict]) -> dict[str, np.ndarray]:
    ac = np.asarray([float(row["AC"]) for row in rows], dtype=float)
    values = {}
    for field in ("BestIoGT", "ABO", "BestPredIoA"):
        other = np.asarray([float(row[field]) for row in rows], dtype=float)
        values[field] = np.histogram2d(
            other, ac, bins=(HISTOGRAM_BINS, HISTOGRAM_BINS)
        )[0]
    return values


def _plot_relationships(histograms: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    figure, axes = plt.subplots(
        3, 2, figsize=(10.8, 12.6), sharex=True, sharey=True,
        constrained_layout=True,
    )
    comparisons = (
        ("BestIoGT", "Best-IoGT"),
        ("ABO", "ABO"),
        ("BestPredIoA", "Pred-IoA"),
    )
    image = None
    for row_index, (field, label) in enumerate(comparisons):
        maximum = max(
            float(np.max(histograms[dataset][field])) for dataset in DATASETS
        )
        norm = LogNorm(vmin=1, vmax=max(1, maximum))
        for column, dataset in enumerate(DATASETS):
            axis = axes[row_index, column]
            counts = histograms[dataset][field]
            masked = np.ma.masked_where(counts.T <= 0, counts.T)
            image = axis.imshow(
                masked,
                origin="lower",
                extent=(0, 1, 0, 1),
                aspect="equal",
                cmap="viridis",
                norm=norm,
            )
            axis.plot([0, 1], [0, 1], "--", color="white", linewidth=1.2)
            axis.set(
                title=f"{DATASET_LABELS[dataset]}: AC vs. {label}",
                xlabel=label if row_index == len(comparisons) - 1 else "",
                ylabel="Area Coverage (AC)" if column == 0 else "",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            axis.grid(False)
        if image is not None:
            figure.colorbar(image, ax=axes[row_index, :], fraction=0.025, pad=0.02,
                            label="Ground-truth observations (log count)")
    figure.suptitle("Paired ground-truth comparison with single-box baselines", fontsize=14)
    figure.savefig(output / "direct_metric_relationships.pdf", bbox_inches="tight")
    figure.savefig(output / "direct_metric_relationships.png", dpi=240,
                   bbox_inches="tight")
    plt.close(figure)


def _plot_disagreements(model_rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), sharey="row")
    top_metrics = (
        ("P_gain_gt_005", r"$P(\Delta_{collective}>0.05)$"),
        ("P_AC050_ABO_below050", r"$P(AC\geq.5, ABO<.5)$"),
        ("P_gain005_and_VC_below050", r"$P(\Delta>.05, VC<.5)$"),
    )
    bottom_metrics = (
        ("P_union050_single_below050", r"$P(AC\geq.5, B_{IoGT}<.5)$"),
        ("P_union075_single_below050", r"$P(AC\geq.75, B_{IoGT}<.5)$"),
    )
    for column, dataset in enumerate(DATASETS):
        selected = [row for row in model_rows if row["dataset"] == dataset]
        selected.sort(key=lambda row: MODEL_ORDER.index(row["model"]))
        names = [MODEL_LABELS[row["model"]] for row in selected]
        x = np.arange(len(names))
        for row_index, metrics in enumerate((top_metrics, bottom_metrics)):
            axis = axes[row_index, column]
            width = 0.78 / len(metrics)
            for index, (field, label) in enumerate(metrics):
                values = [100.0 * float(row[f"{field}_mean"]) for row in selected]
                errors = [100.0 * float(row[f"{field}_std"]) for row in selected]
                axis.bar(
                    x + (index - (len(metrics) - 1) / 2) * width,
                    values,
                    width,
                    yerr=errors,
                    capsize=2,
                    label=label,
                )
            axis.set(
                title=f"{DATASET_LABELS[dataset]}: "
                      f"{'metric disagreement' if row_index == 0 else 'union-only threshold crossing'}",
                xticks=x,
                xticklabels=names,
                ylabel="Ground-truth instances (%)" if column == 0 else "",
                ylim=(0.0, None),
            )
            axis.tick_params(axis="x", rotation=18)
            axis.grid(axis="y", alpha=0.25)
    for row_index, metrics in enumerate((top_metrics, bottom_metrics)):
        maximum = max(
            100.0 * (float(row[f"{field}_mean"]) + float(row[f"{field}_std"]))
            for row in model_rows for field, _ in metrics
        )
        axes[row_index, 0].set_ylim(0.0, maximum * 1.12)
    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    bottom_handles, bottom_labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(top_handles, top_labels, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, 0.995), frameon=False)
    figure.legend(bottom_handles, bottom_labels, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, 0.005), frameon=False)
    figure.subplots_adjust(top=0.91, bottom=0.13, hspace=0.42, wspace=0.10)
    figure.savefig(output / "paired_disagreement_rates.pdf", bbox_inches="tight")
    figure.savefig(output / "paired_disagreement_rates.png", dpi=240,
                   bbox_inches="tight")
    plt.close(figure)


def _canonical_case_candidates(rows: list[dict], dataset: str) -> list[dict]:
    substantial = [row for row in rows if float(row["CollectiveGain"]) > 0.10]
    retained = [
        row for row in substantial
        if float(row["local_VC"]) >= 0.65 and int(row["assigned_pred_count"]) >= 2
    ]
    attenuated = [
        row for row in substantial
        if float(row["local_VC"]) <= 0.45
        and int(row["assigned_pred_count"]) < int(row["marginal_contributor_count"])
    ]
    if not retained or not attenuated:
        raise ValueError(f"unable to select paired qualitative cases for {dataset}")
    retained.sort(key=lambda row: -float(row["CollectiveGain"]))
    attenuated.sort(key=lambda row: -float(row["CollectiveGain"]))
    first = dict(retained[0])
    first["case_type"] = "VC-retained"
    second = dict(attenuated[0])
    second["case_type"] = "VC-attenuated"
    return [first, second]


def _existing_assignment_lookup(source: CachedRun) -> dict:
    rows = _read_csv(source.analysis_directory / "vc_gt_primary.csv")
    return {
        (str(row["image_id"]), str(row["gt_id"])): {
            "assigned_pred_ids": [
                value for value in str(row.get("assigned_pred_ids", "")).split(";")
                if value
            ],
            "assigned_weights": str(row.get("assigned_weights", "")),
        }
        for row in rows
    }


def _image_path(dataset: str, image_id: str) -> Path:
    root = IMAGE_ROOTS[dataset]
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = root / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"source image not found: {dataset}/{image_id}")


def _case_figure(cases: list[dict], source_lookup: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image

    loaded_documents = {}
    assignment_lookups = {}
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.8))
    manifest = []
    for axis, case in zip(axes.flat, cases):
        key = (case["dataset"], case["model"], int(case["seed"]))
        source = source_lookup[key]
        if key not in loaded_documents:
            document = _read_json(source.prediction_file)
            loaded_documents[key] = {
                str(image["image_id"]): image for image in document["images"]
            }
            assignment_lookups[key] = _existing_assignment_lookup(source)
        image_record = loaded_documents[key][str(case["image_id"])]
        gt = next(
            row for row in image_record["gts"] if str(row["id"]) == str(case["gt_id"])
        )
        predictions = {str(row["id"]): row for row in image_record["predictions"]}
        marginal_ids = [
            value for value in str(case["marginal_pred_ids"]).split(";") if value
        ]
        assigned_ids = assignment_lookups[key][
            (str(case["image_id"]), str(case["gt_id"]))
        ]["assigned_pred_ids"]
        relevant_ids = set(marginal_ids) | set(assigned_ids) | {str(case["best_pred_id"])}
        relevant = [predictions[value] for value in relevant_ids if value in predictions]
        with Image.open(_image_path(case["dataset"], str(case["image_id"]))) as image:
            rgb = image.convert("RGB")
            axis.imshow(rgb)
        coordinates = [tuple(map(float, gt["xyxy"]))] + [
            tuple(map(float, row["xyxy"])) for row in relevant
        ]
        x1 = min(row[0] for row in coordinates)
        y1 = min(row[1] for row in coordinates)
        x2 = max(row[2] for row in coordinates)
        y2 = max(row[3] for row in coordinates)
        margin = 0.18 * max(x2 - x1, y2 - y1, 20.0)
        width, height = int(image_record["width"]), int(image_record["height"])
        axis.set_xlim(max(0, x1 - margin), min(width, x2 + margin))
        axis.set_ylim(min(height, y2 + margin), max(0, y1 - margin))
        for pred_id in marginal_ids:
            if pred_id not in predictions:
                continue
            px1, py1, px2, py2 = map(float, predictions[pred_id]["xyxy"])
            axis.add_patch(Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                fill=False, edgecolor="#f39c12", linewidth=1.3,
            ))
        for pred_id in assigned_ids:
            if pred_id not in predictions:
                continue
            px1, py1, px2, py2 = map(float, predictions[pred_id]["xyxy"])
            axis.add_patch(Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                fill=False, edgecolor="#2ca02c", linewidth=2.0,
            ))
        best_id = str(case["best_pred_id"])
        if best_id in predictions:
            px1, py1, px2, py2 = map(float, predictions[best_id]["xyxy"])
            axis.add_patch(Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                fill=False, edgecolor="#1f77b4", linewidth=2.2,
            ))
        gx1, gy1, gx2, gy2 = map(float, gt["xyxy"])
        axis.add_patch(Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            fill=False, edgecolor="#d62728", linewidth=2.6,
        ))
        axis.set_title(
            f"{DATASET_LABELS[case['dataset']]} — {case['case_type']}\n"
            f"AC={float(case['AC']):.2f}, Best-IoGT={float(case['BestIoGT']):.2f}, "
            f"ABO={float(case['ABO']):.2f}, VC={float(case['local_VC']):.2f}"
            f"\nMarginal={int(case['marginal_contributor_count'])}, "
            f"VC-assigned={int(case['assigned_pred_count'])}"
        )
        axis.axis("off")
        manifest.append({
            "case_type": case["case_type"],
            "dataset": case["dataset"],
            "model": case["model"],
            "seed": case["seed"],
            "image_id": case["image_id"],
            "gt_id": case["gt_id"],
            "class_id": case["class_id"],
            "AC": case["AC"],
            "BestIoGT": case["BestIoGT"],
            "ABO": case["ABO"],
            "BestPredIoA": case["BestPredIoA"],
            "CollectiveGain": case["CollectiveGain"],
            "VC": case["local_VC"],
            "marginal_contributor_count": case["marginal_contributor_count"],
            "assigned_pred_count": case["assigned_pred_count"],
            "image_path": str(_image_path(case["dataset"], str(case["image_id"]))),
        })
    figure.legend(
        handles=[
            Patch(facecolor="none", edgecolor="#d62728", label="Ground truth"),
            Patch(facecolor="none", edgecolor="#1f77b4", label="Best single box"),
            Patch(facecolor="none", edgecolor="#f39c12", label="Marginal contributor"),
            Patch(facecolor="none", edgecolor="#2ca02c", label="VC-assigned prediction"),
        ],
        loc="upper center", ncol=4, frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output / "qualitative_collective_cases.pdf", bbox_inches="tight")
    figure.savefig(output / "qualitative_collective_cases.png", dpi=240,
                   bbox_inches="tight")
    plt.close(figure)
    _atomic_csv(output / "qualitative_case_manifest.csv", manifest)


def _latex_table(rows: list[dict], output: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Direct comparison of collective area coverage with pairwise overlap baselines. Results are mean $\pm$ sample standard deviation over five seeds.}",
        r"\label{tab:direct_metric_comparison}",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"Dataset & Model & $\Delta_{\mathrm{coll}}$ & $P(\Delta>0.05)$ & $P(AC\geq.5,\,B_{\mathrm{IoGT}}<.5)$ & $P(AC\geq.5,\,ABO<.5)$ & $r(AC,B_{\mathrm{IoGT}})$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{DATASET_LABELS[row['dataset']]} & {MODEL_LABELS[row['model']]} & "
            f"{float(row['gain_mean_mean']):.4f} $\\pm$ {float(row['gain_mean_std']):.4f} & "
            f"{100*float(row['P_gain_gt_005_mean']):.2f} $\\pm$ {100*float(row['P_gain_gt_005_std']):.2f} & "
            f"{100*float(row['P_union050_single_below050_mean']):.2f} $\\pm$ {100*float(row['P_union050_single_below050_std']):.2f} & "
            f"{100*float(row['P_AC050_ABO_below050_mean']):.2f} $\\pm$ {100*float(row['P_AC050_ABO_below050_std']):.2f} & "
            f"{float(row['r_AC_BestIoGT_mean']):.3f} $\\pm$ {float(row['r_AC_BestIoGT_std']):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    _atomic_text(output / "direct_metric_comparison_table.tex", "\n".join(lines) + "\n")


def _report(model_rows: list[dict], output: Path) -> None:
    lines = [
        "# Direct Metric Comparison", "",
        "All values are computed from saved full-validation predictions. No training or inference was performed.", "",
        "`BestIoGT` is the closest pairwise directional-coverage baseline; `ABO` is the best pairwise IoU; `BestPredIoA` measures the reverse directional containment.", "",
        "## Five-seed results", "",
    ]
    for row in model_rows:
        lines.append(
            f"- {DATASET_LABELS[row['dataset']]} / {MODEL_LABELS[row['model']]}: "
            f"P(gain>0.05)={100*float(row['P_gain_gt_005_mean']):.2f}%, "
            f"union-only 0.5 crossing={100*float(row['P_union050_single_below050_mean']):.2f}%, "
            f"AC-pass/ABO-fail={100*float(row['P_AC050_ABO_below050_mean']):.2f}%, "
            f"r(AC,BestIoGT)={float(row['r_AC_BestIoGT_mean']):.3f}."
        )
    lines.extend([
        "", "## Outputs", "",
        "- `direct_metric_relationships.pdf`: paired ground-truth density plots.",
        "- `paired_disagreement_rates.pdf`: five-seed disagreement rates.",
        "- `qualitative_collective_cases.pdf`: VC-retained and VC-attenuated cases.",
        "- `direct_metric_comparison_table.tex`: paper-ready table source.",
        "- `per_run_comparison.csv` and `model_comparison.csv`: auditable values.",
    ])
    _atomic_text(output / "README.md", "\n".join(lines) + "\n")


def run_direct_metric_comparison(output=DEFAULT_OUTPUT, natural_output=NATURAL_OUTPUT):
    output = Path(output).resolve()
    natural_output = Path(natural_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = preflight_direct_metric_comparison(output, natural_output)
    _atomic_json(output / "preflight.json", preflight)
    sources = main_natural_sources()
    run_rows = []
    histograms = {
        dataset: {field: np.zeros((60, 60), dtype=np.int64)
                  for field in ("BestIoGT", "ABO", "BestPredIoA")}
        for dataset in DATASETS
    }
    cases = []
    source_lookup = {
        (source.dataset, source.model, source.seed): source for source in sources
    }
    for index, source in enumerate(sources, 1):
        print(
            f"COMPARE {index:02d}/{len(sources):02d} START: "
            f"{source.dataset}:{source.model}:seed{source.seed}",
            flush=True,
        )
        per_gt = _source_input(natural_output, source) / "per_gt_collective_gain.csv"
        rows = _read_csv(per_gt)
        run_rows.append({
            "dataset": source.dataset,
            "model": source.model,
            "seed": source.seed,
            "model_id": source.model_id,
            "run_fingerprint": source.run_fingerprint,
            **summarize_paired_rows(rows),
        })
        current_histograms = _histograms(rows)
        for field, values in current_histograms.items():
            histograms[source.dataset][field] += values.astype(np.int64)
        if source.model == "yolo11s" and source.seed == 0:
            selected = _canonical_case_candidates(rows, source.dataset)
            for row in selected:
                row.update(dataset=source.dataset, model=source.model, seed=source.seed)
            cases.extend(selected)
        print(
            f"COMPARE {index:02d}/{len(sources):02d} COMPLETE: "
            f"{source.dataset}:{source.model}:seed{source.seed}",
            flush=True,
        )
    model_rows = _aggregate_five_seeds(run_rows)
    _atomic_csv(output / "per_run_comparison.csv", run_rows)
    _atomic_csv(output / "model_comparison.csv", model_rows)
    _plot_relationships(histograms, output)
    _plot_disagreements(model_rows, output)
    _case_figure(cases, source_lookup, output)
    _latex_table(model_rows, output)
    _report(model_rows, output)
    complete = {
        "status": "complete",
        "comparison_version": COMPARISON_VERSION,
        "natural_analysis_version": NATURAL_ANALYSIS_VERSION,
        "vc_version": VC_VERSION,
        "theta": PRIMARY_THETA,
        "n_runs": len(run_rows),
        "n_model_groups": len(model_rows),
        "n_qualitative_cases": len(cases),
        "training": False,
        "inference": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
    }
    _atomic_json(output / "analysis_complete.json", complete)
    return output

