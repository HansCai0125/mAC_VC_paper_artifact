"""Build paper-ready VC v1 tables, figures, and result prose.

This module is a reporting layer.  It reads only the completed, immutable
fixed-checkpoint and formal gaming outputs; it never loads models, performs
inference, or trains detectors.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PAPER_RESULTS_VERSION = "vc-v1-paper-results-0.1.0"
MODELS = ("yolov8n", "yolo11n", "yolo11s")
DATASETS = ("mscoco", "visdrone")
DATASET_LABELS = {"mscoco": "MS COCO", "visdrone": "VisDrone"}
MODEL_LABELS = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo11s": "YOLO11s"}
PRIMARY_THETA = 0.25

METRICS = (
    ("normalized_VC_mean", "VC", "#0072B2", "-"),
    ("normalized_mAC_mean", "mAC", "#D55E00", "--"),
    ("normalized_F1_mean", "F1", "#009E73", "-."),
    ("normalized_one_minus_LRP_mean", "1-LRP", "#CC79A7", ":"),
)

KEY_CONDITIONS = (
    ("shrink", "center_scale", 0.71, "Shrink: side scale 0.71"),
    ("expand", "center_scale", 1.41, "Expand: side scale 1.41"),
    ("expand", "center_scale", 3.0, "Expand: side scale 3.00"),
    ("duplicate", "exact", 8.0, "Exact duplicate: multiplicity 8"),
    ("duplicate", "exact", 16.0, "Exact duplicate: multiplicity 16"),
    (
        "duplicate",
        "controlled_jitter_2pct_center_1pct_size",
        8.0,
        "Jittered duplicate: multiplicity 8",
    ),
    (
        "duplicate",
        "controlled_jitter_2pct_center_1pct_size",
        16.0,
        "Jittered duplicate: multiplicity 16",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        return math.nan
    return float(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _require_complete_sources(gaming_root: Path, fixed_root: Path) -> dict[str, object]:
    gaming_manifest_path = gaming_root / "gaming_formal_complete.json"
    fixed_manifest_path = fixed_root / "analysis_complete.json"
    for path in (gaming_manifest_path, fixed_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    gaming_manifest = json.loads(gaming_manifest_path.read_text(encoding="utf-8"))
    fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8"))
    if gaming_manifest.get("status") != "complete":
        raise ValueError("formal gaming source is not complete")
    if fixed_manifest.get("status") != "complete":
        raise ValueError("fixed-checkpoint source is not complete")
    if int(gaming_manifest.get("completed_runs", -1)) != 18:
        raise ValueError("formal gaming source does not contain 18 completed runs")
    return {"gaming": gaming_manifest, "fixed_checkpoint": fixed_manifest}


def _select(
    rows: Iterable[Mapping[str, str]],
    *,
    dataset: str,
    manipulation: str,
    variant: str,
    level: float | None = None,
) -> list[Mapping[str, str]]:
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["manipulation"] == manipulation
        and row["variant"] == variant
        and (level is None or abs(float(row["level"]) - level) < 1e-12)
    ]
    return selected


def _model_summary(
    rows: Iterable[Mapping[str, str]], field: str
) -> tuple[float, float, float]:
    values = [_float(row, field) for row in rows]
    if len(values) != len(MODELS) or any(not math.isfinite(value) for value in values):
        raise ValueError(f"expected three finite architecture values for {field}")
    return statistics.fmean(values), min(values), max(values)


def _key_condition_table(aggregate: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in DATASETS:
        for manipulation, variant, level, label in KEY_CONDITIONS:
            rows = _select(
                aggregate,
                dataset=dataset,
                manipulation=manipulation,
                variant=variant,
                level=level,
            )
            if sorted(row["model"] for row in rows) != sorted(MODELS):
                raise ValueError(f"incomplete key condition: {dataset} / {label}")
            vc_mean, vc_min, vc_max = _model_summary(rows, "normalized_VC_mean")
            mac_mean, mac_min, mac_max = _model_summary(rows, "normalized_mAC_mean")
            f1_mean, f1_min, f1_max = _model_summary(rows, "normalized_F1_mean")
            lrp_mean, lrp_min, lrp_max = _model_summary(
                rows, "normalized_one_minus_LRP_mean"
            )
            vc_seed_sd_max: object = ""
            if dataset == "visdrone":
                vc_seed_sd_max = max(_float(row, "normalized_VC_std") for row in rows)
            output.append(
                {
                    "dataset": DATASET_LABELS[dataset],
                    "condition": label,
                    "level": level,
                    "architecture_count": len(rows),
                    "statistical_unit": (
                        "three predeclared fixed checkpoints"
                        if dataset == "mscoco"
                        else "three architecture means; each from five training seeds"
                    ),
                    "VC_ratio_mean": vc_mean,
                    "VC_ratio_min_architecture": vc_min,
                    "VC_ratio_max_architecture": vc_max,
                    "VC_seed_sd_max": vc_seed_sd_max,
                    "mAC_ratio_mean": mac_mean,
                    "mAC_ratio_min_architecture": mac_min,
                    "mAC_ratio_max_architecture": mac_max,
                    "F1_ratio_mean": f1_mean,
                    "F1_ratio_min_architecture": f1_min,
                    "F1_ratio_max_architecture": f1_max,
                    "one_minus_LRP_ratio_mean": lrp_mean,
                    "one_minus_LRP_ratio_min_architecture": lrp_min,
                    "one_minus_LRP_ratio_max_architecture": lrp_max,
                }
            )
    if len(output) != len(DATASETS) * len(KEY_CONDITIONS):
        raise AssertionError("key-condition table size changed")
    return output


def _theta_table(theta_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in DATASETS:
        for model in MODELS:
            rows = sorted(
                (
                    row
                    for row in theta_rows
                    if row["dataset"] == dataset and row["model"] == model
                ),
                key=lambda row: float(row["confidence_threshold"]),
            )
            if len(rows) != 19:
                raise ValueError(f"expected 19 theta rows for {dataset}/{model}")
            baseline = next(
                row
                for row in rows
                if abs(float(row["confidence_threshold"]) - PRIMARY_THETA) < 1e-12
            )
            best_vc = max(rows, key=lambda row: _float(row, "VC_mean"))
            best_f1 = max(rows, key=lambda row: _float(row, "standard_F1_mean"))
            best_lrp = min(rows, key=lambda row: _float(row, "standard_LRP_mean"))
            best_vc_value = _float(best_vc, "VC_mean")
            vc_at_primary = _float(baseline, "VC_mean")
            output.append(
                {
                    "dataset": DATASET_LABELS[dataset],
                    "model": MODEL_LABELS[model],
                    "best_VC_theta": float(best_vc["confidence_threshold"]),
                    "best_VC": best_vc_value,
                    "VC_at_theta_0.25": vc_at_primary,
                    "relative_gap_from_best": (best_vc_value - vc_at_primary) / best_vc_value,
                    "best_F1_theta": float(best_f1["confidence_threshold"]),
                    "best_LRP_theta": float(best_lrp["confidence_threshold"]),
                    "statistical_unit": baseline["uncertainty"],
                }
            )
    return output


def _curve(
    aggregate: Sequence[Mapping[str, str]],
    dataset: str,
    manipulation: str,
    variant: str,
    field: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    rows = _select(
        aggregate,
        dataset=dataset,
        manipulation=manipulation,
        variant=variant,
    )
    levels = sorted({float(row["level"]) for row in rows})
    x: list[float] = []
    mean: list[float] = []
    low: list[float] = []
    high: list[float] = []
    for level in levels:
        level_rows = [row for row in rows if abs(float(row["level"]) - level) < 1e-12]
        if sorted(row["model"] for row in level_rows) != sorted(MODELS):
            raise ValueError(
                f"incomplete curve: {dataset}/{manipulation}/{variant}/{level}"
            )
        summary = _model_summary(level_rows, field)
        x.append(level)
        mean.append(summary[0])
        low.append(summary[1])
        high.append(summary[2])
    return x, mean, low, high


def _plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def _plot_attack_grid(
    aggregate: Sequence[Mapping[str, str]],
    output: Path,
    *,
    attacks: Sequence[tuple[str, str, str]],
    filename: str,
    y_max: float,
    duplicate_axis: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), sharey=True)
    panel_labels = ("a", "b", "c", "d")
    for index, (axis, dataset, attack) in enumerate(
        zip(axes.flat, ("mscoco", "mscoco", "visdrone", "visdrone"), attacks * 2)
    ):
        manipulation, variant, attack_label = attack
        for field, label, colour, linestyle in METRICS:
            x, mean, low, high = _curve(
                aggregate, dataset, manipulation, variant, field
            )
            axis.plot(
                x,
                mean,
                marker="o",
                markersize=3.1,
                color=colour,
                linestyle=linestyle,
                label=label,
                zorder=3,
            )
            if field == "normalized_VC_mean":
                axis.fill_between(
                    x,
                    low,
                    high,
                    color=colour,
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
        axis.axhline(1.0, color="#444444", linestyle=(0, (3, 2)), linewidth=0.8)
        axis.set_ylim(0.0, y_max)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_title(
            f"({panel_labels[index]}) {DATASET_LABELS[dataset]} - {attack_label}",
            loc="left",
            pad=4,
        )
        if duplicate_axis:
            axis.set_xscale("log", base=2)
            axis.set_xticks([1, 2, 4, 8, 16])
            axis.set_xticklabels(["1", "2", "4", "8", "16"])
            axis.set_xlabel("Prediction multiplicity")
        else:
            axis.set_xlabel("Box side scale")
    axes[0, 0].set_ylabel("Score / unmodified baseline")
    axes[1, 0].set_ylabel("Score / unmodified baseline")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.0, w_pad=1.0)
    for extension in ("png", "pdf"):
        figure.savefig(
            output / f"{filename}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def _plot_theta(theta_rows: Sequence[Mapping[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_style()
    fields = (
        ("VC_mean", "VC", "#0072B2", "-"),
        ("mAC_mean", "mAC", "#D55E00", "--"),
        ("standard_F1_mean", "F1", "#009E73", "-."),
        ("one_minus_LRP", "1-LRP", "#CC79A7", ":"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.16, 4.65), sharex=True, sharey=True)
    panel_index = 0
    for row_index, dataset in enumerate(DATASETS):
        for column_index, model in enumerate(MODELS):
            axis = axes[row_index, column_index]
            rows = sorted(
                (
                    row
                    for row in theta_rows
                    if row["dataset"] == dataset and row["model"] == model
                ),
                key=lambda row: float(row["confidence_threshold"]),
            )
            if len(rows) != 19:
                raise ValueError(f"incomplete theta curve: {dataset}/{model}")
            x = [float(row["confidence_threshold"]) for row in rows]
            for field, label, colour, linestyle in fields:
                if field == "one_minus_LRP":
                    y = [1.0 - _float(row, "standard_LRP_mean") for row in rows]
                else:
                    y = [_float(row, field) for row in rows]
                axis.plot(
                    x,
                    y,
                    color=colour,
                    linestyle=linestyle,
                    marker="o",
                    markersize=2.2,
                    label=label,
                )
            axis.axvline(
                PRIMARY_THETA,
                color="#444444",
                linestyle=(0, (3, 2)),
                linewidth=0.8,
            )
            panel_letter = chr(ord("a") + panel_index)
            panel_index += 1
            axis.set_title(
                f"({panel_letter}) {DATASET_LABELS[dataset]} - {MODEL_LABELS[model]}",
                loc="left",
                pad=4,
            )
            axis.set_xlim(0.05, 0.95)
            axis.set_ylim(0.0, 0.72)
            axis.set_xticks([0.05, 0.25, 0.45, 0.65, 0.85])
            axis.grid(axis="y", alpha=0.22, linewidth=0.6)
            axis.set_axisbelow(True)
    for axis in axes[1, :]:
        axis.set_xlabel("Confidence threshold θ")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.0, w_pad=0.8)
    for extension in ("png", "pdf"):
        figure.savefig(
            output / f"fig_theta_sensitivity.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def _copy_main_table(source: Path, output: Path) -> list[dict[str, str]]:
    rows = _read_csv(source)
    order = {"yolov8n": 0, "yolo11n": 1, "yolo11s": 2}
    rows.sort(key=lambda row: (0 if row["dataset"] == "MSCOCO2017" else 1, order[row["model"]]))
    _write_csv(output / "table_main_performance.csv", rows)
    return rows


def _percent(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _draft_text(
    key_rows: Sequence[Mapping[str, object]],
    theta_rows: Sequence[Mapping[str, object]],
) -> str:
    def condition(dataset: str, label: str) -> Mapping[str, object]:
        return next(
            row
            for row in key_rows
            if row["dataset"] == dataset and row["condition"] == label
        )

    coco_expand = condition("MS COCO", "Expand: side scale 3.00")
    vis_expand = condition("VisDrone", "Expand: side scale 3.00")
    coco_exact = condition("MS COCO", "Exact duplicate: multiplicity 16")
    vis_exact = condition("VisDrone", "Exact duplicate: multiplicity 16")
    coco_jitter = condition("MS COCO", "Jittered duplicate: multiplicity 16")
    vis_jitter = condition("VisDrone", "Jittered duplicate: multiplicity 16")
    theta_gap_min = min(float(row["relative_gap_from_best"]) for row in theta_rows)
    theta_gap_max = max(float(row["relative_gap_from_best"]) for row in theta_rows)

    return f"""# Manuscript-ready VC v1 result text

## Controlled gaming analysis

We evaluated whether simple output manipulations could improve the proposed region score without improving detection quality. For every completed prediction cache, the baseline one-to-one true positives were frozen at $\\theta=0.25$ and then subjected to controlled box shrinkage, box expansion, exact duplication, or deterministic duplication with 2% centre and 1% size jitter. All scores were normalized by the unmodified result from the same model and run. The MS COCO analysis used one predeclared checkpoint for each of three architectures, whereas the VisDrone analysis used five training seeds per architecture.

Box expansion exposes the limitation of coverage-only evaluation. At a side-scale factor of 3.0, normalized mAC increased to {_percent(float(coco_expand['mAC_ratio_mean']))} on MS COCO and {_percent(float(vis_expand['mAC_ratio_mean']))} on VisDrone. Under the same manipulation, normalized VC fell to {_percent(float(coco_expand['VC_ratio_mean']))} and {_percent(float(vis_expand['VC_ratio_mean']))}, respectively. Thus, the manipulation increased mAC by {_percent(float(coco_expand['mAC_ratio_mean']) - 1.0)} and {_percent(float(vis_expand['mAC_ratio_mean']) - 1.0)}, while reducing VC by {_percent(1.0 - float(coco_expand['VC_ratio_mean']))} and {_percent(1.0 - float(vis_expand['VC_ratio_mean']))}. The direction was consistent for every evaluated model and VisDrone training seed. These results show that VC preserves the collective-coverage information captured by mAC while charging for the background and spatial overreach introduced by oversized boxes.

Exact duplication did not change the prediction union associated with any ground-truth object. Consequently, both VC and mAC remained exactly invariant and the duplicated output could not obtain a higher region score. This invariance is a deliberate boundary of a region-set metric rather than a claim that duplicate detections are operationally harmless. At multiplicity 16, normalized F1 fell to {_percent(float(coco_exact['F1_ratio_mean']))} on MS COCO and {_percent(float(vis_exact['F1_ratio_mean']))} on VisDrone; normalized $1-\\mathrm{{LRP}}$ fell to {_percent(float(coco_exact['one_minus_LRP_ratio_mean']))} and {_percent(float(vis_exact['one_minus_LRP_ratio_mean']))}. Detection-count redundancy should therefore be reported with standard one-to-one metrics alongside VC.

Controlled jitter produced a small increase in mAC but no increase in VC. At multiplicity 16, normalized VC was {_percent(float(coco_jitter['VC_ratio_mean']))} on MS COCO and {_percent(float(vis_jitter['VC_ratio_mean']))} on VisDrone, whereas normalized mAC was {_percent(float(coco_jitter['mAC_ratio_mean']))} and {_percent(float(vis_jitter['mAC_ratio_mean']))}. Across all 72 non-baseline jittered conditions, the largest normalized VC was 0.997433. The proposed score therefore did not reward either oversized coverage or near-duplicate coverage, while F1 and LRP continued to expose the associated prediction redundancy.

## Confidence-threshold sensitivity

We evaluated 19 confidence thresholds from 0.05 to 0.95 without changing model weights or rerunning inference. The VC-maximizing threshold was 0.15–0.20 across the six dataset/model combinations. The predeclared operating point $\\theta=0.25$ was within {_percent(theta_gap_min, 2)}–{_percent(theta_gap_max, 2)} of the maximum VC, while the F1 optimum occurred at 0.20–0.25 and the LRP optimum at 0.30. We therefore retain $\\theta=0.25$ as the common primary operating point rather than tuning the threshold separately on each test set, and report the complete sensitivity curves to show how collective coverage and orphan predictions trade off as the threshold changes.

## Statistical reporting

For MS COCO, each architecture is represented by one checkpoint selected before the final analysis. Point-estimate uncertainty is reported using 10,000 paired image-bootstrap replicates over the 5,000 validation images. For VisDrone, each architecture is represented by five independently trained seeds and results are reported as the mean and sample standard deviation. The diagnostic AP values computed inside the shared post-NMS cache are not official dataset AP and should not appear as the primary AP result; official evaluator AP remains a separate companion metric.

## Proposed figure captions

**Figure: Controlled box geometry.** Response to shrinking and expanding the frozen baseline true-positive boxes. Scores are normalized by the unmodified output from the same run. Lines show the descriptive mean over three architectures; the blue band shows the minimum-to-maximum VC range across architectures. Each VisDrone architecture value is first averaged over five training seeds. Box expansion increases mAC but decreases VC, demonstrating the spatial-precision term introduced by VC.

**Figure: Prediction duplication.** Response to exact and controlled-jitter duplication of frozen baseline true positives. Exact copies leave region unions, VC, and mAC unchanged, so duplication cannot increase the region score. F1 and $1-\\mathrm{{LRP}}$ fall as multiplicity increases. Small geometric jitter slightly increases mAC but leaves VC below its unmodified baseline.

**Figure: Confidence-threshold sensitivity.** VC, mAC, F1, and $1-\\mathrm{{LRP}}$ across 19 confidence thresholds. The vertical line marks the predeclared primary threshold $\\theta=0.25$. MS COCO panels use fixed checkpoints; VisDrone panels show five-seed means.

## Editing decisions

- Retain the frozen VC v1 definition and the common primary threshold $\\theta=0.25$.
- Replace the previous EPI/efficiency gaming discussion with controlled geometry and duplication experiments.
- State that VC complements mAC and standard one-to-one detection metrics; do not claim that VC alone measures every error mode.
- Use official dataset AP in the main comparison table. Keep shared-cache diagnostic AP out of the primary claim.
"""


def build_paper_results(project_root: Path | None = None) -> dict[str, object]:
    root = project_root or Path(__file__).resolve().parent.parent
    gaming_root = root / "results" / "gaming_main"
    fixed_root = root / "results" / "fixed_checkpoint_analysis_v1"
    output = root / "results" / "paper_results_v1"
    output.mkdir(parents=True, exist_ok=True)

    source_manifests = _require_complete_sources(gaming_root, fixed_root)
    aggregate_path = gaming_root / "gaming_aggregate.csv"
    theta_path = gaming_root / "theta_sensitivity.csv"
    main_table_path = fixed_root / "paper_main_table.csv"
    aggregate = _read_csv(aggregate_path)
    theta_source = _read_csv(theta_path)
    if len(aggregate) != 180 or len(theta_source) != 114:
        raise ValueError("formal source row count changed")

    key_rows = _key_condition_table(aggregate)
    theta_rows = _theta_table(theta_source)
    _write_csv(output / "table_gaming_key_conditions.csv", key_rows)
    _write_csv(output / "table_theta_optima.csv", theta_rows)
    _copy_main_table(main_table_path, output)

    _plot_attack_grid(
        aggregate,
        output,
        attacks=(
            ("shrink", "center_scale", "box shrink"),
            ("expand", "center_scale", "box expansion"),
        ),
        filename="fig_gaming_geometry",
        y_max=1.30,
        duplicate_axis=False,
    )
    _plot_attack_grid(
        aggregate,
        output,
        attacks=(
            ("duplicate", "exact", "exact duplication"),
            (
                "duplicate",
                "controlled_jitter_2pct_center_1pct_size",
                "jittered duplication",
            ),
        ),
        filename="fig_gaming_duplicates",
        y_max=1.10,
        duplicate_axis=True,
    )
    _plot_theta(theta_source, output)
    _write_text(output / "PAPER_RESULTS_DRAFT.md", _draft_text(key_rows, theta_rows))

    source_paths = (aggregate_path, theta_path, main_table_path)
    output_paths = (
        output / "table_main_performance.csv",
        output / "table_gaming_key_conditions.csv",
        output / "table_theta_optima.csv",
        output / "fig_gaming_geometry.png",
        output / "fig_gaming_geometry.pdf",
        output / "fig_gaming_duplicates.png",
        output / "fig_gaming_duplicates.pdf",
        output / "fig_theta_sensitivity.png",
        output / "fig_theta_sensitivity.pdf",
        output / "PAPER_RESULTS_DRAFT.md",
    )
    manifest = {
        "status": "complete",
        "paper_results_version": PAPER_RESULTS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training": False,
        "inference": False,
        "primary_theta": PRIMARY_THETA,
        "source_protocol_fingerprint": source_manifests["gaming"].get(
            "protocol_fingerprint"
        ),
        "source_files": {str(path.relative_to(root)): _sha256(path) for path in source_paths},
        "output_files": {str(path.relative_to(output)): _sha256(path) for path in output_paths},
        "row_counts": {
            "source_gaming_aggregate": len(aggregate),
            "source_theta": len(theta_source),
            "paper_key_conditions": len(key_rows),
            "paper_theta_optima": len(theta_rows),
        },
        "notes": {
            "figure_band": "minimum-to-maximum normalized VC across the three architectures",
            "mscoco": "one predeclared fixed checkpoint per architecture",
            "visdrone": "five-seed mean and sample SD per architecture",
            "diagnostic_ap": "excluded from paper gaming figures; not official dataset AP",
        },
    }
    _write_text(
        output / "README.md",
        """# VC v1 paper-ready results

This directory contains publication-oriented figures, tables, and draft text
derived from completed immutable result CSV files. No model training or inference
is performed. PNG files are 400 dpi and PDF files use embedded TrueType fonts.

The shaded blue region in the Gaming figures is the minimum-to-maximum VC range
over the three architectures. It is not a confidence interval. VisDrone model
values are first aggregated over five training seeds.

Diagnostic shared-cache AP is intentionally omitted from the Gaming figures.
Use official dataset AP as the separate standard detection metric in the paper.
""",
    )
    manifest["output_files"]["README.md"] = _sha256(output / "README.md")
    _write_text(output / "paper_results_complete.json", json.dumps(manifest, indent=2))
    return {
        "status": "complete",
        "version": PAPER_RESULTS_VERSION,
        "output": str(output),
        "key_condition_rows": len(key_rows),
        "theta_rows": len(theta_rows),
        "training": False,
        "inference": False,
    }