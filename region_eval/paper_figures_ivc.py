"""Draft publication figures for Section IV-C from completed VC results.

Run this file directly in PyCharm or with the project Python interpreter. It
only reads existing result CSVs and writes PNG/PDF figures under outputs/vc_v1.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "vc_v1" / "paper_figures_ivc_draft"
DATASETS = (
    ("mscoco", "MS COCO", ("yolov8n", "yolo11n", "yolo11s")),
    ("visdrone", "VisDrone", ("yolov8n", "yolo11n", "yolo11s", "rtdetr_l")),
)
MODEL_NAMES = {
    "yolov8n": "YOLOv8n",
    "yolo11n": "YOLO11n",
    "yolo11s": "YOLO11s",
    "rtdetr_l": "RT-DETR-L",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def _primary_rows() -> tuple[dict, dict]:
    formal = _rows(ROOT / "outputs/vc_v1/formal_matrix_v1/aggregate_primary.csv")
    rtdetr = _rows(
        ROOT / "outputs/vc_v1/paper_results_rtdetr_v1/rtdetr_aggregate_primary.csv"
    )
    primary = {(row["dataset"], row["model"]): row for row in formal}
    primary[("visdrone", "rtdetr_l")] = next(
        row for row in rtdetr if row["dataset"] == "visdrone"
    )

    archived_ap = _rows(
        ROOT / "msi_Git_disk/Results/table_metrics_mean_std_numeric.csv"
    )
    ap = {}
    for row in archived_ap:
        dataset = {"MSCOCO": "mscoco", "VisDrone": "visdrone"}.get(row["dataset"])
        if dataset is None or row["n"] != "5":
            continue
        model = {"yolo8n": "yolov8n", "RTDETR-l": "rtdetr_l"}.get(
            row["model"], row["model"]
        )
        ap[(dataset, model)] = row

    expected = {(dataset, model) for dataset, _, models in DATASETS for model in models}
    if not expected.issubset(primary) or not expected.issubset(ap):
        raise ValueError("The seven Table II model rows are incomplete")
    return primary, ap


def _theta_rows() -> dict[tuple[str, str], list[dict[str, str]]]:
    formal = _rows(ROOT / "outputs/vc_v1/formal_matrix_v1/aggregate_thresholds.csv")
    rtdetr = _rows(
        ROOT / "outputs/vc_v1/paper_results_rtdetr_v1/rtdetr_aggregate_thresholds.csv"
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in formal:
        grouped.setdefault((row["dataset"], row["model"]), []).append(row)
    grouped[("visdrone", "rtdetr_l")] = [
        row for row in rtdetr if row["dataset"] == "visdrone"
    ]
    for dataset, _, models in DATASETS:
        for model in models:
            curve = grouped[(dataset, model)]
            curve.sort(key=lambda row: float(row["confidence_threshold"]))
            if len(curve) != 19:
                raise ValueError(f"Expected 19 thresholds for {dataset}/{model}")
    return grouped


def _save(figure, name: str) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(
            OUTPUT / f"{name}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )


def _plot_main_metrics(primary: dict, ap: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = (
        (r"mAP$_{50:95}$", "mAP50-95", "#ffffff", "///"),
        ("mAC", "mAC", "#bcbcbc", "..."),
        ("VC", "VC", "#555555", ""),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True)
    for index, (axis, (dataset, label, models)) in enumerate(zip(axes, DATASETS)):
        x = np.arange(len(models), dtype=float)
        width = 0.24
        for metric_index, (metric_label, field, color, hatch) in enumerate(metrics):
            values = []
            deviations = []
            for model in models:
                row = ap[(dataset, model)] if field.startswith("mAP") else primary[(dataset, model)]
                values.append(float(row[f"{field}_mean"]))
                deviations.append(float(row[f"{field}_std"]))
            axis.bar(
                x + (metric_index - 1) * width,
                values,
                width,
                yerr=deviations,
                capsize=2,
                color=color,
                edgecolor="black",
                linewidth=0.65,
                hatch=hatch,
                error_kw={"elinewidth": 0.65, "capthick": 0.65},
                label=metric_label,
                zorder=3,
            )
        axis.set_title(f"({chr(97 + index)}) {label}", loc="left", pad=4)
        axis.set_xticks(x, [MODEL_NAMES[model] for model in models])
        axis.set_ylim(0.0, 0.8)
        axis.grid(axis="y", color="#dddddd", linewidth=0.55)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.03))
    figure.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.3)
    _save(figure, "fig_ivc_main_metrics")
    plt.close(figure)


def _plot_theta_sensitivity(curves: dict) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "yolov8n": ("#111111", "-", "o"),
        "yolo11n": ("#666666", "--", "s"),
        "yolo11s": ("#222222", "-.", "^"),
        "rtdetr_l": ("#888888", (0, (1.5, 1.5)), "D"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.6), sharex=True)
    handles: dict[str, object] = {}
    for row_index, (dataset, dataset_label, models) in enumerate(DATASETS):
        for column_index, (field, metric_label) in enumerate(
            (("mAC_mean", "mAC"), ("VC_mean", "VC"))
        ):
            axis = axes[row_index, column_index]
            for model in models:
                curve = curves[(dataset, model)]
                color, linestyle, marker = styles[model]
                (line,) = axis.plot(
                    [float(row["confidence_threshold"]) for row in curve],
                    [float(row[field]) for row in curve],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=2,
                    markersize=3,
                    linewidth=1.45,
                    label=MODEL_NAMES[model],
                    zorder=3,
                )
                handles[model] = line
            axis.axvline(0.25, color="#aaaaaa", linestyle=":", linewidth=0.9,
                         zorder=1)
            axis.grid(axis="y", color="#dddddd", linewidth=0.55)
            axis.set_axisbelow(True)
            axis.set_title(
                f"({chr(97 + row_index * 2 + column_index)}) "
                f"{dataset_label}: {metric_label}",
                loc="left",
                pad=4,
            )
            axis.set_xlim(0.05, 0.95)
            axis.set_ylim(0.0, 0.85 if column_index == 0 else 0.55)
            axis.set_xticks((0.05, 0.25, 0.45, 0.65, 0.85, 0.95))
            if column_index == 0:
                axis.set_ylabel("Metric value")
            if row_index == 1:
                axis.set_xlabel(r"Confidence threshold $\theta$")
    order = ("yolov8n", "yolo11n", "yolo11s", "rtdetr_l")
    figure.legend(
        [handles[model] for model in order],
        [MODEL_NAMES[model] for model in order],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0, w_pad=1.0)
    _save(figure, "fig_ivc_theta_sensitivity")
    plt.close(figure)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    primary, ap = _primary_rows()
    _plot_main_metrics(primary, ap)
    _plot_theta_sensitivity(_theta_rows())
    for name in ("fig_ivc_main_metrics", "fig_ivc_theta_sensitivity"):
        print(OUTPUT / f"{name}.png")
        print(OUTPUT / f"{name}.pdf")


if __name__ == "__main__":
    main()
