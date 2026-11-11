"""Create the manuscript figure for the collective-fragmentation experiment.

This script reads the completed offline result table. It performs no training,
model inference, matching, or metric recomputation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "outputs" / "vc_v1" / "collective_fragmentation_v1"
SOURCE = SOURCE_ROOT / "fragmentation_runs.csv"
COMPLETE = SOURCE_ROOT / "collective_fragmentation_complete.json"
OUTPUT = ROOT / "outputs" / "vc_v1" / "paper_collective_fragmentation_v1"

DATASETS = {
    "mscoco": ("MS COCO", 3),
    "visdrone": ("VisDrone", 15),
}
PANELS = (
    ("fragmentation", "mscoco", r"(a) MS COCO: fixed union"),
    ("fragmentation", "visdrone", r"(b) VisDrone: fixed union"),
    ("retention", "mscoco", r"(c) MS COCO: retained support"),
    ("retention", "visdrone", r"(d) VisDrone: retained support"),
)
METRICS = (
    {
        "key": "normalized_mAC",
        "name": "mAC",
        "label": r"$mAC$",
        "colour": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.0,
        "zorder": 5,
    },
    {
        "key": "normalized_VC",
        "name": "VC",
        "label": r"$VC$",
        "colour": "#D55E00",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.0,
        "zorder": 6,
    },
    {
        "key": "normalized_F1",
        "name": "F1",
        "label": "F1",
        "colour": "#009E73",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 1.45,
        "zorder": 4,
    },
    {
        "key": "normalized_one_minus_LRP",
        "name": "1-LRP",
        "label": "1 - LRP",
        "colour": "#6F6F6F",
        "linestyle": "-.",
        "marker": "D",
        "linewidth": 1.4,
        "zorder": 3,
    },
    {
        "key": "normalized_AP50_95",
        "name": "mAP50:95",
        "label": r"$mAP_{50:95}$",
        "colour": "#7A5195",
        "linestyle": ":",
        "marker": "X",
        "linewidth": 1.55,
        "zorder": 7,
    },
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows():
    if not COMPLETE.is_file():
        raise FileNotFoundError(
            "The collective-fragmentation experiment is not complete: "
            f"{COMPLETE}"
        )
    completion = json.loads(COMPLETE.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError("Collective-fragmentation completion status is not complete")
    if int(completion.get("completed_runs", -1)) != 18:
        raise ValueError("Expected 18 completed source runs")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 162:
        raise ValueError(f"Expected 162 run rows, found {len(rows)}")
    for row in rows:
        for metric in METRICS:
            if row.get(metric["key"], "") == "":
                raise ValueError(
                    f"Missing {metric['key']} for "
                    f"{row['dataset']}/{row['model']}/seed{row['seed']}"
                )
    return rows, completion


def panel_data(rows, experiment, dataset, metric_key):
    selected = [
        row
        for row in rows
        if row["experiment"] == experiment and row["dataset"] == dataset
    ]
    expected_runs = DATASETS[dataset][1]
    levels = sorted({float(row["level"]) for row in selected})
    expected_levels = (
        [1.0, 4.0, 9.0, 16.0]
        if experiment == "fragmentation"
        else [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    if levels != expected_levels:
        raise ValueError(
            f"Unexpected level grid for {dataset}/{experiment}: {levels}"
        )
    means, minima, maxima = [], [], []
    for level in levels:
        values = [
            float(row[metric_key])
            for row in selected
            if float(row["level"]) == level
        ]
        if len(values) != expected_runs:
            raise ValueError(
                f"Expected {expected_runs} values for "
                f"{dataset}/{experiment}/{level}, found {len(values)}"
            )
        means.append(fmean(values))
        minima.append(min(values))
        maxima.append(max(values))
    return levels, means, minima, maxima


def style_axis(axis):
    axis.set_ylim(-0.035, 1.075)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.grid(axis="y", color="0.88", linewidth=0.55)
    axis.axhline(
        1.0, color="0.35", linestyle=(0, (3, 2)), linewidth=0.85, zorder=1
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.75, labelsize=7.5)


def create_figure(rows):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.2,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.15), sharey=True)
    plot_rows = []

    for axis, (experiment, dataset, title) in zip(axes.flat, PANELS):
        style_axis(axis)
        for metric in METRICS:
            levels, means, minima, maxima = panel_data(
                rows, experiment, dataset, metric["key"]
            )
            axis.fill_between(
                levels,
                minima,
                maxima,
                color=metric["colour"],
                alpha=0.075,
                linewidth=0,
                zorder=metric["zorder"] - 2,
            )
            axis.plot(
                levels,
                means,
                color=metric["colour"],
                linestyle=metric["linestyle"],
                marker=metric["marker"],
                linewidth=metric["linewidth"],
                markersize=4.0,
                markeredgewidth=0.6,
                markeredgecolor="white",
                zorder=metric["zorder"],
            )
            for level, mean, minimum, maximum in zip(
                levels, means, minima, maxima
            ):
                plot_rows.append(
                    {
                        "dataset": dataset,
                        "experiment": experiment,
                        "level": level,
                        "metric": metric["name"],
                        "n_runs": DATASETS[dataset][1],
                        "mean": mean,
                        "minimum": minimum,
                        "maximum": maximum,
                    }
                )

        axis.set_title(title, loc="left", weight="bold", pad=4)
        if experiment == "fragmentation":
            axis.set_xticks([1, 4, 9, 16])
            axis.set_xlabel(r"Number of fragments $K$")
        else:
            axis.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
            axis.set_xlabel("Retained support fraction ρ")

    axes[0, 0].set_ylabel("Score / controlled baseline")
    axes[1, 0].set_ylabel("Score / controlled baseline")
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    axes[0, 0].annotate(
        "regional scores preserve\nfixed union support",
        xy=(16, 1.0),
        xytext=(6.1, 0.77),
        fontsize=7.1,
        ha="left",
        va="top",
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4", linewidth=0.7),
    )
    axes[0, 0].annotate(
        "one-to-one scores\nsaturate near zero",
        xy=(4, 0.03),
        xytext=(7.0, 0.29),
        fontsize=7.1,
        ha="left",
        va="top",
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4", linewidth=0.7),
    )
    axes[1, 0].annotate(
        "graded regional response",
        xy=(0.6, 0.73),
        xytext=(0.27, 0.93),
        fontsize=7.1,
        ha="left",
        va="top",
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4", linewidth=0.7),
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=metric["colour"],
            linestyle=metric["linestyle"],
            marker=metric["marker"],
            linewidth=metric["linewidth"],
            markersize=4.2,
            label=metric["label"],
        )
        for metric in METRICS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        columnspacing=1.45,
        handlelength=2.5,
        handletextpad=0.55,
    )
    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.095, top=0.88, wspace=0.11, hspace=0.35
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT / "collective_fragmentation_response.pdf"
    png = OUTPUT / "collective_fragmentation_response.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, bbox_inches="tight", dpi=400)
    plt.close(figure)

    data_path = OUTPUT / "collective_fragmentation_plot_data.csv"
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    return pdf, png, data_path


def write_supporting_files(rows, completion, pdf, png, data_path):
    frag = [row for row in rows if row["experiment"] == "fragmentation"]
    max_mac_delta = max(abs(float(row["normalized_mAC"]) - 1.0) for row in frag)
    max_vc_delta = max(abs(float(row["normalized_VC"]) - 1.0) for row in frag)

    monotonic = {}
    retention = [row for row in rows if row["experiment"] == "retention"]
    identities = sorted(
        {
            (row["dataset"], row["model"], int(row["seed"]))
            for row in retention
        }
    )
    for metric in ("normalized_mAC", "normalized_VC"):
        passed = 0
        for identity in identities:
            subset = sorted(
                (
                    row
                    for row in retention
                    if (
                        row["dataset"],
                        row["model"],
                        int(row["seed"]),
                    )
                    == identity
                ),
                key=lambda row: float(row["retention"]),
            )
            values = [float(row[metric]) for row in subset]
            if all(
                values[index] <= values[index + 1] + 1e-12
                for index in range(len(values) - 1)
            ):
                passed += 1
        monotonic[metric] = {"passed": passed, "total": len(identities)}

    summary = {
        "source": str(SOURCE),
        "source_sha256": file_hash(SOURCE),
        "protocol_fingerprint": completion["protocol_fingerprint"],
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "plot_data": str(data_path),
        "aggregation": {
            "mscoco": "mean and range across 3 fixed model checkpoints",
            "visdrone": "mean and range across 15 model-seed runs",
        },
        "normalization": "each score divided by the controlled unsplit K=1 baseline",
        "fixed_union_max_absolute_normalized_mAC_deviation": max_mac_delta,
        "fixed_union_max_absolute_normalized_VC_deviation": max_vc_delta,
        "retention_monotonicity": monotonic,
        "training_or_inference_performed": False,
    }
    (OUTPUT / "figure_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    caption = (
        "Controlled collective fragmentation on MS COCO and VisDrone. "
        "Top: union support is fixed while the number of fragments K increases. "
        "Bottom: K=16 while the retained support fraction rho varies. "
        "Scores are normalized to the unsplit controlled baseline; lines and "
        "shaded bands show the mean and range across evaluated runs. "
        "One minus LRP is plotted so that higher values are better."
    )
    (OUTPUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# Collective-fragmentation manuscript figure\n\n"
        f"- Source: {SOURCE}\n"
        f"- Figure: {png.name} and {pdf.name}\n"
        f"- Plot data: {data_path.name}\n"
        "- No training, inference, matching, or metric recomputation is "
        "performed by the plotting script.\n"
        "- The bands are descriptive run ranges, not confidence intervals.\n",
        encoding="utf-8",
    )


def main():
    rows, completion = load_rows()
    pdf, png, data_path = create_figure(rows)
    write_supporting_files(rows, completion, pdf, png, data_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "source_rows": len(rows),
                "pdf": str(pdf),
                "png": str(png),
                "plot_data": str(data_path),
                "training": False,
                "inference": False,
            }
        )
    )


if __name__ == "__main__":
    main()