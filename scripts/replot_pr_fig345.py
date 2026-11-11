"""Run directly in VS Code: redraw current manuscript Figures 3, 4 and 5.

Uses the original plotting functions and CSVs; changes presentation only.
Outputs go to outputs/vc_v1/pr_fig345_readable, never the manuscript folder.
Requires matplotlib and numpy in the selected Python environment.
"""
from pathlib import Path
import importlib.util
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Annotation

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/vc_v1/pr_fig345_readable_v2"
DATA = ROOT / "outputs/vc_v1/fig56_dense_20261023_v1"
# Designed for approximately 12 cm printed width. Use width=\linewidth in TeX.
WIDTH_IN = 4.72


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finish(fig, number, name):
    heights = {3: 3.25, 4: 3.75, 5: 2.05}
    fig.set_size_inches(WIDTH_IN, heights[number])
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if number != 3:
        styles = dense.STYLE if number == 4 else dense.STYLE[:4]
        handles = dense.handles(styles)
        upright = {"mAC": "mAC", "VC": "VC", "F1": "F1",
                   "one_minus_LRP": r"$1-\mathrm{LRP}$",
                   "AP50_95": r"diagnostic $\mathrm{mAP}_{50:95}$"}
        labels = [upright[entry[0]] for entry in styles]
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=8, pad=2, length=2.5)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        for location in ("left", "center", "right"):
            title = ax.get_title(loc=location)
            if title:
                ax.set_title(title.replace("MS COCO", "MS-COCO"),
                             loc=location, fontsize=8, pad=5)
    if number == 3:
        # Include RT-DETR-L, which occurs only in the VisDrone panels.
        by_label = {}
        for ax in fig.axes:
            hs, ls = ax.get_legend_handles_labels()
            by_label.update(zip(ls, hs))
        labels = list(by_label)
        handles = [by_label[label] for label in labels]
        for ax in fig.axes:
            ax.set_xticks([.05, .25, .45, .65, .95])
        fig.subplots_adjust(left=.13, right=.98, bottom=.15, top=.78,
                            hspace=.48, wspace=.30)
    elif number == 4:
        for i, ax in enumerate(fig.axes):
            # Original fill_between ranges are real min/max bands; increase
            # contrast only, never widen their numerical extent.
            for band in ax.collections:
                band.set_alpha(.22)
            for artist in list(ax.texts):
                if isinstance(artist, Annotation):
                    artist.remove()
            if i % 2 == 0:
                ax.set_ylabel("Normalized score", fontsize=8)
            ax.set_xlabel(r"Fragments $K$" if i < 2 else r"Retained support $\rho$",
                          fontsize=8)
        fig.subplots_adjust(left=.13, right=.98, bottom=.13, top=.79,
                            hspace=.95, wspace=.22)
    else:
        titles = ["(a) Box scaling", "(b) Orphan injection", "(c) Ownership"]
        xlabels = [r"Box scale $s$", "Orphans per GT", "Gap / target width"]
        for ax, title, label in zip(fig.axes, titles, xlabels):
            ax.set_title(title, fontsize=8, pad=6)
            ax.set_xlabel(label, fontsize=8)
        # Show fewer tick labels, preserving all evaluated points and markers.
        fig.axes[0].set_xticks([.5, 1, 2, 3], labels=["0.5", "1", "2", "3"])
        fig.axes[1].set_xticks([0, 1, 4, 16], labels=["0", "1", "4", "16"])
        fig.subplots_adjust(left=.10, right=.985, bottom=.25, top=.72,
                            wspace=.25)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .995),
               ncol={3: 2, 4: 3, 5: 4}[number], frameon=False, fontsize=8,
               columnspacing=.9, handlelength=1.8, handletextpad=.4)
    # Fixed canvas retains the intended physical size and font sizes.
    for suffix in ("pdf", "png"):
        target = OUTPUT / f"{name}.{suffix}"
        fig.savefig(target, dpi=400, facecolor="white")
        print(target)
    plt.close(fig)


def main():
    global dense
    OUTPUT.mkdir(parents=True, exist_ok=True)
    theta = load("pr_theta_source", "region_eval/paper_figures_ivc.py")
    dense = load("pr_dense_source", "scripts/fig56_dense_20261023/plot_figures.py")
    dense.setup()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    theta._save = lambda fig, name: finish(fig, 3, name)
    theta._plot_theta_sensitivity(theta._theta_rows())
    for name in ("fig5_fragmentation_dense.csv", "fig6_canonical_dense.csv"):
        shutil.copy2(DATA / name, OUTPUT / name)
    dense.save = lambda fig, out, name: finish(
        fig, 4 if name.startswith("fig5_") else 5, name)
    dense.fig5(OUTPUT)
    dense.fig6(OUTPUT)
    print("Done. Original data, plots and manuscript files were not changed.")


if __name__ == "__main__":
    main()
