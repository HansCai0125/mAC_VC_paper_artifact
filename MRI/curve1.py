# map mac epi
"""
Multi-seed curve summary script.
- Read CSV files from directories such as yolo11n_s0~s4 and yolo11s_s0~s4.
- For each model, plot:
    * mAP@0.5 / mAC@0.5 / EPI vs epoch (mean ± std).
- Then compare mean curves of yolo11n and yolo11s on shared plots.
"""

import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT

# ================== Configuration ==================

# Root directory containing run folders such as yolo11n_s0, yolo11s_s0, etc.
ROOT_DIR = RESULT_ROOT / "MRI"

# Model names to process (folder name prefix)
MODELS = ["yolo11n", "yolo11s"]  # Keep ["yolo11n"] if you only need one model

# Seeds
SEEDS = [0, 1, 2, 3, 4]

# Output directory for plots
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
#   Helpers: read CSV -> {epoch -> value}
# =========================================================

def _find_col(cols, keyword):
    """Find the column index that exactly matches keyword; raise if missing."""
    for i, c in enumerate(cols):
        if c.strip() == keyword:
            return i
    raise KeyError(f"Cannot find a column containing keyword '{keyword}' in {cols}")


def read_metric_csv(path: Path, key_col: str, val_col: str):
    """
    Read two columns from path: key_col (epoch) and val_col (metric).
    Return dict: {epoch(int) -> value(float)}.
    """
    if not path.exists():
        print(f"[WARN] File does not exist: {path}")
        return {}

    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[WARN] Empty file: {path}")
            return {}

        key_idx = _find_col(header, key_col)
        val_idx = _find_col(header, val_col)

        for row in reader:
            if not row or all((c.strip() == "" for c in row)):
                continue
            try:
                epoch = int(float(row[key_idx]))
            except ValueError:
                continue
            try:
                val_str = row[val_idx].strip()
                if val_str == "" or val_str.lower() == "nan":
                    continue
                value = float(val_str)
            except ValueError:
                continue
            out[epoch] = value

    return out


# =========================================================
#   Collect per-seed metrics -> dict(seed -> {epoch -> value})
# =========================================================

def collect_metric_dicts_for_model(model_name: str):
    """
    For a given model_name (e.g., 'yolo11n'), collect:
      - mAP50: from results.csv
      - mAC:   from metrics/mac_epoch.csv
      - EPI:   from metrics/epi_results.csv
    Return (map50_dicts, mac_dicts, epi_dicts), where map50_dicts[seed] = {epoch -> value}.
    """
    map50_dicts = {}
    mac_dicts = {}
    epi_dicts = {}

    print(f"\n====== Model: {model_name} ======")

    for seed in SEEDS:
        run_dir = ROOT_DIR / f"{model_name}_s{seed}"
        metrics_dir = run_dir / "metrics"

        if not run_dir.exists():
            print(f"[WARN] Directory does not exist, skip this seed: {run_dir}")
            continue

        print(f"[INFO] Reading seed {seed}, directory: {run_dir}")

        # 1) mAP@0.5 from results.csv column 'metrics/mAP50(B)'
        res_path = run_dir / "results.csv"
        m_map = read_metric_csv(res_path, key_col="epoch", val_col="metrics/mAP50(B)")
        if not m_map:
            print(f"[WARN] No mAP data for this seed: {res_path}")
        else:
            map50_dicts[seed] = m_map

        # 2) mAC@0.5 from metrics/mac_epoch.csv column 'mAC'
        mac_path = metrics_dir / "mac_epoch.csv"
        m_mac = read_metric_csv(mac_path, key_col="epoch", val_col="mAC")
        if not m_mac:
            print(f"[WARN] No mAC data for this seed: {mac_path}")
        else:
            mac_dicts[seed] = m_mac

        # 3) EPI from metrics/epi_results.csv column 'EPI'
        epi_path = metrics_dir / "epi_results.csv"
        m_epi = read_metric_csv(epi_path, key_col="epoch", val_col="EPI")
        if not m_epi:
            print(f"[WARN] No EPI data for this seed: {epi_path}")
        else:
            epi_dicts[seed] = m_epi

    return map50_dicts, mac_dicts, epi_dicts


# =========================================================
#   Compute mean / std
# =========================================================

def compute_mean_std(dicts_per_seed):
    """
    Input: dict(seed -> {epoch -> value})
    Output: (epochs_sorted, mean_arr, std_arr)
    """
    if not dicts_per_seed:
        return np.array([]), np.array([]), np.array([])

    all_epochs = sorted({e for d in dicts_per_seed.values() for e in d.keys()})

    epochs_sorted = []
    mean_vals = []
    std_vals = []

    for e in all_epochs:
        vals = [d[e] for d in dicts_per_seed.values() if e in d]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        epochs_sorted.append(e)
        mean_vals.append(arr.mean())
        std_vals.append(arr.std(ddof=0))

    return np.array(epochs_sorted, dtype=int), np.array(mean_vals), np.array(std_vals)


# =========================================================
#   Single model: seed curves + mean ± std
# =========================================================

def plot_metric_single(
    epochs,
    mean_arr,
    std_arr,
    dicts_per_seed,
    ylabel,
    title,
    out_path,
    show_seeds=True,
    drop_first_n=0,
    clip_upper_percentile=None,
):
    """
    show_seeds: whether to draw thin lines for each seed
    drop_first_n: drop first N epochs (e.g., EPI epoch 0 may be an outlier)
    clip_upper_percentile: cap y-axis upper bound by percentile (e.g., 99)
    """
    if epochs.size == 0:
        print(f"[WARN] No data, skip plotting: {out_path}")
        return

    # Drop initial epochs if requested
    if drop_first_n > 0:
        mask = epochs >= drop_first_n
        epochs = epochs[mask]
        mean_arr = mean_arr[mask]
        std_arr = std_arr[mask]

    plt.figure(figsize=(8, 4))

    # Colors
    seed_colors = ["#A6CEE3", "#FDBF6F", "#B2DF8A", "#FB9A99", "#CAB2D6"]
    mean_color = "#1F78B4"
    shade_color = "#9EC9FF"

    # Draw per-seed thin lines
    if show_seeds:
        for i, (s, d) in enumerate(sorted(dicts_per_seed.items(), key=lambda x: x[0])):
            xs = sorted(e for e in d.keys() if e >= drop_first_n)
            if not xs:
                continue
            ys = [d[e] for e in xs]
            c = seed_colors[i % len(seed_colors)]
            plt.plot(xs, ys, color=c, alpha=0.5, linewidth=1.0, label=f"seed {s}")

    # Draw mean curve
    plt.plot(epochs, mean_arr, color=mean_color, linewidth=2.5, label="mean")

    # Std shading
    upper = mean_arr + std_arr
    lower = mean_arr - std_arr
    plt.fill_between(epochs, lower, upper, color=shade_color, alpha=0.25, label="+/-1 std")

    # Clip upper peaks (useful for EPI outliers)
    if clip_upper_percentile is not None:
        finite_vals = mean_arr[np.isfinite(mean_arr)]
        if len(finite_vals):
            ymax = float(np.percentile(finite_vals, clip_upper_percentile))
            ymin = min(0.0, float(np.nanmin(mean_arr)))
            plt.ylim(ymin, ymax)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] {out_path}")


# =========================================================
#   Multiple models: mean-curve comparison
# =========================================================

def plot_compare_mean(
    metric_name: str,
    model_results: dict,
    ylabel: str,
    title: str,
    out_path: Path,
    drop_first_n: int = 0,
    clip_upper_percentile=None,
):
    """
    model_results: { model_name: (epochs, mean_arr, std_arr) }
    Only draw each model's mean curve (no seed lines / no shaded std).
    """

    # First find common epoch intersection across all models
    epoch_sets = []
    for (epochs, mean_arr, std_arr) in model_results.values():
        if epochs.size == 0:
            continue
        if drop_first_n > 0:
            mask = epochs >= drop_first_n
            e = set(epochs[mask])
        else:
            e = set(epochs)
        if e:
            epoch_sets.append(e)

    if not epoch_sets:
        print(f"[WARN] No valid epoch set, cannot plot comparison: {out_path}")
        return

    common_epochs = sorted(set.intersection(*epoch_sets))
    if not common_epochs:
        print(f"[WARN] Models have no common epochs, cannot plot comparison: {out_path}")
        return

    common_epochs = np.array(common_epochs, dtype=int)

    # Build y-values for each model
    ys_all = {}
    for model_name, (epochs, mean_arr, std_arr) in model_results.items():
        if epochs.size == 0:
            continue
        mapping = {int(e): float(v) for e, v in zip(epochs, mean_arr)}
        ys = [mapping[e] for e in common_epochs]
        ys_all[model_name] = np.array(ys, dtype=float)

    plt.figure(figsize=(8, 4))

    # Model colors
    model_colors = {
        "yolo11n": "#1F78B4",  # blue
        "yolo11s": "#33A02C",  # green
    }

    for model_name, ys in ys_all.items():
        color = model_colors.get(model_name, None)
        plt.plot(common_epochs, ys, linewidth=2.0, label=model_name, alpha=0.95, color=color)

    # Clip peak values (primarily for EPI)
    if clip_upper_percentile is not None:
        finite_vals = np.concatenate([v[np.isfinite(v)] for v in ys_all.values()])
        if len(finite_vals):
            ymax = float(np.percentile(finite_vals, clip_upper_percentile))
            ymin = min(0.0, float(np.nanmin(finite_vals)))
            plt.ylim(ymin, ymax)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] {out_path}")


# =========================================================
#   Main
# =========================================================

def main():
    # Save mean/std per model for later cross-model comparison
    all_model_stats = {}

    for model_name in MODELS:
        map50_dicts, mac_dicts, epi_dicts = collect_metric_dicts_for_model(model_name)

        # ---------- mAP@0.5 ----------
        epochs_map, mean_map, std_map = compute_mean_std(map50_dicts)
        plot_metric_single(
            epochs_map,
            mean_map,
            std_map,
            map50_dicts,
            ylabel="mAP@0.5",
            title=f"{model_name} - mAP@0.5 vs Epoch (mean +/- std over seeds)",
            out_path=PLOTS_DIR / f"{model_name}_mAP50_curve.png",
            show_seeds=True,
            drop_first_n=0,
            clip_upper_percentile=None,
        )

        # ---------- mAC@0.5 ----------
        epochs_mac, mean_mac, std_mac = compute_mean_std(mac_dicts)
        plot_metric_single(
            epochs_mac,
            mean_mac,
            std_mac,
            mac_dicts,
            ylabel="mAC@0.5",
            title=f"{model_name} - mAC@0.5 vs Epoch (mean +/- std over seeds)",
            out_path=PLOTS_DIR / f"{model_name}_mAC50_curve.png",
            show_seeds=True,
            drop_first_n=0,
            clip_upper_percentile=None,
        )

        # ---------- EPI ----------
        epochs_epi, mean_epi, std_epi = compute_mean_std(epi_dicts)
        plot_metric_single(
            epochs_epi,
            mean_epi,
            std_epi,
            epi_dicts,
            ylabel="EPI",
            title=f"{model_name} - EPI vs Epoch (mean +/- std over seeds)",
            out_path=PLOTS_DIR / f"{model_name}_EPI_curve.png",
            show_seeds=False,  # For EPI, only show mean +/- std
            drop_first_n=1,  # Drop epoch 0
            clip_upper_percentile=99,  # Clip top 1% extreme values
        )

        # Store stats for later model-vs-model plots
        all_model_stats[model_name] = {
            "map": (epochs_map, mean_map, std_map),
            "mac": (epochs_mac, mean_mac, std_mac),
            "epi": (epochs_epi, mean_epi, std_epi),
        }

    # ============ yolo11n vs yolo11s mean-curve comparison ============

    # Keep only models we care about
    compare_models = [m for m in ["yolo11n", "yolo11s"] if m in all_model_stats]

    if len(compare_models) >= 2:
        # mAP@0.5
        model_results_map = {m: all_model_stats[m]["map"] for m in compare_models}
        plot_compare_mean(
            metric_name="mAP@0.5",
            model_results=model_results_map,
            ylabel="mAP@0.5",
            title="YOLO11n vs YOLO11s - mAP@0.5 (mean over seeds)",
            out_path=PLOTS_DIR / "compare_mAP50_yolo11n_vs_yolo11s.png",
            drop_first_n=0,
            clip_upper_percentile=None,
        )

        # mAC@0.5
        model_results_mac = {m: all_model_stats[m]["mac"] for m in compare_models}
        plot_compare_mean(
            metric_name="mAC@0.5",
            model_results=model_results_mac,
            ylabel="mAC@0.5",
            title="YOLO11n vs YOLO11s - mAC@0.5 (mean over seeds)",
            out_path=PLOTS_DIR / "compare_mAC50_yolo11n_vs_yolo11s.png",
            drop_first_n=0,
            clip_upper_percentile=None,
        )

        # EPI
        model_results_epi = {m: all_model_stats[m]["epi"] for m in compare_models}
        plot_compare_mean(
            metric_name="EPI",
            model_results=model_results_epi,
            ylabel="EPI",
            title="YOLO11n vs YOLO11s - EPI (mean over seeds)",
            out_path=PLOTS_DIR / "compare_EPI_yolo11n_vs_yolo11s.png",
            drop_first_n=1,  # Also drop epoch 0 here
            clip_upper_percentile=99,  # Clip extreme peaks for readability
        )

    print("\n[FINISHED] All model curves and YOLO11n vs YOLO11s comparison plots are generated.")


if __name__ == "__main__":
    main()
