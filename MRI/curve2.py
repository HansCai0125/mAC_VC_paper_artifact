# vc comparison correlation
# Multi-model / multi-dataset mAP vs mAC scatter plot
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT

# ================== Configuration ==================

# Root directory per dataset (must contain folders like yolo11n_s0, yolo11s_s0)
DATASET_ROOTS = {
    # "CaWeed10": Path(r"E:\Project\Database\CaWeed10"),
    "BTD": RESULT_ROOT / "MRI",
}

# Models to compare
MODELS = ["yolo11n", "yolo11s"]

# Seeds to use
SEEDS = [0, 1, 2, 3, 4]

# Output directory
OUT_DIR = RESULT_ROOT / "MRI" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== Utility functions ==================


def find_col_fuzzy(header, keywords):
    """
    Find column in header:
    1) exact match first
    2) then substring match
    keywords can be str or [str, str, ...]
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    # exact match
    for k in keywords:
        if k in header:
            return header.index(k)

    # substring match
    header_lower = [h.lower() for h in header]
    for k in keywords:
        k_low = k.lower()
        for i, h in enumerate(header_lower):
            if k_low in h:
                return i

    raise KeyError(f"Cannot find a column containing keywords {keywords}, header={header}")


def read_vc_epoch_csv(csv_path: Path):
    """
    Read metrics/vc_epoch.csv
    Return list[dict], each row includes:
      - epoch (if missing, use row index)
      - mAP50, mAC, VC_mAP50, VC_mAC
    """
    if not csv_path.exists():
        print(f"[WARN] vc_epoch.csv not found: {csv_path}")
        return []

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[WARN] Empty file: {csv_path}")
            return []

        header = [h.strip() for h in header]

        # Try to find epoch column (if present)
        try:
            idx_epoch = find_col_fuzzy(header, "epoch")
        except KeyError:
            idx_epoch = None  # fallback to row index

        # mAP50 / mAC / VC_mAP50 / VC_mAC
        idx_mAC = find_col_fuzzy(header, "mAC")
        idx_VC_mAC = find_col_fuzzy(header, "VC_mAC")
        idx_mAP50 = find_col_fuzzy(header, "mAP50")
        idx_VC_mAP50 = find_col_fuzzy(header, "VC_mAP50")

        for row_idx, row in enumerate(reader):
            if not row or all(c.strip() == "" for c in row):
                continue

            # epoch: prefer epoch column, otherwise row index
            if idx_epoch is not None:
                try:
                    epoch = int(float(row[idx_epoch]))
                except ValueError:
                    epoch = row_idx  # fallback
            else:
                epoch = row_idx  # start from 0

            try:
                mAC = float(row[idx_mAC])
                VC_mAC = float(row[idx_VC_mAC])
                mAP50 = float(row[idx_mAP50])
                VC_mAP50 = float(row[idx_VC_mAP50])
            except ValueError:
                continue

            rows.append(
                {
                    "epoch": epoch,
                    "mAC": mAC,
                    "VC_mAC": VC_mAC,
                    "mAP50": mAP50,
                    "VC_mAP50": VC_mAP50,
                }
            )

    return rows


# ================== Collect all points ==================


def collect_points():
    """
    Iterate DATASET_ROOTS / model_s{seed} / metrics/vc_epoch.csv
    Collect all points:
    (dataset, model, seed, epoch, mAP50, mAC, VC_mAP50, VC_mAC)
    """
    all_points = []

    for dataset_name, root in DATASET_ROOTS.items():
        print(f"\n[DATASET] {dataset_name}  root={root}")

        for model in MODELS:
            for seed in SEEDS:
                run_dir = root / f"{model}_s{seed}"
                csv_path = run_dir / "metrics" / "vc_epoch.csv"

                if not csv_path.exists():
                    print(f"  [WARN] {dataset_name} / {model}_s{seed} has no vc_epoch.csv, skip")
                    continue

                rows = read_vc_epoch_csv(csv_path)
                print(f"  [INFO] {dataset_name} / {model}_s{seed}: read {len(rows)} rows")

                for r in rows:
                    all_points.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "seed": seed,
                            "epoch": r["epoch"],
                            "mAP50": r["mAP50"],
                            "mAC": r["mAC"],
                            "VC_mAP50": r["VC_mAP50"],
                            "VC_mAC": r["VC_mAC"],
                        }
                    )

    print(f"\n[SUMMARY] Total points collected: {len(all_points)}")
    return all_points


# ================== Plot mAP vs mAC scatter & compute correlation ==================


def plot_map_vs_mac(points):
    """
    Figure 3: mAP@0.5 vs mAC@0.5
    - Color: dataset (CaWeed10 vs BTD)
    - Marker: model (YOLO11n vs YOLO11s)
    """
    if not points:
        print("[WARN] No points to plot, return directly")
        return

    # Color and marker mapping
    dataset_colors = {
        "CaWeed10": "tab:blue",
        "BTD": "tab:orange",
    }
    default_colors = ["tab:green", "tab:red", "tab:purple", "tab:brown"]

    model_markers = {
        "yolo11n": "o",
        "yolo11s": "s",
    }
    default_markers = ["^", "D", "v", "X"]

    # Group by (dataset, model)
    grouped = {}
    for p in points:
        key = (p["dataset"], p["model"])
        grouped.setdefault(key, []).append((p["mAP50"], p["mAC"]))

    # Correlation per dataset
    corr_per_dataset = {}
    for dataset_name in DATASET_ROOTS.keys():
        xs = [p["mAP50"] for p in points if p["dataset"] == dataset_name]
        ys = [p["mAC"] for p in points if p["dataset"] == dataset_name]
        if len(xs) > 1:
            r = float(np.corrcoef(xs, ys)[0, 1])
            corr_per_dataset[dataset_name] = r
        else:
            corr_per_dataset[dataset_name] = np.nan

    print("\n[Correlation] mAP@0.5 vs mAC@0.5:")
    for ds, r in corr_per_dataset.items():
        if np.isfinite(r):
            print(f"  {ds}: r = {r:.4f}")
        else:
            print(f"  {ds}: r = NaN (too few points)")

    if "CaWeed10" in corr_per_dataset and "BTD" in corr_per_dataset:
        r1 = corr_per_dataset["CaWeed10"]
        r2 = corr_per_dataset["BTD"]
        if np.isfinite(r1) and np.isfinite(r2):
            delta_corr = r2 - r1
            print(f"\n  DeltaCorr (BTD - CaWeed10) = {delta_corr:+.4f}")

    # ========= Plot =========
    plt.figure(figsize=(6, 6))

    color_pool_idx = 0
    marker_pool_idx = 0

    for (dataset_name, model_name), vals in grouped.items():
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]

        # Color
        if dataset_name in dataset_colors:
            color = dataset_colors[dataset_name]
        else:
            color = default_colors[color_pool_idx % len(default_colors)]
            color_pool_idx += 1

        # Marker
        if model_name in model_markers:
            marker = model_markers[model_name]
        else:
            marker = default_markers[marker_pool_idx % len(default_markers)]
            marker_pool_idx += 1

        label = f"{dataset_name} - {model_name}"
        plt.scatter(xs, ys, c=color, marker=marker, alpha=0.6, label=label, edgecolors="none")

    plt.xlabel("mAP@0.5")
    plt.ylabel("mAC@0.5")
    plt.title("mAP@0.5 vs mAC@0.5 (All models & seeds)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    out_path = OUT_DIR / "scatter_mAP50_vs_mAC50_all_models.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n[SAVE] {out_path}")


# ================== main ==================


def main():
    points = collect_points()
    plot_map_vs_mac(points)


if __name__ == "__main__":
    main()
