# vc_curve_plot_ide.py
# Run this file directly in your IDE (PyCharm/VSCode). No terminal args needed.

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT


# =========================
# User Config (EDIT HERE)
# =========================
# This root folder should contain folders like:
#   yolo11n_s0, yolo11n_s1, ..., yolo11s_s4, yolov8n_s4
ROOT_DIR = RESULT_ROOT / "MRI"

# Output folder
OUT_DIR = ROOT_DIR / "mean"

# Which models and seeds to aggregate
MODELS = ["yolov8n", "yolo11n", "yolo11s"]
SEEDS = [0, 1, 2, 3, 4]

# VC column in vc_epoch.csv (must exist): "VC50" or "VC50:95"
VC_COL = "VC50"

# If True, also generate a single plot comparing all models (mean only + std shading)
PLOT_MODEL_COMPARISON = True

# Numerical stability for plotting bounds if needed
# (not used in metric computation; your VC already computed in vc_epoch.csv)
# =========================


def read_vc_epoch(run_dir: Path, vc_col: str) -> pd.DataFrame:
    """
    Read <run_dir>/metrics/vc_epoch.csv with at least columns: epoch, vc_col.
    Returns dataframe with columns: epoch (int), vc (float).
    """
    p = run_dir / "metrics" / "vc_epoch.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    df = pd.read_csv(p)

    if "epoch" not in df.columns:
        raise KeyError(f"'epoch' not found in {p}. Columns={list(df.columns)}")

    if vc_col not in df.columns:
        # Try common fallbacks
        fallbacks = ["VC50", "VC50:95", "VC50_95", "VC5095"]
        raise KeyError(
            f"'{vc_col}' not found in {p}. Available={list(df.columns)}. "
            f"Try one of: {fallbacks}"
        )

    out = df[["epoch", vc_col]].copy()
    out = out.rename(columns={vc_col: "vc"})
    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce").astype("Int64")
    out["vc"] = pd.to_numeric(out["vc"], errors="coerce")
    out = out.dropna(subset=["epoch"]).sort_values("epoch")
    out["epoch"] = out["epoch"].astype(int)
    return out


def merge_seeds_by_epoch(seed_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Outer-merge by epoch to handle early stopping.
    Output columns: epoch, vc_s0, vc_s1, ...
    """
    merged = None
    for i, df in enumerate(seed_dfs):
        df_i = df.rename(columns={"vc": f"vc_s{i}"})
        merged = df_i if merged is None else pd.merge(merged, df_i, on="epoch", how="outer")

    merged = merged.sort_values("epoch").reset_index(drop=True)
    return merged


def compute_mean_std(merged: pd.DataFrame) -> pd.DataFrame:
    vc_cols = [c for c in merged.columns if c.startswith("vc_s")]
    merged["vc_mean"] = merged[vc_cols].mean(axis=1, skipna=True)
    merged["vc_std"] = merged[vc_cols].std(axis=1, ddof=1, skipna=True)
    merged["n_seeds"] = merged[vc_cols].notna().sum(axis=1)
    return merged


def plot_mean_std_curve(df_stat: pd.DataFrame, title: str, out_png: Path, label: str = "VC mean"):
    x = df_stat["epoch"].to_numpy()
    y = df_stat["vc_mean"].to_numpy(dtype=float)
    s = df_stat["vc_std"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(x, y, label=label)

    # Shade ±1 std (treat NaN std as 0)
    s0 = np.nan_to_num(s, nan=0.0)
    plt.fill_between(x, y - s0, y + s0, alpha=0.2, label="±1 std")

    plt.xlabel("Epoch")
    plt.ylabel("VC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_model_comparison(all_model_stats: dict, title: str, out_png: Path):
    """
    all_model_stats: {model_name: df_stat} where df_stat has epoch, vc_mean, vc_std
    Plot mean curves for all models with their own std shading.
    """
    plt.figure()

    for model, df_stat in all_model_stats.items():
        x = df_stat["epoch"].to_numpy()
        y = df_stat["vc_mean"].to_numpy(dtype=float)
        s = df_stat["vc_std"].to_numpy(dtype=float)
        s0 = np.nan_to_num(s, nan=0.0)

        plt.plot(x, y, label=f"{model} mean")
        plt.fill_between(x, y - s0, y + s0, alpha=0.15)

    plt.xlabel("Epoch")
    plt.ylabel("VC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}  # model -> df_stat

    for model in MODELS:
        seed_dfs = []
        used_seeds = []

        for s in SEEDS:
            run_dir = ROOT_DIR / f"{model}_s{s}"
            try:
                df = read_vc_epoch(run_dir, VC_COL)
                seed_dfs.append(df)
                used_seeds.append(s)
            except Exception as e:
                print(f"[SKIP] {model}_s{s}: {e}")

        if len(seed_dfs) == 0:
            print(f"[WARN] No valid seeds found for model={model}. Skipping.")
            continue

        merged = merge_seeds_by_epoch(seed_dfs)
        stat = compute_mean_std(merged)

        # Save mean/std by epoch
        out_csv = OUT_DIR / f"{model}_{VC_COL.replace(':','_')}_mean_std_by_epoch.csv"
        stat.to_csv(out_csv, index=False)

        # Save plot for this model
        out_png = OUT_DIR / f"{model}_{VC_COL.replace(':','_')}_curve.png"
        title = f"{model} on Brain Tumor Detection  : {VC_COL} mean±std over seeds {used_seeds}"
        plot_mean_std_curve(stat, title, out_png, label=f"{model} VC mean")

        all_stats[model] = stat
        print(f"[OK] {model}: wrote\n  {out_csv}\n  {out_png}")

    # Optional: one figure comparing all models
    if PLOT_MODEL_COMPARISON and len(all_stats) >= 2:
        out_png = OUT_DIR / f"model_compare_{VC_COL.replace(':','_')}_curve.png"
        title = f"Brain Tumor Detection : {VC_COL} mean±std (5 seeds) model comparison"
        plot_model_comparison(all_stats, title, out_png)
        print(f"[OK] Model comparison plot: {out_png}")

    print("[DONE]")


if __name__ == "__main__":
    main()
