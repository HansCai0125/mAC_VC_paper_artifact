# epi_plot.py
# Compute mean/std EPI curves across 5 seeds (handles early-stopping safely)
# Output: mean CSV + figure under ROOT_DIR/mean

from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT


# =========================
# User config (edit here)
# =========================
ROOT_DIR = RESULT_ROOT / "cifar100"
OUT_DIR = ROOT_DIR / "mean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We will group by these model prefixes:
MODEL_PREFIXES = ["yolov8n", "yolo11n", "yolo11s"]

# Minimum number of seeds required to plot/keep a mean value at an epoch
MIN_SEEDS_PER_EPOCH = 1  # set to 3 if you want stricter stability


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_epi_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _normalize_cols(df)

    # Flexible column names
    # Expect at least: epoch, EPI
    col_epoch = None
    for c in df.columns:
        if c.lower() == "epoch":
            col_epoch = c
            break
    if col_epoch is None:
        raise ValueError(f"[{csv_path}] missing 'epoch' column. Found: {list(df.columns)}")

    col_epi = None
    for c in df.columns:
        if c.strip().upper() == "EPI":
            col_epi = c
            break
    if col_epi is None:
        # sometimes it might be "epi" or "epi_value"
        for c in df.columns:
            if c.lower() == "epi":
                col_epi = c
                break
    if col_epi is None:
        raise ValueError(f"[{csv_path}] missing 'EPI' column. Found: {list(df.columns)}")

    out = df[[col_epoch, col_epi]].copy()
    out.rename(columns={col_epoch: "epoch", col_epi: "EPI"}, inplace=True)

    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
    out["EPI"] = pd.to_numeric(out["EPI"], errors="coerce")

    out = out.dropna(subset=["epoch", "EPI"])
    out["epoch"] = out["epoch"].astype(int)

    # If there are duplicate epochs, aggregate them (mean)
    out = out.groupby("epoch", as_index=False)["EPI"].mean()
    return out.sort_values("epoch").reset_index(drop=True)


def _infer_model_and_seed(csv_path: Path) -> Tuple[str, str]:
    """
    From a path like .../yolo11n_s3/metrics/epi_results.csv
    infer model='yolo11n', seed='s3'
    """
    parts = [p.name for p in csv_path.parents]
    # find folder like yolo11n_s3
    for name in parts:
        m = re.match(r"^(yolov8n|yolo11n|yolo11s)_s(\d+)$", name, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower(), f"s{m.group(2)}"
    # fallback: unknown bucket
    return "unknown", "unknown"


def _discover_epi_files(root: Path) -> List[Path]:
    # recursive search: any */metrics/epi_results.csv
    return sorted(root.rglob("metrics/epi_results.csv"))


def compute_mean_curve_per_model(epi_files: List[Path]) -> Dict[str, pd.DataFrame]:
    """
    Returns dict: model -> dataframe with columns:
      epoch, mean, std, n
    """
    by_model: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}

    for f in epi_files:
        model, seed = _infer_model_and_seed(f)
        if model not in MODEL_PREFIXES:
            # skip unrelated experiments if any
            continue
        try:
            df = _read_epi_csv(f)
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
            continue
        by_model.setdefault(model, []).append((seed, df))

    results: Dict[str, pd.DataFrame] = {}

    for model, seed_dfs in by_model.items():
        if not seed_dfs:
            continue

        # Outer-join by epoch across seeds
        merged = None
        for seed, df in seed_dfs:
            s = df.set_index("epoch")["EPI"].rename(seed)
            merged = s.to_frame() if merged is None else merged.join(s, how="outer")

        merged = merged.sort_index()

        # Compute mean/std ignoring NaNs
        values = merged.to_numpy(dtype=float)  # shape [epochs, seeds]
        n = np.sum(~np.isnan(values), axis=1)

        mean = np.nanmean(values, axis=1)
        std = np.nanstd(values, axis=1, ddof=1)  # sample std (ddof=1)
        std = np.where(n >= 2, std, 0.0)         # if only 1 seed, std=0

        out = pd.DataFrame({
            "epoch": merged.index.astype(int),
            "epi_mean": mean,
            "epi_std": std,
            "n_seeds": n
        })

        # Optionally drop epochs with too few seeds
        out.loc[out["n_seeds"] < MIN_SEEDS_PER_EPOCH, ["epi_mean", "epi_std"]] = np.nan

        results[model] = out.reset_index(drop=True)

        print(f"[OK] {model}: {len(seed_dfs)} seeds, epochs={out['epoch'].min()}..{out['epoch'].max()}")

    return results


def plot_mean_curves(model_curves: Dict[str, pd.DataFrame], out_png: Path) -> None:
    plt.figure()
    for model in MODEL_PREFIXES:
        if model not in model_curves:
            print(f"[WARN] No valid seeds for {model}.")
            continue

        df = model_curves[model].dropna(subset=["epi_mean"])
        if df.empty:
            print(f"[WARN] {model} curve is empty after filtering.")
            continue

        x = df["epoch"].to_numpy()
        y = df["epi_mean"].to_numpy()
        s = df["epi_std"].to_numpy()

        plt.plot(x, y, label=model)
        plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("EPI")
    plt.title("Mean EPI Curve Across 5 Seeds (CIFAR-100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main() -> None:
    epi_files = _discover_epi_files(ROOT_DIR)
    if not epi_files:
        raise RuntimeError(f"No epi_results.csv found under ROOT_DIR={ROOT_DIR}")

    print(f"Found epi_results.csv files: {len(epi_files)}")

    model_curves = compute_mean_curve_per_model(epi_files)
    if not model_curves:
        raise RuntimeError(
            "No model stats computed. Likely ROOT_DIR is wrong or folder names are not like '*_s0..s4'.\n"
            f"Current ROOT_DIR={ROOT_DIR}"
        )

    # Save a single combined CSV (long format)
    rows = []
    for model, df in model_curves.items():
        tmp = df.copy()
        tmp.insert(0, "model", model)
        rows.append(tmp)
    all_df = pd.concat(rows, ignore_index=True)
    out_csv = OUT_DIR / "cifar100_epi_mean_curve.csv"
    all_df.to_csv(out_csv, index=False)

    out_png = OUT_DIR / "cifar100_epi_mean_curve.png"
    plot_mean_curves(model_curves, out_png)

    print(f"[DONE] Saved:\n  {out_csv}\n  {out_png}")


if __name__ == "__main__":
    main()
