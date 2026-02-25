# epi_plot_stable.py
# Mean/std EPI curves across multiple seeds (robust to early stopping & missing epochs)
# Output under: ROOT_DIR/mean/

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
ROOT_DIR = RESULT_ROOT / "VisDrone"
DATASET_NAME = "VisDrone"                        # "CIFAR-100" / "VisDrone" / "BTD" etc.

MODEL_PREFIXES = ["yolov8n", "yolo11n", "yolo11s"]  # folders like yolov8n_s0 ... yolov8n_s4
MIN_SEEDS_PER_EPOCH = 1                             # set to 3 if you want stricter epochs

IN_FILE = "metrics/epi_results.csv"                 # relative file name to search

# Plot options
PLOT_STD_BAND = True


def _read_epi_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    # find epoch col
    epoch_col = None
    for c in df.columns:
        if c.lower() == "epoch":
            epoch_col = c
            break
    if epoch_col is None:
        raise ValueError(f"Missing 'epoch' column. Found: {list(df.columns)}")

    # find EPI col (robust)
    epi_col = None
    for c in df.columns:
        if c.strip().upper() == "EPI":
            epi_col = c
            break
    if epi_col is None:
        for c in df.columns:
            if c.lower() == "epi":
                epi_col = c
                break
    if epi_col is None:
        raise ValueError(f"Missing 'EPI' column. Found: {list(df.columns)}")

    out = df[[epoch_col, epi_col]].copy()
    out.rename(columns={epoch_col: "epoch", epi_col: "EPI"}, inplace=True)

    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
    out["EPI"] = pd.to_numeric(out["EPI"], errors="coerce")
    out = out.dropna(subset=["epoch", "EPI"])
    out["epoch"] = out["epoch"].astype(int)

    # If duplicate epochs exist, average them
    out = out.groupby("epoch", as_index=False)["EPI"].mean()
    return out.sort_values("epoch").reset_index(drop=True)


def _infer_model_seed_from_path(csv_path: Path) -> Tuple[str, str]:
    """
    infer from folder name: yolo11n_s3 -> (yolo11n, s3)
    """
    for p in csv_path.parents:
        m = re.match(r"^(yolov8n|yolo11n|yolo11s)_s(\d+)$", p.name, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower(), f"s{m.group(2)}"
    return "unknown", "unknown"


def _discover_files(root: Path) -> List[Path]:
    # search any **/metrics/epi_results.csv
    return sorted(root.rglob(IN_FILE))


def _merge_seeds(seed_dfs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Returns wide df indexed by epoch, each seed is a column.
    """
    merged = None
    for seed, df in seed_dfs:
        s = df.set_index("epoch")["EPI"].rename(seed)
        merged = s.to_frame() if merged is None else merged.join(s, how="outer")
    return merged.sort_index()


def _safe_mean_std(values_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    values_2d: shape [T, S], may contain NaN
    returns mean[T], std[T], n[T]
    std uses sample std (ddof=1) only if n>=2; else std=0 for n==1 and NaN for n==0.
    This avoids numpy warnings.
    """
    mask = ~np.isnan(values_2d)
    n = mask.sum(axis=1).astype(int)

    # mean
    sums = np.nansum(values_2d, axis=1)
    mean = np.where(n > 0, sums / np.maximum(n, 1), np.nan)

    # std (safe)
    std = np.full(values_2d.shape[0], np.nan, dtype=float)
    for i in range(values_2d.shape[0]):
        vi = values_2d[i, mask[i]]
        if vi.size >= 2:
            std[i] = float(np.std(vi, ddof=1))
        elif vi.size == 1:
            std[i] = 0.0
        else:
            std[i] = np.nan

    return mean, std, n.astype(float)


def compute_mean_curves(epi_files: List[Path]) -> Dict[str, pd.DataFrame]:
    by_model: Dict[str, List[Tuple[str, pd.DataFrame]]] = {m: [] for m in MODEL_PREFIXES}

    for f in epi_files:
        model, seed = _infer_model_seed_from_path(f)
        if model not in by_model:
            continue
        try:
            df = _read_epi_csv(f)
            by_model[model].append((seed, df))
        except Exception as e:
            print(f"[SKIP] {f}: {e}")

    out: Dict[str, pd.DataFrame] = {}
    for model, seed_dfs in by_model.items():
        if not seed_dfs:
            print(f"[WARN] No valid seeds for {model}.")
            continue

        wide = _merge_seeds(seed_dfs)
        vals = wide.to_numpy(dtype=float)

        mean, std, n = _safe_mean_std(vals)

        curve = pd.DataFrame({
            "epoch": wide.index.astype(int),
            "epi_mean": mean,
            "epi_std": std,
            "n_seeds": n
        })

        # enforce minimum seed count per epoch
        curve.loc[curve["n_seeds"] < MIN_SEEDS_PER_EPOCH, ["epi_mean", "epi_std"]] = np.nan

        out[model] = curve.reset_index(drop=True)
        print(f"[OK] {model}: {len(seed_dfs)} seeds, epochs={curve['epoch'].min()}..{curve['epoch'].max()}")

    return out


def plot_curves(curves: Dict[str, pd.DataFrame], out_png: Path) -> None:
    plt.figure()
    for model in MODEL_PREFIXES:
        if model not in curves:
            continue
        df = curves[model].dropna(subset=["epi_mean"])
        if df.empty:
            continue

        x = df["epoch"].to_numpy()
        y = df["epi_mean"].to_numpy()
        s = df["epi_std"].to_numpy()

        plt.plot(x, y, label=model)

        if PLOT_STD_BAND:
            ok = ~np.isnan(s)
            if ok.any():
                plt.fill_between(x[ok], (y - s)[ok], (y + s)[ok], alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("EPI")
    plt.title(f"Mean EPI Curve Across Seeds ({DATASET_NAME})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main() -> None:
    out_dir = ROOT_DIR / "mean"
    out_dir.mkdir(parents=True, exist_ok=True)

    epi_files = _discover_files(ROOT_DIR)
    if not epi_files:
        raise RuntimeError(f"No '{IN_FILE}' found under ROOT_DIR={ROOT_DIR}")

    print(f"Found epi_results.csv files: {len(epi_files)}")

    curves = compute_mean_curves(epi_files)
    if not curves:
        raise RuntimeError(
            "No model curves computed. Check:\n"
            "1) ROOT_DIR points to the folder containing '*_s0..s4' runs\n"
            "2) folder names look like 'yolo11n_s0'\n"
            f"ROOT_DIR={ROOT_DIR}"
        )

    # Save combined long CSV
    rows = []
    for model, df in curves.items():
        tmp = df.copy()
        tmp.insert(0, "model", model)
        rows.append(tmp)

    all_df = pd.concat(rows, ignore_index=True)
    out_csv = out_dir / f"{DATASET_NAME}_epi_mean_curve.csv"
    all_df.to_csv(out_csv, index=False)

    out_png = out_dir / f"{DATASET_NAME}_epi_mean_curve.png"
    plot_curves(curves, out_png)

    print("[DONE] Saved:")
    print(" ", out_csv)
    print(" ", out_png)


if __name__ == "__main__":
    main()
