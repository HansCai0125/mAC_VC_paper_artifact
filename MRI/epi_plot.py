# robust_mean_curve_plot.py
# Stable mean/std curve across seeds with spike removal + trimmed mean
# Run in IDE directly.

from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT


# =========================
# User config (EDIT HERE)
# =========================
ROOT_DIR = RESULT_ROOT / "MRI"
DATASET_NAME = "Brain Tumor Detection"                         # used in output filenames / plot title

MODEL_PREFIXES = ["yolov8n", "yolo11n", "yolo11s"]  # run folder names: <model>_s0 ... <model>_s4
SEED_REGEX = re.compile(r"^(" + "|".join(MODEL_PREFIXES) + r")_s(\d+)$", re.IGNORECASE)

# Choose what to plot:
#   - "EPI"      -> read metrics/epi_results.csv, column "EPI"
#   - "VC50"     -> read metrics/vc_epoch.csv,   column "VC50"
#   - "VC50:95"  -> read metrics/vc_epoch.csv,   column "VC50:95"
METRIC_NAME = "EPI"

# Where to read the metric from
EPI_FILE_REL = "metrics/epi_results.csv"
VC_FILE_REL = "metrics/vc_epoch.csv"

# Epoch column candidates (robust)
EPOCH_COL_CANDS = ["epoch", "Epoch", "      epoch"]

# Metric column candidates mapping (robust)
METRIC_COL_CANDS = {
    "EPI": ["EPI", "epi"],
    "VC50": ["VC50", "vc50", "VC_50", "vc_50"],
    "VC50:95": ["VC50:95", "VC50-95", "vc50:95", "vc50-95", "VC_50:95"],
}

# Keep an epoch only if it has >= this many seeds contributing (after cleaning)
MIN_SEEDS_PER_EPOCH = 1

# -------- Robustness knobs (for the "spike" problem) --------
# 1) Per-seed spike removal using IQR rule: outside [Q1 - k*IQR, Q3 + k*IQR] => set to NaN
CLEAN_SPIKES_PER_SEED = True
IQR_K = 5.0  # larger => less aggressive. 3~6 usually OK for EPI.

# 2) Per-epoch trimmed mean across seeds (recommended when you have 5 seeds)
# For 5 seeds: trim_k=1 means drop max & min each epoch before mean/std.
USE_TRIMMED_MEAN = True
TRIM_K = 1

# Plot
PLOT_STD_BAND = True
FIG_DPI = 300


# =========================
# Helpers
# =========================
def _find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # exact
    for c in candidates:
        if c in df.columns:
            return c

    # case/space-insensitive fallback
    norm = {str(col).strip().lower().replace(" ", ""): col for col in df.columns}
    for c in candidates:
        key = str(c).strip().lower().replace(" ", "")
        if key in norm:
            return norm[key]
    return None


def _infer_model_seed_from_path(p: Path) -> Tuple[str, str]:
    # find folder like yolo11n_s3 in parents
    for parent in p.parents:
        m = SEED_REGEX.match(parent.name)
        if m:
            return m.group(1).lower(), f"s{m.group(2)}"
    return "unknown", "unknown"


def _discover_metric_files(root: Path) -> List[Path]:
    if METRIC_NAME.upper() == "EPI":
        rel = EPI_FILE_REL
    else:
        rel = VC_FILE_REL
    return sorted(root.rglob(rel))


def _read_metric_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    col_epoch = _find_first_existing_column(df, EPOCH_COL_CANDS)
    if col_epoch is None:
        # fallback: use row index as epoch
        df.insert(0, "epoch", np.arange(len(df), dtype=int))
        col_epoch = "epoch"

    metric_key = METRIC_NAME
    if metric_key not in METRIC_COL_CANDS:
        raise ValueError(f"Unsupported METRIC_NAME={METRIC_NAME}. Supported: {list(MET႔RIC_COL_CANDS.keys())}")

    col_metric = _find_first_existing_column(df, METRIC_COL_CANDS[metric_key])
    if col_metric is None:
        raise ValueError(f"[{csv_path}] missing metric column for {METRIC_NAME}. Found: {list(df.columns)}")

    out = df[[col_epoch, col_metric]].copy()
    out.rename(columns={col_epoch: "epoch", col_metric: "value"}, inplace=True)

    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["epoch", "value"])
    out["epoch"] = out["epoch"].astype(int)

    # duplicate epochs => mean
    out = out.groupby("epoch", as_index=False)["value"].mean()
    return out.sort_values("epoch").reset_index(drop=True)


def _iqr_clean_series(values: np.ndarray, k: float) -> np.ndarray:
    """Return cleaned copy where outliers are set to NaN (IQR rule)."""
    x = values.astype(float).copy()
    finite = np.isfinite(x)
    if finite.sum() < 4:
        return x  # too few points to define IQR meaningfully

    q1 = np.nanpercentile(x[finite], 25.0)
    q3 = np.nanpercentile(x[finite], 75.0)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return x

    low = q1 - k * iqr
    high = q3 + k * iqr
    outlier = (x < low) | (x > high)
    x[outlier] = np.nan
    return x


def _merge_seeds(seed_dfs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Wide DF: index=epoch, columns=seed."""
    merged = None
    for seed, df in seed_dfs:
        s = df.set_index("epoch")["value"].rename(seed)
        merged = s.to_frame() if merged is None else merged.join(s, how="outer")
    return merged.sort_index()


def _robust_epoch_stats(row_vals: np.ndarray) -> Tuple[float, float, int]:
    """Compute (mean, std, n) for one epoch across seeds with optional trimming."""
    v = row_vals.astype(float)
    v = v[np.isfinite(v)]
    n0 = int(v.size)
    if n0 == 0:
        return np.nan, np.nan, 0
    if not USE_TRIMMED_MEAN or TRIM_K <= 0 or n0 < (2 * TRIM_K + 2):
        # normal mean/std
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1)) if n0 >= 2 else 0.0
        return mean, std, n0

    # trimmed: drop TRIM_K smallest & TRIM_K largest
    v_sorted = np.sort(v)
    v_trim = v_sorted[TRIM_K: n0 - TRIM_K]
    n = int(v_trim.size)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(v_trim))
    std = float(np.std(v_trim, ddof=1)) if n >= 2 else 0.0
    return mean, std, n


def compute_mean_curves(metric_files: List[Path]) -> Dict[str, pd.DataFrame]:
    by_model: Dict[str, List[Tuple[str, pd.DataFrame]]] = {m: [] for m in MODEL_PREFIXES}

    for f in metric_files:
        model, seed = _infer_model_seed_from_path(f)
        if model not in by_model:
            continue
        try:
            df = _read_metric_csv(f)
            by_model[model].append((seed, df))
        except Exception as e:
            print(f"[SKIP] {f}: {e}")

    out: Dict[str, pd.DataFrame] = {}

    for model, seed_dfs in by_model.items():
        if not seed_dfs:
            print(f"[WARN] No valid seeds for {model}.")
            continue

        wide = _merge_seeds(seed_dfs)  # index=epoch, columns=seeds
        vals = wide.to_numpy(dtype=float)

        # per-seed spike cleaning (column-wise)
        if CLEAN_SPIKES_PER_SEED:
            for j in range(vals.shape[1]):
                vals[:, j] = _iqr_clean_series(vals[:, j], IQR_K)

        means = np.full(vals.shape[0], np.nan, dtype=float)
        stds = np.full(vals.shape[0], np.nan, dtype=float)
        ns = np.zeros(vals.shape[0], dtype=int)

        for i in range(vals.shape[0]):
            mu, sd, n = _robust_epoch_stats(vals[i, :])
            means[i] = mu
            stds[i] = sd
            ns[i] = n

        curve = pd.DataFrame({
            "epoch": wide.index.astype(int),
            f"{METRIC_NAME}_mean": means,
            f"{METRIC_NAME}_std": stds,
            "n_seeds": ns
        })

        # enforce minimum seed count
        curve.loc[curve["n_seeds"] < MIN_SEEDS_PER_EPOCH, [f"{METRIC_NAME}_mean", f"{METRIC_NAME}_std"]] = np.nan

        out[model] = curve.reset_index(drop=True)
        print(f"[OK] {model}: seeds={len(seed_dfs)}, epochs={curve['epoch'].min()}..{curve['epoch'].max()}")

    return out


def plot_curves(curves: Dict[str, pd.DataFrame], out_png: Path) -> None:
    plt.figure()

    for model in MODEL_PREFIXES:
        if model not in curves:
            continue
        df = curves[model].dropna(subset=[f"{METRIC_NAME}_mean"])
        if df.empty:
            continue

        x = df["epoch"].to_numpy()
        y = df[f"{METRIC_NAME}_mean"].to_numpy()
        s = df[f"{METRIC_NAME}_std"].to_numpy()

        plt.plot(x, y, label=model)

        if PLOT_STD_BAND:
            ok = np.isfinite(s)
            if ok.any():
                plt.fill_between(x[ok], (y - s)[ok], (y + s)[ok], alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel(METRIC_NAME)
    plt.title(f"Mean {METRIC_NAME} Curve Across Seeds ({DATASET_NAME})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI)
    plt.close()


def main() -> None:
    out_dir = ROOT_DIR / "mean"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_files = _discover_metric_files(ROOT_DIR)
    if not metric_files:
        rel = EPI_FILE_REL if METRIC_NAME.upper() == "EPI" else VC_FILE_REL
        raise RuntimeError(f"No '{rel}' found under ROOT_DIR={ROOT_DIR}")

    print(f"Found files: {len(metric_files)}")

    curves = compute_mean_curves(metric_files)
    if not curves:
        raise RuntimeError("No model curves computed. Check ROOT_DIR and folder naming like 'yolo11n_s0'.")

    # Save combined long CSV
    rows = []
    for model, df in curves.items():
        tmp = df.copy()
        tmp.insert(0, "model", model)
        rows.append(tmp)
    all_df = pd.concat(rows, ignore_index=True)

    out_csv = out_dir / f"{DATASET_NAME}_{METRIC_NAME}_mean_curve.csv"
    out_png = out_dir / f"{DATASET_NAME}_{METRIC_NAME}_mean_curve.png"

    all_df.to_csv(out_csv, index=False)
    plot_curves(curves, out_png)

    print("[DONE] Saved:")
    print(" ", out_csv)
    print(" ", out_png)


if __name__ == "__main__":
    main()
