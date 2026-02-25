from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT

EPS = 1e-12  # ε_s for numerical stability


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case-insensitive) from candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def compute_vc(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """
    Add VC columns to df:
      - VC50: harmonic mean of (mAC, mAP50-like)
      - VC50_95: harmonic mean of (mAC, mAP50:95-like) if exists
    """
    # mAP50-like column: try strict names first, then fallback to "mAP"
    col_map50 = find_col(df, ["mAP50", "map50", "mAP@0.5", "mAP@50", "mAP"])
    col_mac = find_col(df, ["mAC", "mac", "mAC@0.5", "mAC@50"])

    if col_map50 is None or col_mac is None:
        raise ValueError(f"Required columns not found. Columns={list(df.columns)}")

    mAP = df[col_map50].astype(float)
    mAC = df[col_mac].astype(float)

    df = df.copy()
    df["VC50"] = (2.0 * mAP * mAC) / (mAP + mAC + eps)

    # Optional mAP50:95-like column
    col_map5095 = find_col(df, ["mAP50:95", "map50:95", "mAP@0.5:0.95", "map5095", "mAP_50_95", "mAP50-95"])
    if col_map5095 is not None:
        mAP5095 = df[col_map5095].astype(float)
        df["VC50_95"] = (2.0 * mAP5095 * mAC) / (mAP5095 + mAC + eps)

    return df


def process_one_seed(metrics_dir: Path, out_name: str = "vc_results.csv") -> Path:
    epi_csv = metrics_dir / "epi_results.csv"
    if not epi_csv.exists():
        raise FileNotFoundError(f"Missing: {epi_csv}")

    df = pd.read_csv(epi_csv)
    df_vc = compute_vc(df, eps=EPS)

    out_path = metrics_dir / out_name
    df_vc.to_csv(out_path, index=False)
    return out_path


def batch_process(root_dir: str,
                  models: list[str],
                  seeds: list[int],
                  out_name: str = "vc_results.csv",
                  make_global_summary: bool = True) -> None:
    root = Path(root_dir)

    summary_rows = []

    for model in models:
        for s in seeds:
            exp_dir = root / f"{model}_s{s}" / "metrics"
            try:
                out_path = process_one_seed(exp_dir, out_name=out_name)
                print(f"[OK] {model}_s{s} -> {out_path}")

                # Optional: also record best-VC epoch for quick checking
                df_out = pd.read_csv(out_path)
                if "epoch" in df_out.columns and "VC50" in df_out.columns:
                    best_idx = df_out["VC50"].astype(float).idxmax()
                    best_row = df_out.loc[best_idx]
                    summary_rows.append({
                        "model": model,
                        "seed": s,
                        "best_epoch_by_VC50": int(best_row["epoch"]),
                        "best_VC50": float(best_row["VC50"]),
                        "best_mAP_col": float(best_row[find_col(df_out, ["mAP50","map50","mAP@0.5","mAP@50","mAP"])]),
                        "best_mAC_col": float(best_row[find_col(df_out, ["mAC","mac","mAC@0.5","mAC@50"])]),
                    })

            except Exception as e:
                print(f"[SKIP] {model}_s{s}: {e}")

    # Optional: write a global summary at root
    if make_global_summary and summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values(["model", "seed"])
        sum_path = root / "vc_summary_best_by_seed.csv"
        df_sum.to_csv(sum_path, index=False)
        print(f"[OK] Global summary -> {sum_path}")


if __name__ == "__main__":
    # 你的根目录
    ROOT = str(RESULT_ROOT / "MRI")

    MODELS = ["yolo11n", "yolo11s", "yolov8n"]
    SEEDS = [0, 1, 2, 3, 4]

    batch_process(
        root_dir=ROOT,
        models=MODELS,
        seeds=SEEDS,
        out_name="vc_results.csv",          # 每个 metrics 文件夹生成这个文件
        make_global_summary=True            # 额外生成一个总表（可关）
    )
