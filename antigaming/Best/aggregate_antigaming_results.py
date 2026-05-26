"""
Aggregate and summarize anti-gaming experiment CSV files.

Expected directory pattern (flexible, recursive):
<root>/Best/<dataset>/<model>/<attack>/anti_gaming/*.csv

Outputs:
1) combined_all_rows.csv
2) combo_baseline_and_endpoints.csv
3) attack_consistency_summary.csv
4) severity_mean_curves.csv
5) paper_ready_notes.txt
6) anti_gaming_summary.xlsx
7) plots/*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# USER CONFIG
# =========================
ROOT_DIR = Path(r"F:\mAC_VC_EPI\antigaming")
SEARCH_SUBDIR = "Best"
OUTPUT_DIRNAME = "_summary"
MAKE_PLOTS = True

# Grouping key strategy:
# - model is often a checkpoint name (e.g., "best.pt") in CSVs, so default to path-model.
# - keep attack from CSV by default so baseline rows remain correct.
PREFER_PATH_FOR_DATASET = True
PREFER_PATH_FOR_MODEL = True
PREFER_PATH_FOR_ATTACK = False


def normalize_attack_name(s: str) -> str:
    s0 = str(s).strip().lower()
    mapping = {
        "shrink attack": "shrink",
        "shrink": "shrink",
        "expand": "expand",
        "duplicate": "duplicate",
        "threshold": "threshold",
        "baseline": "baseline",
    }
    return mapping.get(s0, s0)


def normalize_attack_series(series: pd.Series) -> pd.Series:
    return series.map(lambda x: normalize_attack_name(x) if pd.notna(x) else pd.NA)


def coalesce_label(csv_series: pd.Series, path_value: Optional[str], prefer_path: bool) -> pd.Series:
    out = csv_series.copy()
    if prefer_path and path_value is not None:
        return pd.Series(path_value, index=csv_series.index)
    if path_value is not None:
        out = out.fillna(path_value)
    return out


def find_best_subpath_parts(csv_path: Path, anchor: str = "Best") -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = list(csv_path.parts)
    if anchor not in parts:
        return None, None, None

    idx = parts.index(anchor)
    dataset = parts[idx + 1] if len(parts) > idx + 1 else None
    model = parts[idx + 2] if len(parts) > idx + 2 else None
    attack = parts[idx + 3] if len(parts) > idx + 3 else None
    return dataset, model, attack


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def norm_by_baseline(row_val: float, base_val: float) -> float:
    if pd.isna(row_val) or pd.isna(base_val):
        return np.nan
    if abs(base_val) < 1e-12:
        return np.nan
    return float(row_val) / float(base_val)


def pick_endpoint_severity(df_attack: pd.DataFrame, attack: str):
    sev = pd.to_numeric(df_attack["severity"], errors="coerce")
    if attack in {"shrink", "threshold"}:
        idx = sev.idxmin()
    else:
        idx = sev.idxmax()
    return df_attack.loc[idx]


def severity_rank_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["severity"] = pd.to_numeric(out["severity"], errors="coerce")
    out["attack_progress"] = np.nan

    for _, g in out.groupby(["dataset", "model", "attack"], dropna=False):
        attack = str(g["attack"].iloc[0]).lower()
        sev = g["severity"].astype(float)

        smin, smax = sev.min(), sev.max()
        if pd.isna(smin) or pd.isna(smax) or abs(smax - smin) < 1e-12:
            out.loc[g.index, "attack_progress"] = 0.0
            continue

        if attack in {"shrink", "threshold"}:
            prog = (smax - sev) / (smax - smin)
        else:
            prog = (sev - smin) / (smax - smin)

        out.loc[g.index, "attack_progress"] = prog.values

    return out


def load_all_csvs(root_dir: Path, search_subdir: str = "Best") -> pd.DataFrame:
    base = root_dir / search_subdir
    if not base.exists():
        raise FileNotFoundError(f"Cannot find folder: {base}")

    csv_files = sorted(base.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {base}")

    frames = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Skipped unreadable CSV: {csv_path}\n       {e}")
            continue

        if df.empty:
            print(f"[WARN] Empty CSV skipped: {csv_path}")
            continue

        dataset_p, model_p, attack_p = find_best_subpath_parts(csv_path, anchor=search_subdir)
        df.columns = [str(c).strip() for c in df.columns]

        dataset_csv = df["dataset"] if "dataset" in df.columns else pd.Series(np.nan, index=df.index)
        model_csv = df["model"] if "model" in df.columns else pd.Series(np.nan, index=df.index)
        attack_csv = df["attack"] if "attack" in df.columns else pd.Series(np.nan, index=df.index)

        df["dataset_csv"] = dataset_csv
        df["model_csv"] = model_csv
        df["attack_csv"] = normalize_attack_series(attack_csv)
        df["dataset_path"] = dataset_p
        df["model_path"] = model_p
        df["attack_path"] = normalize_attack_name(attack_p) if attack_p is not None else pd.NA

        df["dataset"] = coalesce_label(dataset_csv, dataset_p, PREFER_PATH_FOR_DATASET)
        df["model"] = coalesce_label(model_csv, model_p, PREFER_PATH_FOR_MODEL)
        df["attack"] = coalesce_label(attack_csv, attack_p, PREFER_PATH_FOR_ATTACK)
        df["attack"] = normalize_attack_series(df["attack"])

        for c in ["dataset", "model", "attack", "dataset_csv", "model_csv", "attack_csv", "dataset_path", "model_path", "attack_path"]:
            df[c] = df[c].astype("string")

        df["source_csv"] = str(csv_path)

        frames.append(df)

    if not frames:
        raise RuntimeError("No valid CSV files were loaded.")

    all_df = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "severity", "mAP50", "mAP50_95", "mAC", "VC50", "VC50_95",
        "avg_preds", "EPI50", "EPI50_95"
    ]
    all_df = safe_numeric(all_df, numeric_cols)

    if "split" not in all_df.columns:
        all_df["split"] = np.nan

    # Quick diagnostics for label drift across sources.
    model_mismatch = all_df[
        all_df["model_csv"].notna()
        & all_df["model_path"].notna()
        & (all_df["model_csv"].str.lower() != all_df["model_path"].str.lower())
    ]
    if not model_mismatch.empty:
        n_src = model_mismatch["source_csv"].nunique()
        print(f"[INFO] model_csv != model_path in {n_src} source CSVs (grouping uses selected strategy).")

    return all_df


def build_combo_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    attack_groups = all_df[all_df["attack"] != "baseline"].groupby(["dataset", "model", "attack"], dropna=False)

    for (dataset, model, attack), ga in attack_groups:
        gb = all_df[
            (all_df["dataset"] == dataset) &
            (all_df["model"] == model) &
            (all_df["attack"] == "baseline")
        ].copy()

        if gb.empty:
            print(f"[WARN] Missing baseline for dataset={dataset}, model={model}, attack={attack}")
            continue

        base_row = gb.iloc[0]
        end_row = pick_endpoint_severity(ga, attack)

        rec = {
            "dataset": dataset,
            "model": model,
            "attack": attack,
            "baseline_severity": base_row.get("severity", np.nan),
            "endpoint_severity": end_row.get("severity", np.nan),
        }

        metrics = ["mAP50", "mAP50_95", "mAC", "VC50", "VC50_95", "avg_preds", "EPI50", "EPI50_95"]
        for m in metrics:
            b = pd.to_numeric(base_row.get(m, np.nan), errors="coerce")
            e = pd.to_numeric(end_row.get(m, np.nan), errors="coerce")
            rec[f"{m}_baseline"] = b
            rec[f"{m}_endpoint"] = e
            rec[f"{m}_delta"] = e - b if pd.notna(b) and pd.notna(e) else np.nan
            rec[f"{m}_ratio"] = norm_by_baseline(e, b)

        if attack == "shrink":
            rec["pattern_note"] = "mAP50 typically degrades slower than mAC/VC under shrink"
        elif attack == "expand":
            rec["pattern_note"] = "mAC typically stays high while mAP50:95/VC decrease"
        elif attack == "duplicate":
            rec["pattern_note"] = "mAC gain saturates while mAP/VC and efficiency worsen"
        elif attack == "threshold":
            rec["pattern_note"] = "generally moderate sensitivity; usually weaker than structural attacks"
        else:
            rec["pattern_note"] = ""

        rows.append(rec)

    return pd.DataFrame(rows).sort_values(["dataset", "model", "attack"]).reset_index(drop=True)


def build_attack_consistency(combo_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attack, g in combo_df.groupby("attack", dropna=False):
        rec = {"attack": attack, "n_combinations": len(g)}

        summarize_cols = [
            "mAP50_ratio", "mAP50_95_ratio", "mAC_ratio", "VC50_95_ratio",
            "avg_preds_ratio", "EPI50_ratio", "EPI50_95_ratio"
        ]
        for c in summarize_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            rec[f"{c}_mean"] = vals.mean()
            rec[f"{c}_std"] = vals.std(ddof=1)
            rec[f"{c}_min"] = vals.min()
            rec[f"{c}_max"] = vals.max()

        def count_condition(series, fn):
            s = pd.to_numeric(series, errors="coerce").dropna()
            return int(sum(fn(v) for v in s)), int(len(s))

        if attack == "shrink":
            n1, d1 = count_condition(g["mAP50_ratio"], lambda v: v > 0.9)
            n2, d2 = count_condition(g["mAC_ratio"], lambda v: v < 0.9)
            rec["consistency_text"] = (
                f"{n1}/{d1} combos kept mAP50_ratio > 0.9 while "
                f"{n2}/{d2} combos had mAC_ratio < 0.9 at the strongest shrink level."
            )
        elif attack == "expand":
            n1, d1 = count_condition(g["mAC_ratio"], lambda v: v > 0.95)
            n2, d2 = count_condition(g["mAP50_95_ratio"], lambda v: v < 0.8)
            rec["consistency_text"] = (
                f"{n1}/{d1} combos kept mAC_ratio > 0.95 while "
                f"{n2}/{d2} combos had mAP50:95_ratio < 0.8 at the strongest expand level."
            )
        elif attack == "duplicate":
            n1, d1 = count_condition(g["mAC_ratio"], lambda v: v <= 1.05)
            n2, d2 = count_condition(g["mAP50_95_ratio"], lambda v: v < 1.0)
            rec["consistency_text"] = (
                f"{n1}/{d1} combos showed limited mAC gain (<= 1.05x) while "
                f"{n2}/{d2} combos showed mAP50:95 degradation under duplicate flooding."
            )
        elif attack == "threshold":
            n1, d1 = count_condition(g["VC50_95_ratio"], lambda v: 0.95 <= v <= 1.05)
            rec["consistency_text"] = (
                f"{n1}/{d1} combos stayed within ±5% of baseline VC50:95 "
                f"at the strongest threshold setting."
            )
        else:
            rec["consistency_text"] = ""

        rows.append(rec)

    return pd.DataFrame(rows).sort_values("attack").reset_index(drop=True)


def build_severity_mean_curves(all_df: pd.DataFrame) -> pd.DataFrame:
    df = all_df.copy()

    baseline_map = (
        df[df["attack"] == "baseline"]
        .groupby(["dataset", "model"], dropna=False)
        .agg({
            "mAP50": "first",
            "mAP50_95": "first",
            "mAC": "first",
            "VC50": "first",
            "VC50_95": "first",
            "avg_preds": "first",
            "EPI50": "first",
            "EPI50_95": "first",
        })
        .reset_index()
        .rename(columns={
            "mAP50": "mAP50_base",
            "mAP50_95": "mAP50_95_base",
            "mAC": "mAC_base",
            "VC50": "VC50_base",
            "VC50_95": "VC50_95_base",
            "avg_preds": "avg_preds_base",
            "EPI50": "EPI50_base",
            "EPI50_95": "EPI50_95_base",
        })
    )

    df = df[df["attack"] != "baseline"].copy()
    df = df.merge(baseline_map, on=["dataset", "model"], how="left")

    metrics = ["mAP50", "mAP50_95", "mAC", "VC50", "VC50_95", "avg_preds", "EPI50", "EPI50_95"]
    for m in metrics:
        df[f"{m}_norm"] = [norm_by_baseline(v, b) for v, b in zip(df[m], df[f"{m}_base"])]

    df = severity_rank_for_plot(df)

    agg = (
        df.groupby(["attack", "severity", "attack_progress"], dropna=False)
        .agg(
            n=("dataset", "count"),
            mAP50_norm_mean=("mAP50_norm", "mean"),
            mAP50_95_norm_mean=("mAP50_95_norm", "mean"),
            mAC_norm_mean=("mAC_norm", "mean"),
            VC50_95_norm_mean=("VC50_95_norm", "mean"),
            avg_preds_norm_mean=("avg_preds_norm", "mean"),
            EPI50_95_norm_mean=("EPI50_95_norm", "mean"),
        )
        .reset_index()
        .sort_values(["attack", "attack_progress", "severity"])
        .reset_index(drop=True)
    )
    return agg


def build_paper_ready_notes(combo_df: pd.DataFrame, attack_df: pd.DataFrame) -> str:
    lines = []
    lines.append("ANTI-GAMING SUMMARY NOTES")
    lines.append("=" * 80)
    lines.append("These notes are generated from the aggregated CSV summaries.")
    lines.append("Use them as drafting material, then verify against final plots/tables.\n")

    for _, row in attack_df.iterrows():
        attack = row["attack"]
        lines.append(f"[{attack.upper()}]")
        lines.append(f"- Number of dataset-model combinations: {int(row['n_combinations'])}")
        for col, label in [
            ("mAP50_ratio_mean", "Mean strongest-level mAP50 ratio"),
            ("mAP50_95_ratio_mean", "Mean strongest-level mAP50:95 ratio"),
            ("mAC_ratio_mean", "Mean strongest-level mAC ratio"),
            ("VC50_95_ratio_mean", "Mean strongest-level VC50:95 ratio"),
        ]:
            if col in row and pd.notna(row[col]):
                lines.append(f"- {label}: {row[col]:.4f}")
        lines.append(f"- Consistency note: {row['consistency_text']}")
        lines.append("")

    lines.append("SUGGESTED WRITING ANGLES")
    lines.append("- Shrink: emphasize that mAP@50 is often more tolerant than mAC/VC to partial coverage.")
    lines.append("- Expand: emphasize that mAC may remain high while localization-sensitive metrics drop.")
    lines.append("- Duplicate: emphasize that mAC gains saturate quickly, whereas precision/efficiency degrade.")
    lines.append("- Threshold: present as sensitivity analysis unless it causes strong inflation on harder datasets.")
    lines.append("")
    lines.append("IMPORTANT")
    lines.append("- Use 'qualitatively consistent across dataset-model combinations' rather than claiming identical magnitudes.")
    lines.append("- Double-check any combinations where mAP values are zero or implausibly low; there may be upstream evaluation issues.")
    return "\n".join(lines)


def save_attack_plot(curve_df: pd.DataFrame, attack: str, out_path: Path):
    g = curve_df[curve_df["attack"] == attack].sort_values("attack_progress")
    if g.empty:
        return

    plt.figure(figsize=(7.0, 4.8))

    plt.plot(g["attack_progress"], g["mAP50_norm_mean"],
             marker="o", color="tab:blue", label="mAP50 / baseline")
    plt.plot(g["attack_progress"], g["mAP50_95_norm_mean"],
             marker="o", color="tab:orange", label="mAP50:95 / baseline")
    plt.plot(g["attack_progress"], g["mAC_norm_mean"],
             marker="o", color="tab:green", label="mAC / baseline")
    plt.plot(g["attack_progress"], g["VC50_95_norm_mean"],
             marker="o", color="tab:red", label="VC50:95 / baseline")

    plt.axhline(y=1.0, linestyle="--", linewidth=1.3, color="tab:gray", label="baseline = 1")

    plt.xlabel("Attack progress (0 = weakest, 1 = strongest)")
    plt.ylabel("Normalized metric")
    plt.title(f"{attack.capitalize()} attack: normalized metric trends")

    # 不要卡死到 0.6，不然 expand / shrink 后段会被截掉
    ymin = min(
        g["mAP50_norm_mean"].min(),
        g["mAP50_95_norm_mean"].min(),
        g["mAC_norm_mean"].min(),
        g["VC50_95_norm_mean"].min()
    )
    ymax = max(
        g["mAP50_norm_mean"].max(),
        g["mAP50_95_norm_mean"].max(),
        g["mAC_norm_mean"].max(),
        g["VC50_95_norm_mean"].max()
    )
    plt.ylim(max(0.0, ymin - 0.05), ymax + 0.05)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_efficiency_plot(curve_df: pd.DataFrame, attack: str, out_path: Path):
    g = curve_df[curve_df["attack"] == attack].sort_values("attack_progress")
    if g.empty:
        return

    c_avg = "tab:blue"
    c_epi = "tab:orange"
    c_base = "tab:gray"

    fig, ax1 = plt.subplots(figsize=(7.0, 4.8))

    l1 = ax1.plot(
        g["attack_progress"],
        g["avg_preds_norm_mean"],
        marker="o",
        color=c_avg,
        label="avg_preds / baseline"
    )
    ax1.set_xlabel("Attack progress (0 = weakest, 1 = strongest)")
    ax1.set_ylabel("avg_preds / baseline", color=c_avg)
    ax1.tick_params(axis="y", labelcolor=c_avg)
    ax1.axhline(y=1.0, linestyle="--", linewidth=1.2, color=c_base, alpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l2 = ax2.plot(
        g["attack_progress"],
        g["EPI50_95_norm_mean"],
        marker="o",
        color=c_epi,
        label="EPI / baseline"
    )
    ax2.set_ylabel("EPI / baseline", color=c_epi)
    ax2.tick_params(axis="y", labelcolor=c_epi)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.1, 0.95))
    plt.tight_layout()

    plt.title(f"{attack.capitalize()} attack: efficiency trends")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    output_dir = ROOT_DIR / OUTPUT_DIRNAME
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading CSVs from: {ROOT_DIR / SEARCH_SUBDIR}")
    all_df = load_all_csvs(ROOT_DIR, SEARCH_SUBDIR)
    print(f"[INFO] Loaded rows: {len(all_df)}")

    combined_csv = output_dir / "combined_all_rows.csv"
    all_df.to_csv(combined_csv, index=False)

    combo_df = build_combo_summary(all_df)
    combo_csv = output_dir / "combo_baseline_and_endpoints.csv"
    combo_df.to_csv(combo_csv, index=False)

    attack_df = build_attack_consistency(combo_df)
    attack_csv = output_dir / "attack_consistency_summary.csv"
    attack_df.to_csv(attack_csv, index=False)

    curve_df = build_severity_mean_curves(all_df)
    curve_csv = output_dir / "severity_mean_curves.csv"
    curve_df.to_csv(curve_csv, index=False)

    notes = build_paper_ready_notes(combo_df, attack_df)
    notes_path = output_dir / "paper_ready_notes.txt"
    notes_path.write_text(notes, encoding="utf-8")

    xlsx_path = output_dir / "anti_gaming_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="all_rows", index=False)
        combo_df.to_excel(writer, sheet_name="combo_endpoints", index=False)
        attack_df.to_excel(writer, sheet_name="attack_summary", index=False)
        curve_df.to_excel(writer, sheet_name="mean_curves", index=False)

    if MAKE_PLOTS:
        for attack in ["shrink", "expand", "duplicate", "threshold"]:
            save_attack_plot(curve_df, attack, plots_dir / f"{attack}_mean_normalized_metrics.png")
            save_efficiency_plot(curve_df, attack, plots_dir / f"{attack}_mean_efficiency.png")

    suspicious = all_df[
        (pd.to_numeric(all_df.get("mAP50", np.nan), errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(all_df.get("mAC", np.nan), errors="coerce").fillna(0) > 0)
    ]
    if not suspicious.empty:
        suspicious_csv = output_dir / "suspicious_zero_map_rows.csv"
        suspicious.to_csv(suspicious_csv, index=False)
        print(f"[WARN] Found rows with mAP50 == 0 but mAC > 0. Saved to: {suspicious_csv}")

    print("[DONE] Summary files created:")
    print(f"  - {combined_csv}")
    print(f"  - {combo_csv}")
    print(f"  - {attack_csv}")
    print(f"  - {curve_csv}")
    print(f"  - {notes_path}")
    print(f"  - {xlsx_path}")
    if MAKE_PLOTS:
        print(f"  - plots folder: {plots_dir}")
    print("[INFO] Finished successfully.")


if __name__ == "__main__":
    main()
