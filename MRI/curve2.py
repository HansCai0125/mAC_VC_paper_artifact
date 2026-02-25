#vc 对比 相关性
#多模型多数据集的 mAP vs mAC 散点图
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT

# ================== 配置区域 ==================

# 不同数据集的根目录（里面要有 yolo11n_s0, yolo11s_s0 这种结构）
DATASET_ROOTS = {
#    "CaWeed10": Path(r"E:\Project\Database\CaWeed10"),   # TODO: 换成你 CaWeed10 实验真正有 vc_epoch.csv 的目录
    "BTD":      RESULT_ROOT / "MRI",
}

# 要对比的模型
MODELS = ["yolo11n", "yolo11s"]

# 使用的 seeds
SEEDS = [0, 1, 2, 3, 4]

# 输出图片目录
OUT_DIR = (RESULT_ROOT / "MRI" / "plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== 工具函数 ==================

def find_col_fuzzy(header, keywords):
    """
    在 header 里找列：
    1) 先尝试 exact match
    2) 再尝试 keyword 作为子串出现
    keywords 可以是 str 或 [str, str, ...]
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

    raise KeyError(f"找不到包含关键词 {keywords} 的列名, 在 header={header} 里")


def read_vc_epoch_csv(csv_path: Path):
    """
    读取 metrics/vc_epoch.csv
    返回 list[dict]，每一行包括:
      - epoch (如果文件没有这一列，用行号替代)
      - mAP50, mAC, VC_mAP50, VC_mAC
    """
    if not csv_path.exists():
        print(f"[WARN] 找不到 vc_epoch.csv: {csv_path}")
        return []

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[WARN] 空文件: {csv_path}")
            return []

        header = [h.strip() for h in header]

        # 尝试找 epoch 列（如果有的话）
        try:
            idx_epoch = find_col_fuzzy(header, "epoch")
        except KeyError:
            idx_epoch = None  # 没有 epoch 列就用行号

        # mAP50 / mAC / VC_mAP50 / VC_mAC
        idx_mAC      = find_col_fuzzy(header, "mAC")
        idx_VC_mAC   = find_col_fuzzy(header, "VC_mAC")
        idx_mAP50    = find_col_fuzzy(header, "mAP50")
        idx_VC_mAP50 = find_col_fuzzy(header, "VC_mAP50")

        for row_idx, row in enumerate(reader):
            if not row or all(c.strip() == "" for c in row):
                continue

            # epoch：优先用文件里的 epoch 列，否则用行号
            if idx_epoch is not None:
                try:
                    epoch = int(float(row[idx_epoch]))
                except ValueError:
                    epoch = row_idx  # 退一步
            else:
                epoch = row_idx  # 从 0 开始

            try:
                mAC      = float(row[idx_mAC])
                VC_mAC   = float(row[idx_VC_mAC])
                mAP50    = float(row[idx_mAP50])
                VC_mAP50 = float(row[idx_VC_mAP50])
            except ValueError:
                continue

            rows.append({
                "epoch": epoch,
                "mAC": mAC,
                "VC_mAC": VC_mAC,
                "mAP50": mAP50,
                "VC_mAP50": VC_mAP50,
            })

    return rows


# ================== 汇总所有点 ==================

def collect_points():
    """
    遍历 DATASET_ROOTS / model_s{seed} / metrics/vc_epoch.csv
    收集所有 (dataset, model, seed, epoch, mAP50, mAC, VC_mAP50, VC_mAC)
    """
    all_points = []

    for dataset_name, root in DATASET_ROOTS.items():
        print(f"\n[DATASET] {dataset_name}  root={root}")

        for model in MODELS:
            for seed in SEEDS:
                run_dir = root / f"{model}_s{seed}"
                csv_path = run_dir / "metrics" / "vc_epoch.csv"

                if not csv_path.exists():
                    print(f"  [WARN] {dataset_name} / {model}_s{seed} 没有 vc_epoch.csv，跳过")
                    continue

                rows = read_vc_epoch_csv(csv_path)
                print(f"  [INFO] {dataset_name} / {model}_s{seed}: 读取 {len(rows)} 行")

                for r in rows:
                    all_points.append({
                        "dataset": dataset_name,
                        "model": model,
                        "seed": seed,
                        "epoch": r["epoch"],
                        "mAP50": r["mAP50"],
                        "mAC": r["mAC"],
                        "VC_mAP50": r["VC_mAP50"],
                        "VC_mAC": r["VC_mAC"],
                    })

    print(f"\n[SUMMARY] 一共收集到 {len(all_points)} 个点")
    return all_points


# ================== 画 mAP vs mAC 散点 & 计算相关系数 ==================

def plot_map_vs_mac(points):
    """
    Figure 3: mAP@0.5 vs mAC@0.5
    - 颜色: 数据集 (CaWeed10 vs BTD)
    - marker: 模型 (YOLO11n vs YOLO11s)
    """
    if not points:
        print("[WARN] 没有点可画，直接返回")
        return

    # 颜色和 marker 映射
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

    # (dataset, model) 分组
    grouped = {}
    for p in points:
        key = (p["dataset"], p["model"])
        grouped.setdefault(key, []).append((p["mAP50"], p["mAC"]))

    # 每个数据集的相关系数
    corr_per_dataset = {}
    for dataset_name in DATASET_ROOTS.keys():
        xs = [p["mAP50"] for p in points if p["dataset"] == dataset_name]
        ys = [p["mAC"]   for p in points if p["dataset"] == dataset_name]
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
            print(f"  {ds}: r = NaN (点太少)")

    if "CaWeed10" in corr_per_dataset and "BTD" in corr_per_dataset:
        r1 = corr_per_dataset["CaWeed10"]
        r2 = corr_per_dataset["BTD"]
        if np.isfinite(r1) and np.isfinite(r2):
            delta_corr = r2 - r1
            print(f"\n  ΔCorr (BTD - CaWeed10) = {delta_corr:+.4f}")

    # ========= 画图 =========
    plt.figure(figsize=(6, 6))

    color_pool_idx = 0
    marker_pool_idx = 0

    for (dataset_name, model_name), vals in grouped.items():
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]

        # 颜色
        if dataset_name in dataset_colors:
            color = dataset_colors[dataset_name]
        else:
            color = default_colors[color_pool_idx % len(default_colors)]
            color_pool_idx += 1

        # marker
        if model_name in model_markers:
            marker = model_markers[model_name]
        else:
            marker = default_markers[marker_pool_idx % len(default_markers)]
            marker_pool_idx += 1

        label = f"{dataset_name} – {model_name}"
        plt.scatter(xs, ys, c=color, marker=marker, alpha=0.6,
                    label=label, edgecolors="none")

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
