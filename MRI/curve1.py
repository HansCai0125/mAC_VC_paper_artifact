#map mac epi
"""
多 seed 曲线汇总脚本
- 读取 yolo11n_s0~s4、yolo11s_s0~s4 等目录里的 CSV
- 为每个模型画：
    * mAP@0.5 / mAC@0.5 / EPI vs epoch 的 mean±std 曲线
- 再把 yolo11n 和 yolo11s 的 mean 曲线画在同一张图上做对比
"""

import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT

# ================== 配置区域 ==================

# 项目根目录（包含 yolo11n_s0、yolo11s_s0 等文件夹）
ROOT_DIR = RESULT_ROOT / "MRI"

# 要处理的模型名字（对应文件夹前缀）
MODELS = ["yolo11n", "yolo11s"]   # 想只画一个就留 ["yolo11n"] 即可

# seeds
SEEDS = [0, 1, 2, 3, 4]

# 输出图片目录
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
#   辅助函数：读取 CSV → {epoch -> value}
# =========================================================

def _find_col(cols, keyword):
    """在表头里找到精确等于 keyword 的列索引，不存在就报错。"""
    for i, c in enumerate(cols):
        if c.strip() == keyword:
            return i
    raise KeyError(f"找不到包含关键字 '{keyword}' 的列名，在 {cols} 里")


def read_metric_csv(path: Path, key_col: str, val_col: str):
    """
    从 path 读取两列：key_col (epoch) 和 val_col (metric)
    返回 dict: {epoch(int) -> value(float)}
    """
    if not path.exists():
        print(f"[WARN] 文件不存在: {path}")
        return {}

    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[WARN] 空文件: {path}")
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
#   收集各个 seed 的 metric → dict(seed -> {epoch->value})
# =========================================================

def collect_metric_dicts_for_model(model_name: str):
    """
    对给定 model_name（例如 'yolo11n'），收集：
      - mAP50:  from results.csv
      - mAC:    from metrics/mac_epoch.csv
      - EPI:    from metrics/epi_results.csv
    返回 (map50_dicts, mac_dicts, epi_dicts)
      其中 map50_dicts[seed] = {epoch -> value}
    """
    map50_dicts = {}
    mac_dicts = {}
    epi_dicts = {}

    print(f"\n====== Model: {model_name} ======")

    for seed in SEEDS:
        run_dir = ROOT_DIR / f"{model_name}_s{seed}"
        metrics_dir = run_dir / "metrics"

        if not run_dir.exists():
            print(f"[WARN] 目录不存在，跳过该 seed: {run_dir}")
            continue

        print(f"[INFO] 读取 seed {seed} 结果，目录: {run_dir}")

        # 1) mAP@0.5: results.csv 里的 'metrics/mAP50(B)'
        res_path = run_dir / "results.csv"
        m_map = read_metric_csv(res_path, key_col="epoch",
                                val_col="metrics/mAP50(B)")
        if not m_map:
            print(f"[WARN] 该 seed 没有 mAP 数据: {res_path}")
        else:
            map50_dicts[seed] = m_map

        # 2) mAC@0.5: metrics/mac_epoch.csv 里的 'mAC'
        mac_path = metrics_dir / "mac_epoch.csv"
        m_mac = read_metric_csv(mac_path, key_col="epoch", val_col="mAC")
        if not m_mac:
            print(f"[WARN] 该 seed 没有 mAC 数据: {mac_path}")
        else:
            mac_dicts[seed] = m_mac

        # 3) EPI: metrics/epi_results.csv 里的 'EPI'
        epi_path = metrics_dir / "epi_results.csv"
        m_epi = read_metric_csv(epi_path, key_col="epoch", val_col="EPI")
        if not m_epi:
            print(f"[WARN] 该 seed 没有 EPI 数据: {epi_path}")
        else:
            epi_dicts[seed] = m_epi

    return map50_dicts, mac_dicts, epi_dicts


# =========================================================
#   计算 mean / std
# =========================================================

def compute_mean_std(dicts_per_seed):
    """
    输入: dict(seed -> {epoch -> value})
    输出: (epochs_sorted, mean_arr, std_arr)
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
#   单模型：种子曲线 + mean ± std
# =========================================================

def plot_metric_single(epochs, mean_arr, std_arr, dicts_per_seed,
                       ylabel, title, out_path,
                       show_seeds=True,
                       drop_first_n=0,
                       clip_upper_percentile=None):
    """
    show_seeds:     是否画每个 seed 的细线
    drop_first_n:   丢掉前 N 个 epoch（例如 EPI 的 epoch 0 可能有极端值）
    clip_upper_percentile: 把 y 轴上限裁到某个百分位数，比如 99，避免单个 peak 压扁细节
    """
    if epochs.size == 0:
        print(f"[WARN] 没有数据，跳过绘图: {out_path}")
        return

    # 丢掉前若干 epoch
    if drop_first_n > 0:
        mask = epochs >= drop_first_n
        epochs = epochs[mask]
        mean_arr = mean_arr[mask]
        std_arr = std_arr[mask]

    plt.figure(figsize=(8, 4))

    # 颜色
    seed_colors = ["#A6CEE3", "#FDBF6F", "#B2DF8A", "#FB9A99", "#CAB2D6"]
    mean_color = "#1F78B4"
    shade_color = "#9EC9FF"

    # 画每个 seed 的细线
    if show_seeds:
        for i, (s, d) in enumerate(sorted(dicts_per_seed.items(), key=lambda x: x[0])):
            xs = sorted(e for e in d.keys() if e >= drop_first_n)
            if not xs:
                continue
            ys = [d[e] for e in xs]
            c = seed_colors[i % len(seed_colors)]
            plt.plot(xs, ys,
                     color=c,
                     alpha=0.5,
                     linewidth=1.0,
                     label=f"seed {s}")

    # 画 mean 曲线
    plt.plot(epochs, mean_arr,
             color=mean_color,
             linewidth=2.5,
             label="mean")

    # std 阴影
    upper = mean_arr + std_arr
    lower = mean_arr - std_arr
    plt.fill_between(epochs, lower, upper,
                     color=shade_color,
                     alpha=0.25,
                     label="±1 std")

    # 裁掉上面的峰值（比如 EPI 的异常值）
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
#   多模型：只画 mean 曲线做对比
# =========================================================

def plot_compare_mean(metric_name: str,
                      model_results: dict,
                      ylabel: str,
                      title: str,
                      out_path: Path,
                      drop_first_n: int = 0,
                      clip_upper_percentile=None):
    """
    model_results: {
        model_name: (epochs, mean_arr, std_arr)
    }
    只画每个模型的 mean，不画 seed / 阴影。
    """

    # 先找所有模型的 epoch 交集（保证对齐）
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
        print(f"[WARN] 没有有效 epoch，无法画对比图: {out_path}")
        return

    common_epochs = sorted(set.intersection(*epoch_sets))
    if not common_epochs:
        print(f"[WARN] 各模型 epoch 没有交集，无法画对比图: {out_path}")
        return

    common_epochs = np.array(common_epochs, dtype=int)

    # 为每个模型准备 y
    ys_all = {}
    for model_name, (epochs, mean_arr, std_arr) in model_results.items():
        if epochs.size == 0:
            continue
        # 构建映射 epoch -> mean
        mapping = {int(e): float(v) for e, v in zip(epochs, mean_arr)}
        ys = [mapping[e] for e in common_epochs]
        ys_all[model_name] = np.array(ys, dtype=float)

    plt.figure(figsize=(8, 4))

    # 不同模型的颜色
    model_colors = {
        "yolo11n": "#1F78B4",  # 蓝
        "yolo11s": "#33A02C",  # 绿
    }

    for model_name, ys in ys_all.items():
        color = model_colors.get(model_name, None)
        plt.plot(common_epochs, ys,
                 linewidth=2.0,
                 label=model_name,
                 alpha=0.95,
                 color=color)

    # 裁掉峰值（主要给 EPI 用）
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
#   主流程
# =========================================================

def main():
    # 保存每个模型的 mean/std 结果，方便后面做对比
    all_model_stats = {}

    for model_name in MODELS:
        map50_dicts, mac_dicts, epi_dicts = collect_metric_dicts_for_model(model_name)

        # ---------- mAP@0.5 ----------
        epochs_map, mean_map, std_map = compute_mean_std(map50_dicts)
        # 单模型图（可注释掉）
        plot_metric_single(
            epochs_map, mean_map, std_map, map50_dicts,
            ylabel="mAP@0.5",
            title=f"{model_name} – mAP@0.5 vs Epoch (mean ± std over seeds)",
            out_path=PLOTS_DIR / f"{model_name}_mAP50_curve.png",
            show_seeds=True,
            drop_first_n=0,
            clip_upper_percentile=None,
        )

        # ---------- mAC@0.5 ----------
        epochs_mac, mean_mac, std_mac = compute_mean_std(mac_dicts)
        plot_metric_single(
            epochs_mac, mean_mac, std_mac, mac_dicts,
            ylabel="mAC@0.5",
            title=f"{model_name} – mAC@0.5 vs Epoch (mean ± std over seeds)",
            out_path=PLOTS_DIR / f"{model_name}_mAC50_curve.png",
            show_seeds=True,
            drop_first_n=0,
            clip_upper_percentile=None,
        )

        # ---------- EPI ----------
        epochs_epi, mean_epi, std_epi = compute_mean_std(epi_dicts)
        plot_metric_single(
            epochs_epi, mean_epi, std_epi, epi_dicts,
            ylabel="EPI",
            title=f"{model_name} – EPI vs Epoch (mean ± std over seeds)",
            out_path=PLOTS_DIR / f"{model_name}_EPI_curve.png",
            show_seeds=False,          # EPI 不画种子，只画 mean±std
            drop_first_n=1,            # 丢掉 epoch 0
            clip_upper_percentile=99   # 裁掉上端 1% 极端值
        )

        # 把结果存起来，后面多模型对比要用
        all_model_stats[model_name] = {
            "map": (epochs_map, mean_map, std_map),
            "mac": (epochs_mac, mean_mac, std_mac),
            "epi": (epochs_epi, mean_epi, std_epi),
        }

    # ============ yolo11n vs yolo11s 的 mean 曲线对比 ============

    # 只挑出我们关心的模型
    compare_models = [m for m in ["yolo11n", "yolo11s"] if m in all_model_stats]

    if len(compare_models) >= 2:
        # mAP@0.5
        model_results_map = {
            m: all_model_stats[m]["map"] for m in compare_models
        }
        plot_compare_mean(
            metric_name="mAP@0.5",
            model_results=model_results_map,
            ylabel="mAP@0.5",
            title="YOLO11n vs YOLO11s – mAP@0.5 (mean over seeds)",
            out_path=PLOTS_DIR / "compare_mAP50_yolo11n_vs_yolo11s.png",
            drop_first_n=0,
            clip_upper_percentile=None,
        )

        # mAC@0.5
        model_results_mac = {
            m: all_model_stats[m]["mac"] for m in compare_models
        }
        plot_compare_mean(
            metric_name="mAC@0.5",
            model_results=model_results_mac,
            ylabel="mAC@0.5",
            title="YOLO11n vs YOLO11s – mAC@0.5 (mean over seeds)",
            out_path=PLOTS_DIR / "compare_mAC50_yolo11n_vs_yolo11s.png",
            drop_first_n=0,
            clip_upper_percentile=None,
        )

        # EPI
        model_results_epi = {
            m: all_model_stats[m]["epi"] for m in compare_models
        }
        plot_compare_mean(
            metric_name="EPI",
            model_results=model_results_epi,
            ylabel="EPI",
            title="YOLO11n vs YOLO11s – EPI (mean over seeds)",
            out_path=PLOTS_DIR / "compare_EPI_yolo11n_vs_yolo11s.png",
            drop_first_n=1,          # 同样丢掉 epoch 0
            clip_upper_percentile=99 # 裁掉极端峰值，方便看细节
        )

    print("\n[FINISHED] 所有模型的曲线以及 yolo11n vs yolo11s 对比图已生成。")


if __name__ == "__main__":
    main()


