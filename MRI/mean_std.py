import csv
import statistics
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESULT_ROOT

# ============= 配置区域 =============
ROOT = RESULT_ROOT / "MRI"

# 当前 MRI 目录里的模型 & seeds
MODELS = ["yolo11n", "yolo11s"]
SEEDS  = [0, 1, 2, 3, 4]

# 用哪个指标来选“最佳 epoch”（results.csv 里的列名里会包含这个关键字）
BEST_BY_KEYWORD = "mAP50(B)"   # 对应 metrics/mAP50(B)


# ============= 工具函数 =============

def read_csv_rows(path: Path):
    """读取一个 CSV，返回 List[Dict]"""
    if not path.exists():
        print(f"[WARN] CSV 不存在: {path}")
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def find_col_with_keyword(cols, keyword):
    """在列名里找包含 keyword 的那一列（不区分大小写）"""
    keyword = keyword.lower()
    for c in cols:
        if keyword in c.lower():
            return c
    raise KeyError(f"找不到包含关键字 '{keyword}' 的列名，在 {list(cols)} 里")


# ============= 核心函数：取单个 run 的最佳 epoch 指标 =============

def get_best_epoch_metrics(run_dir: Path):
    """
    从一个 run 目录中：
    - 用 results.csv 选出 BEST_BY 对应的最大值所在的行，得到 best_epoch
    - 再从 mac_epoch.csv / vc_epoch.csv / epi_results.csv 提取同一 epoch 的指标
      （VC 部分用 mAP50 & mAC 最接近的那行来对齐）
    返回 dict: {epoch, mAP50, mAP95, mAC, VC_mAP, VC_mAC, EPI}
    """
    results_path = run_dir / "results.csv"
    mac_path     = run_dir / "metrics" / "mac_epoch.csv"
    vc_path      = run_dir / "metrics" / "vc_epoch.csv"
    epi_path     = run_dir / "metrics" / "epi_results.csv"

    # ---------- 1) 从 results.csv 中找 best epoch ----------
    results_rows = read_csv_rows(results_path)
    if not results_rows:
        print(f"[WARN] {run_dir} 没有 results.csv 或内容为空，跳过。")
        return None

    cols_res = results_rows[0].keys()

    # epoch 列名基本就是 "epoch"
    epoch_col_res = find_col_with_keyword(cols_res, "epoch")

    # mAP50 列
    try:
        map50_col_res = find_col_with_keyword(cols_res, BEST_BY_KEYWORD)
    except KeyError:
        # 某些版本列名可能简化成 metrics/mAP50
        map50_col_res = find_col_with_keyword(cols_res, "mAP50")

    # mAP50-95 列
    try:
        map5095_col_res = find_col_with_keyword(cols_res, "mAP50-95")
    except KeyError:
        map5095_col_res = None

    # 选出 mAP50 最大的一行
    best_row = max(results_rows, key=lambda r: safe_float(r.get(map50_col_res, 0.0)))
    best_epoch = int(safe_float(best_row.get(epoch_col_res, 0.0)))
    best_mAP50 = safe_float(best_row.get(map50_col_res, 0.0))
    best_mAP95 = safe_float(best_row.get(map5095_col_res, 0.0)) if map5095_col_res else 0.0

    # ---------- 2) 从 mac_epoch.csv 里取这个 epoch 的 mAC ----------
    mac_rows = read_csv_rows(mac_path)
    if mac_rows:
        # 列名：epoch, mAC, avg_preds
        epoch_col_mac = find_col_with_keyword(mac_rows[0].keys(), "epoch")
        mac_col_mac   = "mAC"  # 你文件里是这个名字

        mac_row = None
        for r in mac_rows:
            if int(safe_float(r.get(epoch_col_mac, -1))) == best_epoch:
                mac_row = r
                break
        if mac_row is None:
            # 找不到同 epoch，就用 mAC 最大的行兜底
            mac_row = max(mac_rows, key=lambda r: safe_float(r.get(mac_col_mac, 0.0)))

        best_mAC = safe_float(mac_row.get(mac_col_mac, 0.0))
    else:
        best_mAC = 0.0

    # ---------- 3) 从 vc_epoch.csv 里，用“最接近 mAP50& mAC”的行来对齐 ----------
    vc_rows = read_csv_rows(vc_path)
    if vc_rows:
        # 列名已经固定：mAC, VC_mAC, mAP50, VC_mAP50, mAP50-95, VC_mAP95, ...
        mAC_col_vc      = "mAC"
        VC_mAC_col_vc   = "VC_mAC"
        mAP50_col_vc    = "mAP50"
        VC_mAP50_col_vc = "VC_mAP50"
        mAP95_col_vc    = "mAP50-95"   # 这是 mAP@0.5:0.95
        VC_mAP95_col_vc = "VC_mAP95"

        best_vc_row = None
        best_dist   = float("inf")

        for r in vc_rows:
            r_mAP50 = safe_float(r.get(mAP50_col_vc, 0.0))
            r_mAP95 = safe_float(r.get(mAP95_col_vc, 0.0))
            r_mAC   = safe_float(r.get(mAC_col_vc, 0.0))

            # 和 best_epoch 对应的 (mAP50, mAP95, mAC) 的距离
            dist = abs(r_mAP50 - best_mAP50) + abs(r_mAP95 - best_mAP95) + abs(r_mAC - best_mAC)
            if dist < best_dist:
                best_dist = dist
                best_vc_row = r

        VC_mAP = safe_float(best_vc_row.get(VC_mAP50_col_vc, 0.0))
        VC_mAC = safe_float(best_vc_row.get(VC_mAC_col_vc, 0.0))
    else:
        VC_mAP = VC_mAC = 0.0

    # ---------- 4) 从 epi_results.csv 里取同 epoch 的 EPI ----------
    epi_rows = read_csv_rows(epi_path)
    if epi_rows:
        epoch_col_epi = find_col_with_keyword(epi_rows[0].keys(), "epoch")
        epi_col       = "EPI"  # 你的文件里就是这个名字

        epi_row = None
        for r in epi_rows:
            if int(safe_float(r.get(epoch_col_epi, -1))) == best_epoch:
                epi_row = r
                break
        if epi_row is None:
            epi_row = max(epi_rows, key=lambda r: safe_float(r.get(epi_col, 0.0)))
        EPI = safe_float(epi_row.get(epi_col, 0.0))
    else:
        EPI = 0.0

    return {
        "epoch":  best_epoch,
        "mAP50":  best_mAP50,
        "mAP95":  best_mAP95,
        "mAC":    best_mAC,
        "VC_mAP": VC_mAP,
        "VC_mAC": VC_mAC,
        "EPI":    EPI,
    }


# ============= 主流程：对每个模型做 mean/std =============

if __name__ == "__main__":
    metric_keys = ["mAP50", "mAP95", "mAC", "VC_mAP", "VC_mAC", "EPI"]

    for model in MODELS:
        print(f"\n====== Model: {model} (Brain Tumor / MRI) ======")
        runs_metrics = []

        for s in SEEDS:
            run_dir = ROOT / f"{model}_s{s}"
            if not run_dir.exists():
                print(f"[WARN] 目录不存在: {run_dir}，跳过该 seed。")
                continue

            m = get_best_epoch_metrics(run_dir)
            if m is None:
                continue

            runs_metrics.append(m)
            print(f"  Seed {s}: epoch={m['epoch']}, "
                  f"mAP50={m['mAP50']:.4f}, mAP95={m['mAP95']:.4f}, "
                  f"mAC={m['mAC']:.4f}, VC_mAP={m['VC_mAP']:.4f}, "
                  f"VC_mAC={m['VC_mAC']:.4f}, EPI={m['EPI']:.4f}")

        if not runs_metrics:
            continue

        print("\n  ---- Mean ± Std over seeds ----")
        summary = {}
        for k in metric_keys:
            arr = [m[k] for m in runs_metrics]
            mean = statistics.mean(arr)
            std  = statistics.stdev(arr) if len(arr) > 1 else 0.0
            summary[k] = (mean, std)
            print(f"  {k:7s}: {mean:.4f} ± {std:.4f}")

        # 顺便给一行 LaTeX table 形式（可以直接复制到论文表格里）
        latex_row = (
            f"{model} & Brain Tumor Detection & "
            f"{summary['mAP50'][0]:.3f} $\\pm$ {summary['mAP50'][1]:.3f} & "
            f"{summary['mAP95'][0]:.3f} $\\pm$ {summary['mAP95'][1]:.3f} & "
            f"{summary['mAC'][0]:.3f} $\\pm$ {summary['mAC'][1]:.3f} & "
            f"{summary['VC_mAP'][0]:.3f} $\\pm$ {summary['VC_mAP'][1]:.3f} & "
            f"{summary['VC_mAC'][0]:.3f} $\\pm$ {summary['VC_mAC'][1]:.3f} & "
            f"{summary['EPI'][0]:.3f} $\\pm$ {summary['EPI'][1]:.3f} \\\\"
        )
        print("\n  LaTeX row:")
        print("  " + latex_row)
