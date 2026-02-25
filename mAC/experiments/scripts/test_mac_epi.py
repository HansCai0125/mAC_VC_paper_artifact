import argparse, os, json, math
import pandas as pd
from typing import Optional
import importlib.util
import subprocess
import sys
import glob

# --- 小工具：把浮点转成文件名友好的字符串 0.25 -> 0p25, 1.0 -> 1 ---
def f2s(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")  # 限 3 位小数，并去掉末尾 0
    return s.replace(".", "p") if "." in s else s

# --- 输入检查 ---
def check_inputs(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"❌ Missing input file: {p}")


def run_epi_batch_via_subprocess():
    # Root
    base = r"E:\Project\mAC"
    epi_script = os.path.join(base, "experiments", "scripts", "EPI.py")
    acc = os.path.join(base, "experiments", "test_file", "table_accuracy.csv")
    avg = os.path.join(base, "experiments", "test_file", "table_avgpreds.csv")
    lat = os.path.join(base, "experiments", "test_file", "table_latency.csv")
    out = os.path.join(base, "experiments", "test_file", "epi_results.csv")

    # 检查输入是否存在
    check_inputs(epi_script, acc, avg, lat)

    # 想测的 α/λ 组合
    param_pairs = [
        (0.5, 0.5),
        (0.25, 0.75),   # 更强调 mAC
        (0.75, 0.25),   # 更强调 mAP
    ]

    out_dir = os.path.dirname(out)
    os.makedirs(out_dir, exist_ok=True)

    for alpha, lam in param_pairs:
        out_file = os.path.join(out_dir, f"epi_alpha{f2s(alpha)}_lam{f2s(lam)}.csv")
        print(f"\n>>> Running EPI α={alpha}, λ={lam}")
        try:
            subprocess.run([
                sys.executable, epi_script,
                "--accuracy", acc,
                "--avgpreds", avg,
                "--latency",  lat,
                "--alpha", str(alpha),
                "--lam",   str(lam),
                "--out",   out_file
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed α={alpha}, λ={lam}: {e}")
            continue

    print("\n✅ All EPI runs finished.")
    summarize_all(out_dir)  # 汇总所有结果

def load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("user_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def tail_value_from_csv(path: str, colnames):
    df = pd.read_csv(path)
    for c in colnames:
        if c in df.columns and not df[c].empty:
            return float(df[c].dropna().iloc[-1])
    raise ValueError(f"{path} 未找到列 {colnames}")

def _float(x) -> Optional[float]:
    if x is None: return None
    try: return float(x)
    except: return None

def compute_epi_single(args):
    # 1) mAC
    mac = None
    if args.mac_value is not None:
        mac = _float(args.mac_value)
    elif args.mac_csv:
        mac = tail_value_from_csv(args.mac_csv, ["mAC", "mac"])
    elif args.mac_module and args.gt and args.pred:
        mod = load_module_from_path(args.mac_module)
        if hasattr(mod, "compute_mac"):
            mac, n = mod.compute_mac(args.gt, args.pred, args.tau, True, 0.25)  # 临时参数
        else:
            raise RuntimeError(f"{args.mac_module} 未找到 compute_mac(...) 函数，请改成你实际的 API")
    else:
        raise RuntimeError("缺少 mAC 输入（--mac_value 或 --mac_csv 或 --mac_module+--gt+--pred）")

    # 2) mAP
    mapv = None
    if args.map_value is not None:
        mapv = _float(args.map_value)
    elif args.map_csv:
        mapv = tail_value_from_csv(args.map_csv, ["mAP","map","map50","metrics/mAP50","metrics/mAP50-95"])
    else:
        raise RuntimeError("缺少 mAP 输入（--map_value 或 --map_csv）")

    # 3) Avg#Preds
    avg_preds = None
    if args.avgpreds_value is not None:
        avg_preds = _float(args.avgpreds_value)
    elif args.avgpreds_csv:
        avg_preds = tail_value_from_csv(args.avgpreds_csv, ["avg_preds","avg#preds","avg_num_preds"])
    else:
        raise RuntimeError("缺少 Avg#Preds 输入（--avgpreds_value 或 --avgpreds_csv）")

    # 4) Latency
    latency_s = None
    if args.latency_s is not None:
        latency_s = _float(args.latency_s)
    elif args.latency_ms is not None:
        latency_s = _float(args.latency_ms) / 1000.0
    elif args.latency_csv:
        df = pd.read_csv(args.latency_csv)
        if "latency_s" in df.columns:
            latency_s = float(df["latency_s"].dropna().iloc[-1])
        elif "latency_ms" in df.columns:
            latency_s = float(df["latency_ms"].dropna().iloc[-1]) / 1000.0
        elif "latency" in df.columns:
            latency_s = float(df["latency"].dropna().iloc[-1]) / 1000.0
        else:
            raise RuntimeError(f"{args.latency_csv} 未找到 latency 列")
    else:
        raise RuntimeError("缺少 Latency 输入（--latency_s 或 --latency_ms 或 --latency_csv）")

    # 5) 计算 EPI
    alpha = float(args.alpha)
    lam = float(args.lam)
    combined_acc = alpha * float(mapv) + lam * float(mac)
    denom = max(1e-12, float(avg_preds) * float(latency_s))
    epi = combined_acc / denom

    row = {
        "model": args.model,
        "alpha": alpha, "lambda": lam,
        "mAP": float(mapv), "mAC": float(mac),
        "avg_preds": float(avg_preds),
        "latency_s": float(latency_s),
        "EPI": float(epi)
    }
    df = pd.DataFrame([row])
    print("\n=== Single-model EPI ===")
    print(df.to_string(index=False))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print(f"Saved: {args.out_json}")

def compute_epi_multi(args):
    acc = pd.read_csv(args.accuracy_csv)
    apc = pd.read_csv(args.avgpreds_csv)
    lat = pd.read_csv(args.latency_csv)

    if "latency_s" in lat.columns:
        pass
    elif "latency_ms" in lat.columns:
        lat = lat.rename(columns={"latency_ms":"latency_s"})
        lat["latency_s"] = lat["latency_s"] / 1000.0
    elif "latency" in lat.columns:
        lat = lat.rename(columns={"latency":"latency_s"})
        lat["latency_s"] = lat["latency_s"] / 1000.0
    else:
        raise RuntimeError("latency_csv 必须包含 latency/latency_ms/latency_s 之一")

    df = acc.merge(apc, on="model").merge(lat[["model","latency_s"]], on="model")

    alpha = float(args.alpha)
    lam = float(args.lam)
    df["combined_acc"] = alpha*df["mAP"].astype(float) + lam*df["mAC"].astype(float)
    df["denom"] = (df["avg_preds"].astype(float) * df["latency_s"].astype(float)).clip(lower=1e-12)
    df["EPI"] = df["combined_acc"] / df["denom"]
    df["rank_desc"] = df["EPI"].rank(ascending=False, method="min")

    cols = ["model","mAP","mAC","avg_preds","latency_s","EPI","rank_desc","combined_acc"]
    print("\n=== Multi-model EPI ===")
    print(df[cols].sort_values("EPI", ascending=False).to_string(index=False))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    s1 = sub.add_parser("single", help="single-model quick test")
    s1.add_argument("--model", default="model_x")
    s1.add_argument("--mac_value", type=float)
    s1.add_argument("--mac_csv")
    s1.add_argument("--mac_module")
    s1.add_argument("--gt")
    s1.add_argument("--pred")
    s1.add_argument("--tau", type=float, default=0.5)
    s1.add_argument("--map_value", type=float)
    s1.add_argument("--map_csv")
    s1.add_argument("--avgpreds_value", type=float)
    s1.add_argument("--avgpreds_csv")
    s1.add_argument("--latency_s", type=float)
    s1.add_argument("--latency_ms", type=float)
    s1.add_argument("--latency_csv")
    s1.add_argument("--alpha", type=float, default=1.0)
    s1.add_argument("--lam", type=float, default=1.0)
    s1.add_argument("--out_csv")
    s1.add_argument("--out_json")

    s2 = sub.add_parser("multi", help="multi-model from CSVs")
    s2.add_argument("--accuracy_csv", required=True)
    s2.add_argument("--avgpreds_csv", required=True)
    s2.add_argument("--latency_csv", required=True)
    s2.add_argument("--alpha", type=float, default=1.0)
    s2.add_argument("--lam", type=float, default=1.0)
    s2.add_argument("--out_csv")

    args = ap.parse_args()
    if args.mode == "single":
        compute_epi_single(args)
    else:
        compute_epi_multi(args)

def summarize_all(out_dir: str):
    files = glob.glob(os.path.join(out_dir, "epi_alpha*_lam*.csv"))
    if not files:
        print("No epi_alpha*_lam*.csv found to summarize.")
        return
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        cfg = os.path.splitext(os.path.basename(f))[0]
        df["config"] = cfg
        dfs.append(df)
    summary = pd.concat(dfs, ignore_index=True)
    summary_path = os.path.join(out_dir, "epi_summary_all.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved: {summary_path}")
    cols = ["config","model","EPI","rank_desc","mAP","mAC","avg_preds","latency_s"]
    print(summary[cols].sort_values(["config","EPI"], ascending=[True, False]).to_string(index=False))

if __name__ == "__main__":
    # 直接批量跑不同 α/λ 组合（无需命令行）
    run_epi_batch_via_subprocess()
    # 如果要保留命令行模式，用下面的替换：
    # main()
