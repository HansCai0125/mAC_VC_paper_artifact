# EPI evaluation utility
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EPI = (alpha * mAP + lambda * mAC) / (Avg#Predictions * Latency_s)

import argparse, pandas as pd, numpy as np, os

def _load_csv(path, required_cols):
    """
    Robust CSV reader: try multiple encodings + delimiters.
    Accepts encodings: utf-8, utf-8-sig, gbk, gb18030, cp1252, latin1
            delimiters: ',', ';', '\t'
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "cp1252", "latin1"]
    seps = [",", ";", "\t"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                df.columns = [str(c).strip() for c in df.columns]
                missing = [c for c in required_cols if c not in df.columns]
                if not missing:
                    return df
            except Exception as e:
                last_err = e
                continue
    raise ValueError(
        f"Failed to parse CSV: {path}. Last error: {last_err}. "
        f"Please ensure encoding is UTF-8/GBK, delimiter is comma/semicolon/tab, and columns include {required_cols}."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accuracy", required=True, help="CSV: model,mAP,mAC")
    ap.add_argument("--avgpreds", required=True, help="CSV: model,avg_preds")
    ap.add_argument("--latency", required=True, help="CSV: model,latency | latency_ms | latency_s")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--latency_unit", choices=["ms","s"], default="ms")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acc = _load_csv(args.accuracy, ["model", "mAP", "mAC"])
    apc = _load_csv(args.avgpreds, ["model", "avg_preds"])
    lat = _load_csv(args.latency, ["model"])  # latency_* will be normalized to seconds below

    # normalize latency to seconds
    if "latency_s" in lat.columns:
        pass
    elif "latency_ms" in lat.columns:
        lat = lat.rename(columns={"latency_ms":"latency_s"})
        lat["latency_s"] = lat["latency_s"] / 1000.0
    elif "latency" in lat.columns:
        lat = lat.rename(columns={"latency":"latency_s"})
        if args.latency_unit == "ms":
            lat["latency_s"] = lat["latency_s"] / 1000.0
    else:
        raise ValueError("latency csv must include latency/latency_ms/latency_s")

    df = acc.merge(apc, on="model").merge(lat[["model","latency_s"]], on="model")

    df["combined_acc"] = args.alpha*df["mAP"].astype(float) + args.lam*df["mAC"].astype(float)
    denom = (df["avg_preds"].astype(float) * df["latency_s"].astype(float)).replace(0, np.nan)
    df["EPI"] = df["combined_acc"] / denom
    df["rank_desc"] = df["EPI"].rank(ascending=False, method="min")
    df["alpha"] = args.alpha; df["lambda"] = args.lam

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.sort_values("EPI", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
