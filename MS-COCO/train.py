import os
import sys
import csv
import json
import time
import math
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# Allow importing config from repo root
from config import MAC_ROOT

#Metric Callback
sys.path.append(str(MAC_ROOT))
sys.path.append(str(MAC_ROOT / "experiments" / "scripts"))
from callback_mac import MACCallback
from epi_callback import EPICallback
from vc_callback import VCCallback

#Configaration
MODELS = [
            'yolo11s'
]
#, , 'yolov8n',0'yolo11n'
SEEDS = [1,2,3,4]  # 5 Round with Seed

DATA_PATH = r"E:\Project\Database\MS-COCO\data.yaml"

COMMON_KW = dict(
    data=DATA_PATH,
    epochs=300,
    patience=30,
    batch=64,
    imgsz=640,
    pretrained=True,
    optimizer="SGD",
    cos_lr=True,
    amp=True,
    rect=False,
    workers=8,
    freeze=0,
    val=True,
    iou=0.7,
    max_det=500,
    plots=True,
    # --- Augmentations ---
    hsv_h=0, hsv_s=0, hsv_v=0,
    translate=0.05, scale=0, fliplr=0, flipud=0.0,
    mosaic=0, mixup=0.0, copy_paste=0, copy_paste_mode="flip",
    # --- Hyper ---
    lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    nbs=64,
    device=0,
)

#Unified Output
BASE_OUT = Path(r"E:\Project\Database\MS-COCO")
PROJECT_DIR = BASE_OUT
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

(PROJECT_DIR / "mean").mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = PROJECT_DIR / "summary.csv"

def ensure_summary_header():
    header = [
        "timestamp","model","seed","exp_name","save_dir",
        "epoch","metrics/precision(B)","metrics/recall(B)",
        "metrics/mAP50(B)","metrics/mAP50-95(B)",
        "val/box_loss","val/cls_loss","val/dfl_loss",
        "lr/pg0","lr/pg1","lr/pg2",
        "best_path","last_path","status","error"
    ]
    if not SUMMARY_CSV.exists():
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def read_last_row(csv_path: Path):
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        return rows[0], rows[-1]  # header, last
    except Exception:
        return None


def _clean_num(val):
    if val is None:
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        x = float(s)
        if math.isfinite(x):
            return f"{x:.4f}"
        return ""
    except Exception:
        return s


def append_summary(model_name: str, seed: int, save_dir: Path, status: str = "ok", err_msg: str = ""):
    results_csv = save_dir / "results.csv"
    item = read_last_row(results_csv)

    # 提取 best/last 权重路径
    wdir = save_dir / "weights"
    best_pt = (wdir / "best.pt").resolve() if (wdir / "best.pt").exists() else None
    last_pt = (wdir / "last.pt").resolve() if (wdir / "last.pt").exists() else None

    if item:
        header, last = item
        col = {h: (last[i] if i < len(last) else "") for i, h in enumerate(header)}
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            model_name, seed, save_dir.name, str(save_dir),
            _clean_num(col.get("epoch","")),
            _clean_num(col.get("metrics/precision(B)","")),
            _clean_num(col.get("metrics/recall(B)","")),
            _clean_num(col.get("metrics/mAP50(B)","")),
            _clean_num(col.get("metrics/mAP50-95(B)","")),
            _clean_num(col.get("val/box_loss","")),
            _clean_num(col.get("val/cls_loss","")),
            _clean_num(col.get("val/dfl_loss","")),
            _clean_num(col.get("lr/pg0","")),
            _clean_num(col.get("lr/pg1","")),
            _clean_num(col.get("lr/pg2","")),
            str(best_pt) if best_pt else "",
            str(last_pt) if last_pt else "",
            status,
            err_msg.strip(),
        ]
    else:
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            model_name, seed, save_dir.name, str(save_dir),
            "", "", "", "", "", "", "", "", "", "", "",
            str(best_pt) if best_pt else "",
            str(last_pt) if last_pt else "",
            status,
            err_msg.strip(),
        ]

    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def set_global_seed(seed: int):
    import numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 如需“更强”确定性，可开启（会降速）：
    # torch.use_deterministic_algorithms(True)


def safe_load_model(model_name: str):
    try:
        local_pt = model_name if model_name.endswith('.pt') else model_name + ".pt"
        if os.path.exists(local_pt):
            return YOLO(local_pt)
    except Exception:
        pass

    # 再尝试直接名称
    try:
        return YOLO(model_name)
    except Exception as e:
        print(f"[WARN] Load model '{model_name}' failed: {e}")
        return None


def attach_callbacks(model: YOLO, exp_metrics_dir: Path):
    exp_metrics_dir.mkdir(parents=True, exist_ok=True)

    mac_csv = str(exp_metrics_dir / "mac_epoch.csv")
    epi_csv = str(exp_metrics_dir / "epi_results.csv")
    vc_csv  = str(exp_metrics_dir / "vc_epoch.csv")

    mac_cb = MACCallback(
        tau=0.50,
        score_thresh=0.25,
        class_aware=True,
        mac_out_csv=mac_csv
    )
    epi_cb = EPICallback(
        alpha=0.5, lam=0.5,
        out_csv=epi_csv,
        prefer_map50=True,
        mac_csv=mac_csv
    )
    vc_cb = VCCallback(
        epsilon=0.01,
        out_csv=vc_csv,
        mac_csv=mac_csv
    )

    model.add_callback("on_val_start",     mac_cb.on_val_start)
    model.add_callback("on_val_end",       mac_cb.on_val_end)
    model.add_callback("on_train_start",   mac_cb.on_train_start)
    model.add_callback("on_train_end",     mac_cb.on_train_end)
    model.add_callback("on_fit_epoch_end", epi_cb.on_fit_epoch_end)
    model.add_callback("on_val_end",       vc_cb.on_val_end)
    model.add_callback("on_train_end",     vc_cb.on_train_end)


def dump_hparams(save_root: Path, model_name: str, seed: int, train_args: dict):
    out = {
        "model": model_name,
        "seed": seed,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_args": train_args
    }
    with open(save_root / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ========== 主流程 ==========

def main():
    print(f"[INFO] Project dir: {PROJECT_DIR}")
    ensure_summary_header()

    failed_models = []

    for m in MODELS:
        print(f"\n[MODEL] ===== {m} =====")

        for s in SEEDS:
            model = safe_load_model(m)
            if model is None:
                if m not in failed_models:
                    failed_models.append(m)
                print(f"[SKIP] {m} (cannot load)")
                # 即便某次加载失败，也写入一行占位，方便排查
                save_root = PROJECT_DIR / f"{m}_s{s}"
                save_root.mkdir(parents=True, exist_ok=True)
                append_summary(m, s, save_root, status="load_failed", err_msg="model load failed")
                continue

            set_global_seed(s)

            exp_name = f"{m}_s{s}"
            save_root = PROJECT_DIR / exp_name
            if save_root.exists():
                shutil.rmtree(save_root, ignore_errors=True)
            save_root.mkdir(parents=True, exist_ok=True)

            metrics_dir = save_root / "metrics"
            attach_callbacks(model, metrics_dir)

            train_args = dict(COMMON_KW)
            dump_hparams(save_root, m, s, train_args)

            print(f"[TRAIN] Model={m} | Seed={s} | Save={save_root}")
            status, err = "ok", ""

            try:
                _ = model.train(
                    **COMMON_KW,
                    seed=s,
                    deterministic=True,
                    project=str(PROJECT_DIR),
                    name=exp_name,
                    exist_ok=True,
                    resume=False
                )
            except Exception as e:
                status, err = "train_error", f"{type(e).__name__}: {e}"
                err_log = save_root / "error.txt"
                with open(err_log, "w", encoding="utf-8") as f:
                    f.write(f"{type(e).__name__}: {e}\n")
                print(f"[ERROR] {m} s={s} -> {e}")

            append_summary(m, s, save_root, status=status, err_msg=err)

            wdir = save_root / "weights"
            best_pt = wdir / "best.pt"
            last_pt = wdir / "last.pt"
            with open(save_root / "weights_paths.txt", "w", encoding="utf-8") as f:
                f.write(f"best: {best_pt.resolve() if best_pt.exists() else 'N/A'}\n")
                f.write(f"last: {last_pt.resolve() if last_pt.exists() else 'N/A'}\n")

    if failed_models:
        print("\n[SUMMARY] Models failed to load:")
        for mm in sorted(set(failed_models)):
            print("  -", mm)

    print("\n[FINISHED] 全部训练完成。")
    print(f"汇总文件: {SUMMARY_CSV}")
    print("每个实验的 mAC/EPI/VC CSV 在各自目录的 metrics/ 下。")


if __name__ == "__main__":
    main()
