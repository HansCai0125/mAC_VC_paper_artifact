import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO

from config import DATA_ROOT, MAC_ROOT, RESULT_ROOT

# Metric callbacks
sys.path.append(str(MAC_ROOT))
sys.path.append(str(MAC_ROOT / "experiments" / "scripts"))
from callback_mac import MACCallback
from epi_callback import EPICallback
from vc_callback import VCCallback


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PRESETS: Dict[str, Dict[str, object]] = {
    "cifar100": {
        "data_candidates": [
            PROJECT_ROOT / "cifar100" / "cifar100_yolo_det" / "cifar100_fullbox.yaml",
            DATA_ROOT / "cifar100" / "cifar100_yolo_det" / "cifar100_fullbox.yaml",
        ],
        "default_models": ["yolo11n"],
        "train_kw": {
            "epochs": 300,
            "patience": 100,
            "batch": 1024,
            "imgsz": 192,
            "optimizer": "SGD",
            "cos_lr": True,
            "lr0": 0.01,
            "nbs": 1024,
            "amp": True,
            "warmup_epochs": 3.0,
            "mosaic": 0.0,
            "copy_paste": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "fliplr": 0.5,
            "close_mosaic": 0,
            "workers": 8,
            "device": 0,
            "val": True,
            "plots": True,
            "max_det": 500,
            "iou": 0.7,
        },
    },
    "mri": {
        "data_candidates": [
            PROJECT_ROOT / "MRI" / "MRI.yaml",
            DATA_ROOT / "MRI" / "MRI.yaml",
        ],
        "default_models": ["yolov8n"],
        "train_kw": {
            "epochs": 300,
            "patience": 100,
            "batch": 96,
            "imgsz": 640,
            "pretrained": True,
            "optimizer": "SGD",
            "cos_lr": True,
            "close_mosaic": 10,
            "amp": True,
            "rect": False,
            "workers": 4,
            "freeze": 0,
            "val": True,
            "iou": 0.7,
            "max_det": 500,
            "plots": True,
            "hsv_h": 0.015,
            "hsv_s": 0.5,
            "hsv_v": 0.3,
            "translate": 0.05,
            "scale": 0.3,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.1,
            "copy_paste_mode": "flip",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "nbs": 64,
            "device": 0,
        },
    },
    "visdrone": {
        "data_candidates": [
            PROJECT_ROOT / "VisDrone" / "visdrone.yaml",
            DATA_ROOT / "VisDrone" / "visdrone.yaml",
        ],
        "default_models": ["yolo11n"],
        "train_kw": {
            "epochs": 300,
            "patience": 100,
            "batch": 64,
            "imgsz": 640,
            "pretrained": True,
            "optimizer": "SGD",
            "cos_lr": True,
            "close_mosaic": 10,
            "amp": True,
            "rect": False,
            "workers": 4,
            "freeze": 0,
            "val": True,
            "iou": 0.7,
            "max_det": 500,
            "plots": True,
            "hsv_h": 0.015,
            "hsv_s": 0.5,
            "hsv_v": 0.3,
            "translate": 0.05,
            "scale": 0.3,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.1,
            "copy_paste_mode": "flip",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "nbs": 64,
            "device": 0,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training entrypoint for CIFAR100, MRI, and VisDrone.")
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()), default="cifar100")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML file (overrides preset).")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names (e.g. yolo11n,yolo11s).")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds (e.g. 0,1,2,3,4).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--project", type=str, default=None, help="Output root directory.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--clean", action="store_true", help="Delete an existing run directory before training.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit.")
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_seeds(raw: str) -> List[int]:
    items = parse_csv_list(raw)
    if not items:
        raise ValueError("At least one seed is required.")
    return [int(x) for x in items]


def resolve_data_path(dataset: str, explicit_data: str | None) -> str:
    if explicit_data:
        p = Path(explicit_data).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset yaml not found: {p}")
        return str(p)

    for p in DATASET_PRESETS[dataset]["data_candidates"]:
        pp = Path(p).resolve()
        if pp.exists():
            return str(pp)

    searched = [str(Path(x).resolve()) for x in DATASET_PRESETS[dataset]["data_candidates"]]
    raise FileNotFoundError(f"No dataset yaml found for '{dataset}'. Checked: {searched}")


def build_runtime_config(args: argparse.Namespace) -> Dict[str, object]:
    preset = DATASET_PRESETS[args.dataset]
    models = parse_csv_list(args.models) if args.models else list(preset["default_models"])
    seeds = parse_seeds(args.seeds)
    data_path = resolve_data_path(args.dataset, args.data)
    train_kw = dict(preset["train_kw"])
    train_kw["data"] = data_path

    if args.epochs is not None:
        train_kw["epochs"] = args.epochs
    if args.batch is not None:
        train_kw["batch"] = args.batch
    if args.imgsz is not None:
        train_kw["imgsz"] = args.imgsz
    if args.device is not None:
        train_kw["device"] = args.device
    if args.workers is not None:
        train_kw["workers"] = args.workers
    if args.patience is not None:
        train_kw["patience"] = args.patience

    if args.amp and args.no_amp:
        raise ValueError("Use only one of --amp and --no-amp.")
    if args.amp:
        train_kw["amp"] = True
    if args.no_amp:
        train_kw["amp"] = False

    project_dir = Path(args.project).expanduser().resolve() if args.project else (RESULT_ROOT / args.dataset)
    summary_csv = project_dir / "summary.csv"

    return {
        "dataset": args.dataset,
        "models": models,
        "seeds": seeds,
        "train_kw": train_kw,
        "project_dir": project_dir,
        "summary_csv": summary_csv,
        "clean": args.clean,
        "dry_run": args.dry_run,
    }


def ensure_summary_header(summary_csv: Path):
    header = [
        "timestamp",
        "dataset",
        "model",
        "seed",
        "exp_name",
        "save_dir",
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "lr/pg0",
        "lr/pg1",
        "lr/pg2",
        "best_path",
        "last_path",
        "status",
        "error",
    ]
    if not summary_csv.exists():
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def read_last_row(csv_path: Path):
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        return rows[0], rows[-1]
    except Exception:
        return None


def clean_num(val):
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


def append_summary(dataset: str, model_name: str, seed: int, save_dir: Path, summary_csv: Path, status: str = "ok", err_msg: str = ""):
    results_csv = save_dir / "results.csv"
    item = read_last_row(results_csv)

    wdir = save_dir / "weights"
    best_pt = (wdir / "best.pt").resolve() if (wdir / "best.pt").exists() else None
    last_pt = (wdir / "last.pt").resolve() if (wdir / "last.pt").exists() else None

    if item:
        header, last = item
        col = {h: (last[i] if i < len(last) else "") for i, h in enumerate(header)}
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset,
            model_name,
            seed,
            save_dir.name,
            str(save_dir),
            clean_num(col.get("epoch", "")),
            clean_num(col.get("metrics/precision(B)", "")),
            clean_num(col.get("metrics/recall(B)", "")),
            clean_num(col.get("metrics/mAP50(B)", "")),
            clean_num(col.get("metrics/mAP50-95(B)", "")),
            clean_num(col.get("val/box_loss", "")),
            clean_num(col.get("val/cls_loss", "")),
            clean_num(col.get("val/dfl_loss", "")),
            clean_num(col.get("lr/pg0", "")),
            clean_num(col.get("lr/pg1", "")),
            clean_num(col.get("lr/pg2", "")),
            str(best_pt) if best_pt else "",
            str(last_pt) if last_pt else "",
            status,
            err_msg.strip(),
        ]
    else:
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset,
            model_name,
            seed,
            save_dir.name,
            str(save_dir),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            str(best_pt) if best_pt else "",
            str(last_pt) if last_pt else "",
            status,
            err_msg.strip(),
        ]

    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def set_global_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_load_model(model_name: str):
    try:
        local_pt = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        if os.path.exists(local_pt):
            return YOLO(local_pt)
    except Exception:
        pass

    try:
        return YOLO(model_name)
    except Exception as e:
        print(f"[WARN] Load model '{model_name}' failed: {e}")
        return None


def attach_callbacks(model: YOLO, exp_metrics_dir: Path):
    exp_metrics_dir.mkdir(parents=True, exist_ok=True)
    mac_csv = str(exp_metrics_dir / "mac_epoch.csv")
    epi_csv = str(exp_metrics_dir / "epi_results.csv")
    vc_csv = str(exp_metrics_dir / "vc_epoch.csv")

    mac_cb = MACCallback(tau=0.50, score_thresh=0.25, class_aware=True, mac_out_csv=mac_csv)
    epi_cb = EPICallback(alpha=0.5, lam=0.5, out_csv=epi_csv, prefer_map50=True, mac_csv=mac_csv)
    vc_cb = VCCallback(epsilon=0.01, out_csv=vc_csv, mac_csv=mac_csv)

    model.add_callback("on_val_start", mac_cb.on_val_start)
    model.add_callback("on_val_end", mac_cb.on_val_end)
    model.add_callback("on_train_start", mac_cb.on_train_start)
    model.add_callback("on_train_end", mac_cb.on_train_end)
    model.add_callback("on_fit_epoch_end", epi_cb.on_fit_epoch_end)
    model.add_callback("on_val_end", vc_cb.on_val_end)
    model.add_callback("on_train_end", vc_cb.on_train_end)


def dump_hparams(save_root: Path, model_name: str, seed: int, train_args: dict):
    out = {
        "model": model_name,
        "seed": seed,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_args": train_args,
    }
    with open(save_root / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def print_runtime_config(cfg: Dict[str, object]):
    view = {
        "dataset": cfg["dataset"],
        "models": cfg["models"],
        "seeds": cfg["seeds"],
        "project_dir": str(cfg["project_dir"]),
        "summary_csv": str(cfg["summary_csv"]),
        "clean": cfg["clean"],
        "train_kw": cfg["train_kw"],
    }
    print(json.dumps(view, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    cfg = build_runtime_config(args)
    project_dir: Path = cfg["project_dir"]
    summary_csv: Path = cfg["summary_csv"]
    project_dir.mkdir(parents=True, exist_ok=True)
    ensure_summary_header(summary_csv)

    print("[INFO] Runtime configuration:")
    print_runtime_config(cfg)
    if cfg["dry_run"]:
        print("[INFO] Dry run enabled. Exit without training.")
        return

    failed_models = []
    for m in cfg["models"]:
        print(f"\n[MODEL] ===== {m} =====")
        for s in cfg["seeds"]:
            model = safe_load_model(m)
            if model is None:
                if m not in failed_models:
                    failed_models.append(m)
                print(f"[SKIP] {m} (cannot load)")
                save_root = project_dir / f"{m}_s{s}"
                save_root.mkdir(parents=True, exist_ok=True)
                append_summary(cfg["dataset"], m, s, save_root, summary_csv, status="load_failed", err_msg="model load failed")
                continue

            set_global_seed(s)

            exp_name = f"{m}_s{s}"
            save_root = project_dir / exp_name
            if cfg["clean"] and save_root.exists():
                shutil.rmtree(save_root, ignore_errors=True)
            save_root.mkdir(parents=True, exist_ok=True)

            metrics_dir = save_root / "metrics"
            attach_callbacks(model, metrics_dir)

            train_args = dict(cfg["train_kw"])
            dump_hparams(save_root, m, s, train_args)

            print(f"[TRAIN] Dataset={cfg['dataset']} | Model={m} | Seed={s} | Save={save_root}")
            status, err = "ok", ""
            try:
                model.train(
                    **cfg["train_kw"],
                    seed=s,
                    deterministic=True,
                    project=str(project_dir),
                    name=exp_name,
                    exist_ok=True,
                    resume=False,
                )
            except Exception as e:
                status, err = "train_error", f"{type(e).__name__}: {e}"
                err_log = save_root / "error.txt"
                with open(err_log, "w", encoding="utf-8") as f:
                    f.write(f"{type(e).__name__}: {e}\n")
                print(f"[ERROR] {m} s={s} -> {e}")

            append_summary(cfg["dataset"], m, s, save_root, summary_csv, status=status, err_msg=err)

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

    print("\n[FINISHED] Training jobs completed.")
    print(f"Summary file: {summary_csv}")
    print("Per-run mAC/EPI/VC CSV files are under each run's metrics/ directory.")


if __name__ == "__main__":
    main()
