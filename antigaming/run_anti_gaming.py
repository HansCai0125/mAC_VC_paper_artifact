#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml


# ============================================================
# IDE direct-run defaults
# Set USE_IDE_DEFAULTS = True to run directly in PyCharm/IDE
# ============================================================
USE_IDE_DEFAULTS = True

IDE_DEFAULTS = {
    "data": r"D:\Desktop\antigaming\MSCOCO\data.yaml",
    "model": r"D:\Desktop\antigaming\best.pt",
    "split": "val",
    "attack": "threshold",   # shrink / expand / duplicate / threshold
    "severities": "",
    "baseline_json": "",
    "cache_dir": r"D:\Desktop\antigaming\runs\anti_gaming_cache",
    "out_csv": r"D:\Desktop\antigaming\runs\anti_gaming\visdrone_shrink.csv",
    "device": "0",
    "conf_export": 0.001,
    "iou_export": 0.95,
    "max_det": 500,
    "match_iou": 0.5,
    "latency_s": 1.0,
    "alpha": 1.0,
    "lam": 1.0,
    "jitter_xy": 0.02,
    "jitter_wh": 0.01,
    "seed": 0,
}


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class BoxRecord:
    cls: int
    conf: float
    xyxy: List[float]


# ----------------------------
# Basic geometry
# ----------------------------
def clip_box(xyxy: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = box_area([ix1, iy1, ix2, iy2])
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def scale_box_about_center(box: List[float], scale: float, w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    nw = bw * scale
    nh = bh * scale
    out = [cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0]
    return clip_box(out, w, h)


def jitter_box(box: List[float], jitter_xy: float, jitter_wh: float, w: int, h: int, rng: random.Random) -> List[float]:
    x1, y1, x2, y2 = box
    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)
    cx = (x1 + x2) / 2.0 + rng.uniform(-jitter_xy, jitter_xy) * bw
    cy = (y1 + y2) / 2.0 + rng.uniform(-jitter_xy, jitter_xy) * bh
    nw = bw * (1.0 + rng.uniform(-jitter_wh, jitter_wh))
    nh = bh * (1.0 + rng.uniform(-jitter_wh, jitter_wh))
    out = [cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0]
    return clip_box(out, w, h)


# ----------------------------
# Dataset IO
# ----------------------------
def load_data_yaml(data_yaml: Path) -> Dict:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def resolve_split_paths(data_yaml: Path, split: str = "val") -> Tuple[Path, List[Path], Dict[int, str]]:
    data = load_data_yaml(data_yaml)
    root = Path(data.get("path", data_yaml.parent)).expanduser()
    split_ref = data[split]

    if isinstance(split_ref, str):
        split_path = Path(split_ref)
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()
    else:
        raise ValueError(f"Unsupported {split} spec in YAML: {type(split_ref)}")

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    else:
        names = {}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if split_path.is_dir():
        images = sorted([p for p in split_path.rglob("*") if p.suffix.lower() in exts])
    elif split_path.is_file() and split_path.suffix.lower() == ".txt":
        images = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (root / p).resolve()
            images.append(p)
    else:
        raise FileNotFoundError(f"Could not resolve image split path: {split_path}")

    return split_path, images, names


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
        return lbl
    except ValueError:
        s = str(image_path)
        s = s.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        return Path(s).with_suffix(".txt")


def yolo_label_line_to_xyxy(line: str, img_w: int, img_h: int) -> BoxRecord:
    vals = line.strip().split()
    if len(vals) < 5:
        raise ValueError(f"Invalid YOLO label line: {line}")
    cls = int(float(vals[0]))
    xc, yc, bw, bh = map(float, vals[1:5])
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return BoxRecord(cls=cls, conf=1.0, xyxy=[x1, y1, x2, y2])


def read_image_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(image_path) as im:
        w, h = im.size
    return w, h


def build_gt_records(data_yaml: Path, split: str = "val") -> List[Dict]:
    _, image_paths, _ = resolve_split_paths(data_yaml, split=split)
    records = []
    for img_path in image_paths:
        w, h = read_image_size(img_path)
        label_path = image_to_label_path(img_path)
        gts = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                gts.append(yolo_label_line_to_xyxy(line, w, h).__dict__)

        records.append(
            {
                "image_id": img_path.stem,
                "image_path": str(img_path),
                "width": w,
                "height": h,
                "gts": gts,
                "preds": [],
            }
        )
    return records


# ----------------------------
# Inference export
# ----------------------------
def export_baseline_predictions(
    data_yaml: Path,
    model_path: str,
    split: str,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    save_json: Path,
) -> List[Dict]:
    try:
        from ultralytics import RTDETR, YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics is required for baseline export.") from e

    records = build_gt_records(data_yaml, split=split)

    model_name = Path(model_path).name.lower()
    model = RTDETR(model_path) if "rtdetr" in model_name else YOLO(model_path)

    for rec in records:
        img_path = rec["image_path"]
        results = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
            device=device,
        )
        res = results[0]
        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            cls = res.boxes.cls.detach().cpu().numpy().astype(int)
            confs = res.boxes.conf.detach().cpu().numpy()
            for b, c, cf in zip(xyxy, cls, confs):
                preds.append(
                    {
                        "cls": int(c),
                        "conf": float(cf),
                        "xyxy": [float(x) for x in b.tolist()],
                    }
                )
        rec["preds"] = preds

    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return records


def load_baseline_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Matching for matched-only attacks
# ----------------------------
def greedy_match_indices(gts: List[Dict], preds: List[Dict], iou_thr: float = 0.5) -> List[Tuple[int, int]]:
    pairs = []
    used_pred = set()
    for gi, gt in enumerate(gts):
        best_pi = None
        best_iou = -1.0
        for pi, pr in enumerate(preds):
            if pi in used_pred:
                continue
            if int(pr["cls"]) != int(gt["cls"]):
                continue
            ov = iou_xyxy(gt["xyxy"], pr["xyxy"])
            if ov >= iou_thr and ov > best_iou:
                best_iou = ov
                best_pi = pi
        if best_pi is not None:
            used_pred.add(best_pi)
            pairs.append((gi, best_pi))
    return pairs


# ----------------------------
# Attacks
# ----------------------------
def apply_shrink_attack(records: List[Dict], scale: float, match_iou: float) -> List[Dict]:
    out = copy.deepcopy(records)
    for rec in out:
        pairs = greedy_match_indices(rec["gts"], rec["preds"], iou_thr=match_iou)
        w, h = rec["width"], rec["height"]
        for _, pi in pairs:
            rec["preds"][pi]["xyxy"] = scale_box_about_center(rec["preds"][pi]["xyxy"], scale, w, h)
    return out


def apply_expand_attack(records: List[Dict], scale: float, match_iou: float) -> List[Dict]:
    return apply_shrink_attack(records, scale, match_iou)


def apply_duplicate_attack(
    records: List[Dict],
    dup_factor: int,
    match_iou: float,
    jitter_xy: float,
    jitter_wh: float,
    seed: int,
) -> List[Dict]:
    out = copy.deepcopy(records)
    rng = random.Random(seed)
    for rec in out:
        pairs = greedy_match_indices(rec["gts"], rec["preds"], iou_thr=match_iou)
        w, h = rec["width"], rec["height"]
        new_preds = []
        for _, pi in pairs:
            base = rec["preds"][pi]
            for j in range(max(0, dup_factor - 1)):
                p = copy.deepcopy(base)
                p["xyxy"] = jitter_box(base["xyxy"], jitter_xy, jitter_wh, w, h, rng)
                p["conf"] = float(base["conf"]) * (0.97 ** (j + 1))
                new_preds.append(p)
        rec["preds"].extend(new_preds)
    return out


def apply_threshold_attack(records: List[Dict], conf_thr: float) -> List[Dict]:
    out = copy.deepcopy(records)
    for rec in out:
        rec["preds"] = [p for p in rec["preds"] if float(p["conf"]) >= conf_thr]
    return out


# ----------------------------
# mAC utilities
# ----------------------------
def rect_union_area(rects: List[List[float]]) -> float:
    rects = [r for r in rects if box_area(r) > 0]
    if not rects:
        return 0.0

    xs = sorted(set([r[0] for r in rects] + [r[2] for r in rects]))
    area = 0.0

    for i in range(len(xs) - 1):
        x_left = xs[i]
        x_right = xs[i + 1]
        if x_right <= x_left:
            continue

        intervals = []
        for x1, y1, x2, y2 in rects:
            if x1 < x_right and x2 > x_left:
                intervals.append((y1, y2))

        if not intervals:
            continue

        intervals.sort()
        merged = []
        cy1, cy2 = intervals[0]
        for y1, y2 in intervals[1:]:
            if y1 <= cy2:
                cy2 = max(cy2, y2)
            else:
                merged.append((cy1, cy2))
                cy1, cy2 = y1, y2
        merged.append((cy1, cy2))

        y_union = sum(max(0.0, b - a) for a, b in merged)
        area += (x_right - x_left) * y_union

    return area


def clip_pred_to_gt(pred_box: List[float], gt_box: List[float]) -> List[float]:
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box
    return [max(px1, gx1), max(py1, gy1), min(px2, gx2), min(py2, gy2)]


def compute_mac(records: List[Dict]) -> float:
    vals = []
    for rec in records:
        preds_by_cls = defaultdict(list)
        for p in rec["preds"]:
            preds_by_cls[int(p["cls"])].append(p["xyxy"])

        for gt in rec["gts"]:
            gt_cls = int(gt["cls"])
            gt_box = gt["xyxy"]
            gt_area = box_area(gt_box)
            if gt_area <= 0:
                continue

            clipped = []
            for pbox in preds_by_cls.get(gt_cls, []):
                inter_box = clip_pred_to_gt(pbox, gt_box)
                if box_area(inter_box) > 0:
                    clipped.append(inter_box)

            cov = rect_union_area(clipped) / gt_area if clipped else 0.0
            vals.append(float(max(0.0, min(1.0, cov))))

    return float(np.mean(vals)) if vals else 0.0


# ----------------------------
# mAP utilities (self-contained COCO-style AP)
# ----------------------------
def ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap_at_iou(records: List[Dict], iou_thr: float) -> float:
    gt_count_by_cls = defaultdict(int)
    gt_by_image_cls = defaultdict(list)
    pred_by_cls = defaultdict(list)

    for rec in records:
        image_id = rec["image_id"]
        for gt in rec["gts"]:
            c = int(gt["cls"])
            gt_count_by_cls[c] += 1
            gt_by_image_cls[(image_id, c)].append(gt["xyxy"])

        for pred in rec["preds"]:
            c = int(pred["cls"])
            pred_by_cls[c].append((image_id, float(pred["conf"]), pred["xyxy"]))

    aps = []
    all_classes = sorted(set(list(gt_count_by_cls.keys()) + list(pred_by_cls.keys())))

    for c in all_classes:
        n_gt = gt_count_by_cls[c]
        preds = sorted(pred_by_cls[c], key=lambda x: x[1], reverse=True)

        if n_gt == 0:
            continue

        matched = {
            key: np.zeros(len(boxes), dtype=bool)
            for key, boxes in gt_by_image_cls.items()
            if key[1] == c
        }

        tp = np.zeros(len(preds), dtype=np.float64)
        fp = np.zeros(len(preds), dtype=np.float64)

        for i, (image_id, _, pbox) in enumerate(preds):
            gt_boxes = gt_by_image_cls.get((image_id, c), [])
            if not gt_boxes:
                fp[i] = 1.0
                continue

            ious = np.array([iou_xyxy(pbox, g) for g in gt_boxes], dtype=np.float64)
            best = int(np.argmax(ious)) if len(ious) else -1
            best_iou = float(ious[best]) if len(ious) else 0.0

            if best_iou >= iou_thr and not matched[(image_id, c)][best]:
                tp[i] = 1.0
                matched[(image_id, c)][best] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        aps.append(ap_from_pr(precision, recall))

    return float(np.mean(aps)) if aps else 0.0


def compute_map(records: List[Dict]) -> Tuple[float, float]:
    ap50 = compute_ap_at_iou(records, 0.50)
    thrs = np.arange(0.50, 0.96, 0.05)
    ap_all = [compute_ap_at_iou(records, t) for t in thrs]
    ap5095 = float(np.mean(ap_all)) if ap_all else 0.0
    return ap50, ap5095


# ----------------------------
# Composite metrics
# ----------------------------
def harmonic_mean(a: float, b: float, eps: float = 1e-12) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return float(2.0 * a * b / max(a + b, eps))


def compute_avg_preds(records: List[Dict]) -> float:
    if not records:
        return 0.0
    return float(np.mean([len(r["preds"]) for r in records]))


def compute_epi(m: float, mac: float, avg_preds: float, latency_s: float, alpha: float, lam: float) -> float:
    denom = max(avg_preds * max(latency_s, 1e-12), 1e-12)
    return float((alpha * m + lam * mac) / denom)


def evaluate_records(records: List[Dict], latency_s: float, alpha: float, lam: float) -> Dict[str, float]:
    mAP50, mAP5095 = compute_map(records)
    mAC = compute_mac(records)
    VC50 = harmonic_mean(mAP50, mAC)
    VC5095 = harmonic_mean(mAP5095, mAC)
    avg_preds = compute_avg_preds(records)
    EPI50 = compute_epi(mAP50, mAC, avg_preds, latency_s, alpha, lam)
    EPI5095 = compute_epi(mAP5095, mAC, avg_preds, latency_s, alpha, lam)

    return {
        "mAP50": mAP50,
        "mAP50_95": mAP5095,
        "mAC": mAC,
        "VC50": VC50,
        "VC50_95": VC5095,
        "avg_preds": avg_preds,
        "EPI50": EPI50,
        "EPI50_95": EPI5095,
    }


# ----------------------------
# Experiment driver
# ----------------------------
def parse_severities(s: str, attack: str) -> List[float]:
    if s:
        return [float(x) for x in s.split(",") if x.strip()]

    presets = {
        "shrink": [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.71, 0.65, 0.60],
        "expand": [1.0, 1.05, 1.10, 1.20, 1.30, 1.41, 1.60, 1.80, 2.00],
        "duplicate": [1, 2, 4, 8, 16],
        "threshold": [0.50, 0.25, 0.10, 0.05, 0.01, 0.001],
    }
    return [float(x) for x in presets[attack]]


def run_attack(records: List[Dict], attack: str, severity: float, args) -> List[Dict]:
    if attack == "shrink":
        return apply_shrink_attack(records, scale=severity, match_iou=args.match_iou)
    if attack == "expand":
        return apply_expand_attack(records, scale=severity, match_iou=args.match_iou)
    if attack == "duplicate":
        return apply_duplicate_attack(
            records,
            dup_factor=int(round(severity)),
            match_iou=args.match_iou,
            jitter_xy=args.jitter_xy,
            jitter_wh=args.jitter_wh,
            seed=args.seed,
        )
    if attack == "threshold":
        return apply_threshold_attack(records, conf_thr=severity)

    raise ValueError(f"Unknown attack: {attack}")


def save_metrics_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "model",
        "split",
        "attack",
        "severity",
        "mAP50",
        "mAP50_95",
        "mAC",
        "VC50",
        "VC50_95",
        "avg_preds",
        "EPI50",
        "EPI50_95",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_args():
    p = argparse.ArgumentParser(description="Prototype anti-gaming evaluator for mAC / VC / EPI.")
    p.add_argument("--data", required=True, help="Path to YOLO dataset YAML")
    p.add_argument("--model", required=True, help="Model path or name used for baseline export")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--attack", required=True, choices=["shrink", "expand", "duplicate", "threshold"])
    p.add_argument("--severities", default="", help="Comma-separated severities; empty uses defaults")
    p.add_argument("--baseline-json", default="", help="Optional existing baseline JSON cache")
    p.add_argument("--cache-dir", default="runs/anti_gaming_cache")
    p.add_argument("--out-csv", default="runs/anti_gaming/results.csv")
    p.add_argument("--device", default="0")
    p.add_argument("--conf-export", type=float, default=0.001)
    p.add_argument("--iou-export", type=float, default=0.95)
    p.add_argument("--max-det", type=int, default=500)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--latency-s", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--jitter-xy", type=float, default=0.02)
    p.add_argument("--jitter-wh", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)

    if USE_IDE_DEFAULTS:
        return argparse.Namespace(**IDE_DEFAULTS)
    return p.parse_args()


def main():
    args = build_args()

    data_yaml = Path(args.data)
    cache_dir = Path(args.cache_dir)
    out_csv = Path(args.out_csv)

    if args.baseline_json:
        baseline_json = Path(args.baseline_json)
    else:
        baseline_json = cache_dir / f"{Path(args.model).stem}_{Path(args.data).stem}_{args.split}_baseline.json"

    if baseline_json.exists():
        baseline_records = load_baseline_json(baseline_json)
        print(f"Loaded baseline cache: {baseline_json}")
    else:
        print(f"Exporting baseline predictions -> {baseline_json}")
        baseline_records = export_baseline_predictions(
            data_yaml=data_yaml,
            model_path=args.model,
            split=args.split,
            conf=args.conf_export,
            iou=args.iou_export,
            max_det=args.max_det,
            device=args.device,
            save_json=baseline_json,
        )

    severities = parse_severities(args.severities, args.attack)
    rows = []
    dataset_name = Path(args.data).stem
    model_name = Path(args.model).name

    base_metrics = evaluate_records(baseline_records, args.latency_s, args.alpha, args.lam)
    rows.append(
        {
            "dataset": dataset_name,
            "model": model_name,
            "split": args.split,
            "attack": "baseline",
            "severity": 1.0,
            **base_metrics,
        }
    )

    for sev in severities:
        attacked = run_attack(baseline_records, args.attack, sev, args)
        metrics = evaluate_records(attacked, args.latency_s, args.alpha, args.lam)

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "split": args.split,
            "attack": args.attack,
            "severity": sev,
            **metrics,
        }
        rows.append(row)

        print(
            f"{args.attack:>9} severity={sev:<6} | "
            f"mAP50={metrics['mAP50']:.4f} "
            f"mAP50:95={metrics['mAP50_95']:.4f} "
            f"mAC={metrics['mAC']:.4f} "
            f"VC50:95={metrics['VC50_95']:.4f}"
        )

    save_metrics_csv(rows, out_csv)
    print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    main()
