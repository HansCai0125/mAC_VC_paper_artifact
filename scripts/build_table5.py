"""Rebuild paper Table 5 (detection results) from the archived summaries.

Offline: no dataset, model or GPU is needed. Sources:

* mAC, VC, F1 and LRP at theta = 0.25: five-run mean and sample SD from
  ``results/detection/aggregate_primary.csv`` (YOLO models) and
  ``results/paper_tables/rtdetr_aggregate_primary.csv`` (RT-DETR-L, VisDrone).
* mAP50:95: best-epoch validation value of each archived Ultralytics run in
  ``training_logs/``, then mean and sample SD over the five runs.

Run from the repository root::

    python scripts/build_table5.py
"""

import csv
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "paper_tables" / "table5_detection_results.csv"

# (dataset key, model key, paper label, training-log folder)
ROWS = (
    ("mscoco", "yolov8n", "YOLOv8n", "MSCOCO/yolo8n"),
    ("mscoco", "yolo11n", "YOLO11n", "MSCOCO/yolo11n"),
    ("mscoco", "yolo11s", "YOLO11s", "MSCOCO/yolo11s"),
    ("visdrone", "yolov8n", "YOLOv8n", "VisDrone/yolo8n"),
    ("visdrone", "yolo11n", "YOLO11n", "VisDrone/yolo11n"),
    ("visdrone", "yolo11s", "YOLO11s", "VisDrone/yolo11s"),
    ("visdrone", "rtdetr_l", "RT-DETR-L", "VisDrone/RTDETR-l"),
)
DATASET_LABELS = {"mscoco": "MS-COCO", "visdrone": "VisDrone"}
METRICS = (("mAC", "mAC"), ("VC", "VC"),
           ("F1", "standard_F1"), ("LRP", "standard_LRP"))


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _best_map(results_csv):
    rows = _read(results_csv)
    return max(float(value) for row in rows for key, value in row.items()
               if key.strip() == "metrics/mAP50-95(B)")


def build():
    aggregates = {}
    for path in (ROOT / "results" / "detection" / "aggregate_primary.csv",
                 ROOT / "results" / "paper_tables" / "rtdetr_aggregate_primary.csv"):
        for row in _read(path):
            runs = int(row.get("n_seeds") or row.get("n_runs"))
            if runs == 5:
                aggregates[(row["dataset"], row["model"])] = row

    table = []
    for dataset, model, label, folder in ROWS:
        logs = sorted((ROOT / "training_logs" / folder).glob("*/results.csv"))
        if len(logs) != 5:
            raise ValueError(f"expected five training logs in {folder}")
        values = [_best_map(path) for path in logs]
        source = aggregates[(dataset, model)]
        record = dict(
            dataset=DATASET_LABELS[dataset], model=label, n_runs=5,
            mAP50_95_mean=f"{statistics.fmean(values):.4f}",
            mAP50_95_std=f"{statistics.stdev(values):.4f}",
        )
        for name, field in METRICS:
            record[f"{name}_mean"] = f"{float(source[field + '_mean']):.4f}"
            record[f"{name}_std"] = f"{float(source[field + '_std']):.4f}"
        table.append(record)

    with open(OUTPUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return table


if __name__ == "__main__":
    for row in build():
        print(row)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
