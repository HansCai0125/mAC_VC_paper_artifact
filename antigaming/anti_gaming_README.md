# Anti-gaming prototype for mAC / VC / EPI

This prototype is intended for a **pilot offline study** on a YOLO-format dataset.
It does **not retrain** the detector. Instead, it:

1. Exports a low-threshold baseline prediction cache from a chosen model.
2. Applies one attack at a time to the cached predictions.
3. Recomputes `mAP50`, `mAP50:95`, `mAC`, `VC50`, `VC50:95`, `avg_preds`, and `EPI`.

## What it supports now

- `shrink`: center shrink on matched predictions only
- `expand`: center expansion on matched predictions only
- `duplicate`: duplicate matched predictions with slight jitter
- `threshold`: confidence filtering on cached predictions

## Inputs expected

- A YOLO dataset YAML like `E:\Project\Database\MRI\MRI.yaml`
- A model path/name that Ultralytics can load, such as:
  - `yolo11n.pt`
  - `E:\Project\Database\MS-COCO\yolo11n_s0\weights\best.pt`
  - `rtdetr-l.pt`

## First recommended run

Use MRI + YOLO11n_s0 first.

```bash
python run_anti_gaming.py \
  --data "E:\Project\Database\MRI\MRI.yaml" \
  --model "E:\Project\Database\MS-COCO\yolo11n_s0\weights\best.pt" \
  --split val \
  --attack shrink \
  --device 0 \
  --cache-dir "runs\anti_gaming_cache" \
  --out-csv "runs\anti_gaming\mri_yolo11n_shrink.csv"
```

## Then run the other attacks

### Expand
```bash
python run_anti_gaming.py \
  --data "E:\Project\Database\MRI\MRI.yaml" \
  --model "E:\Project\Database\MS-COCO\yolo11n_s0\weights\best.pt" \
  --split val \
  --attack expand \
  --device 0 \
  --baseline-json "runs\anti_gaming_cache\best_MRI_val_baseline.json" \
  --out-csv "runs\anti_gaming\mri_yolo11n_expand.csv"
```

### Duplicate flooding
```bash
python run_anti_gaming.py \
  --data "E:\Project\Database\MRI\MRI.yaml" \
  --model "E:\Project\Database\MS-COCO\yolo11n_s0\weights\best.pt" \
  --split val \
  --attack duplicate \
  --device 0 \
  --baseline-json "runs\anti_gaming_cache\best_MRI_val_baseline.json" \
  --out-csv "runs\anti_gaming\mri_yolo11n_duplicate.csv"
```

### Threshold sweep
```bash
python run_anti_gaming.py \
  --data "E:\Project\Database\MRI\MRI.yaml" \
  --model "E:\Project\Database\MS-COCO\yolo11n_s0\weights\best.pt" \
  --split val \
  --attack threshold \
  --device 0 \
  --baseline-json "runs\anti_gaming_cache\best_MRI_val_baseline.json" \
  --out-csv "runs\anti_gaming\mri_yolo11n_threshold.csv"
```

## Recommended default severities

If you do not specify `--severities`, the script uses these presets:

- `shrink`: `1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.71, 0.65, 0.60`
- `expand`: `1.0, 1.05, 1.10, 1.20, 1.30, 1.41, 1.60, 1.80, 2.00`
- `duplicate`: `1, 2, 4, 8, 16`
- `threshold`: `0.50, 0.25, 0.10, 0.05, 0.01, 0.001`

You can override them, for example:

```bash
python run_anti_gaming.py --data ... --model ... --attack shrink --severities 1.0,0.9,0.8,0.71,0.6
```

## Important note about this prototype

This evaluator is **self-contained** so that you can run a pilot quickly on the 5060 Ti machine.
That means:

- `mAP` is computed by an internal COCO-style AP implementation.
- `mAC` is implemented as class-aware union coverage over each GT region.
- `VC` is harmonic mean of `mAC` with `mAP50` or `mAP50:95`.
- `EPI` is computed from the supplied `latency_s`, `alpha`, `lam`, and `avg_preds`.

For the final paper version, you can keep the *attack generation* part exactly the same and replace only the metric backend with your existing official evaluation pipeline if you want perfect consistency with prior experiments.

## Good next step after the pilot

1. Run MRI first.
2. Confirm the trend shape is sensible.
3. Then switch to VisDrone and focus on `duplicate` + `threshold`.
4. After the 5090 machine is back, expand to more models and seeds.
