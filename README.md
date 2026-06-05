# mAC / VC / EPI Metrics

This repository contains code and experiment scripts for computing **mAC**, **EPI**, and **VC** metrics with YOLO models and RT-DETR model across CIFAR-100, MRI, and VisDrone datasets. The repo also includes selected result summaries and plots for reference.

## Repository Structure
- `mAC/` Core mAC implementation and callbacks.
- `cifar100/`, `MRI/`, `VisDrone/` Dataset-specific scripts and utilities.
- `train.py` Unified training entrypoint with CLI options.
- `config.py` Centralized path configuration.
- `environment.yml` Conda environment definition.

## Metrics
- **mAC** is computed during validation in `mAC/callback_mac.py`.
- **EPI** is computed in `mAC/experiments/scripts/epi_callback.py`.
- **VC** is computed in `mAC/experiments/scripts/vc_callback.py` using the harmonic mean:
  - `VC50 = 2 * mAP50 * mAC / (mAP50 + mAC + eps)`
  - `VC50_95 = 2 * mAP50_95 * mAC / (mAP50_95 + mAC + eps)`

## Quick Start
1. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate mac-metrics
   ```
2. (Optional) Set paths via environment variables:
   - `DATA_ROOT` (default: `./data`)
   - `RESULT_ROOT` (default: `./result`)
   - `MAC_ROOT` (default: `./mAC`)
3. Run training (defaults to CIFAR100):
   ```bash
   python train.py
   ```
4. Run a specific dataset/model:
   ```bash
   python train.py --dataset mri --models yolov8n --seeds 0,1,2,3,4
   python train.py --dataset visdrone --models yolo11n --epochs 100 --device 0
   ```
5. Preview resolved config only:
   ```bash
   python train.py --dataset cifar100 --dry-run
   ```

Outputs are written to `result/<dataset>/<model>_s<seed>/metrics/` unless `RESULT_ROOT`/`--project` is set.

## Dataset Paths
`train.py` resolves dataset YAML in this order:
1. `--data` explicit path (if provided).
2. Repository-local preset path.
3. `DATA_ROOT` preset path.

You can always override with `--data /abs/path/to/your.yaml`.

## Results Included
This repository includes selected summary CSVs and plots under dataset folders (e.g., `cifar100/`, `MRI/`, `VisDrone/`) for reference.

## License
See `LICENSE`.
