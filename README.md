# mAC / VC / EPI Metrics

This repository contains code and experiment scripts for computing **mAC**, **EPI**, and **VC** metrics with YOLO models across CIFAR-100, MRI, and VisDrone datasets. The repo also includes selected result summaries and plots for reference.

## Repository Structure
- `mAC/` Core mAC implementation and callbacks.
- `cifar100/`, `MRI/`, `VisDrone/` Dataset-specific scripts and utilities.
- `train.py` Example training entrypoint (writes outputs to `result/`).
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
3. Run training:
   ```bash
   python train.py
   ```

Outputs are written to `result/<model>_s<seed>/metrics/` unless `RESULT_ROOT` is set.

## Dataset Paths
By default, `train.py` uses dataset files within the repository root. If your datasets are stored elsewhere, set `DATA_ROOT` to your dataset root before running.

## Results Included
This repository includes selected summary CSVs and plots under dataset folders (e.g., `cifar100/`, `MRI/`, `VisDrone/`) for reference.

## License
See `LICENSE`.
