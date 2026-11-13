# Reproducing the paper

Run commands from the repository root. Paths below are relative to that directory.

## Where each result in the paper comes from

| Paper item | Entry point | Summary files |
|---|---|---|
| Table 2: collective vs. best single prediction | `experiments/compare_metrics.py` | `results/metric_comparison/model_comparison.csv`, `per_run_comparison.csv` |
| Eq. (18), Sec. 4.2.1 supporting breakdowns | `experiments/collective_gain.py` | `results/collective_gain/*.csv` |
| Table 3: box-to-mask coverage validation | `experiments/mask_polygon.py` | `results/mask_polygon/model_summary.csv`, `threshold_by_*.csv`, `size_by_*.csv` |
| Table 4: residual support and foreground enrichment | `experiments/diagnostics/run_experiment.py --mode formal` | `.../results/main/reports/A_seed_mean_SD.csv`, `B_C_seed_mean_SD.csv`, `main_results.csv` |
| Table 5: detection results (YOLO models) | `experiments/evaluate_detectors.py` | `results/detection/aggregate_primary.csv`, `matrix_thresholds.csv` |
| Table 5: RT-DETR-L and four-model table | `experiments/evaluate_rtdetr.py`, `experiments/paper_tables.py` | `results/paper_tables/table_main_performance_four_models.csv` |
| mAP50:95 in Table 5 | archived Ultralytics validation logs | `training_logs/*/*/*/results.csv` |
| Fig. 3: threshold sensitivity | `scripts/plot_paper_figures.py` | data: `results/paper_tables/theta_sensitivity_four_models.csv`; figure: `results/figures/fig_ivc_theta_sensitivity.pdf` |
| Sec. 4.2.6, Table S1: operating point and ownership | `experiments/robustness.py` | `results/robustness/operating_point_summary.csv`, `ownership_per_seed.csv` |
| Table 6(a), Tables S3-S4: frozen SAM threshold selection | `experiments/segmentation/run_experiment.py` | `.../results/main/reports/model_summary.csv`, `paired_differences.csv`, `cross_model_selection.csv` |
| Table 6(b): fixed-threshold incremental R^2 | `experiments/diagnostics/run_experiment.py` | `.../results/main/reports/C_folds_*.csv`, `main_results.csv` |
| Fig. 4: collective fragmentation | `experiments/fragmentation.py`, `scripts/plot_paper_figures.py` | `results/fragmentation/fragmentation_aggregate.csv`; figure: `results/figures/fig5_collective_fragmentation_dense.pdf` |
| Fig. 5: controlled tests of VC components | `experiments/vc_components.py`, `scripts/plot_paper_figures.py` | `results/vc_components/canonical_response.csv`; figure: `results/figures/fig6_vc_incremental_value_dense.pdf` |
| Sec. 5.3, Sec. S6: scaling and duplicate flooding | `experiments/perturbations.py` | `results/perturbations/gaming_aggregate.csv` |
| Sec. 5.4, Sec. S5: annotation coarsening | `experiments/diagnostics/run_experiment.py` | `.../results/main/reports/D2_per_seed.csv` |
| Secs. S2-S3: residual-support strata, translation reference | same | `.../reports/A3_strata_*.csv`, `A_per_seed.csv`, `B_C_per_seed.csv`, `B_area_shift_audit_*.csv` |
| Sec. S8, Table S7: 12-test Holm family | same | `.../reports/primary_tests_holm.csv` |
| Sec. S7: training settings (Table S6) | archived runs | `training_logs/MSCOCO/*/*/args.yaml`, `training_logs/VisDrone/*/*/args.yaml`, `results.csv` |
| Secondary SAM endpoints (mask AP, foreground precision/recall) | `experiments/segmentation/run_experiment.py` | `.../results/main/mask_ap/summary.csv` |

The figure files in `results/figures/` are named after their numbering in an earlier draft. `fig5_*` is paper Fig. 4, and `fig6_*` is paper Fig. 5.

Archived CSV/JSON records and training logs retain their original paths, run identifiers and code hashes as provenance. Those historical fields are not current filesystem locations; use the paths in the table above. Model caches and checkpoints still need to be supplied before rerunning full experiments.

`results/mask_raster/` is an earlier raster (OpenCV) version of the mask audit. The paper reports the exact polygon version in `mask_polygon/`.

## Data and models

* **Datasets.** MS-COCO 2017 validation (5,000 images, with instance annotations) and VisDrone2019-DET validation (548 images). Both are publicly available from their providers under their own terms. No dataset images are redistributed here.
* **Dataset paths.** Paths are currently hard-coded to the authors' machine (`E:\Project\Database\MS-COCO`, `E:\Project\Database\VisDrone`). Edit them in:
  `experiments/evaluate_detectors.py`, `experiments/evaluate_rtdetr.py`, `experiments/smoke_test.py`, `region_eval/fixed_checkpoint_analysis.py`, `region_eval/gaming_from_best.py`, `experiments/segmentation/downstream/protocol.py`.
* **Detectors.** Five archived runs per setting: YOLOv8n, YOLO11n and YOLO11s on MS-COCO and VisDrone, and RT-DETR-L on VisDrone, trained with Ultralytics. The 35 run configurations and training logs are in `training_logs/`. Model checkpoints are not included.
* **Inference settings for the prediction cache** (Supplementary Sec. S7): image size 640, confidence floor 0.001, class-aware NMS at IoU 0.7, at most 1,000 detections, batch size 1, FP32, no test-time augmentation. RT-DETR-L uses its native predictor without extra NMS.
* **Frozen SAM.** `segment-anything` at commit `dca509fe793f601edb92606367a655c15ac00fdf`, ViT-B checkpoint `sam_vit_b_01ec64.pth`. The SHA-256 of the checkpoint is recorded in `experiments/segmentation/weights/weight_manifest.json`; download it from the URL given there. Mask handling uses `pycocotools==2.0.11` (see `experiments/segmentation/requirements.txt`).

## Environment

The recorded formal runs used Python 3.11.15, numpy 2.2.6, torch 2.11.0+cu128 and pycocotools 2.0.11. A package version was not recorded for every historical training run (see Supplementary Sec. S7). Install Ultralytics, PyTorch, numpy, pandas, matplotlib, pycocotools and pytest, then:

```bash
python -m pytest tests
```

The historical mAC parity test is skipped when its two original source files are unavailable in this artifact. The remaining tests validate the bundled implementation.

## What is not included, and why

* Model checkpoints and SAM weights, which are too large and are published by their providers.
* Per-image prediction caches, per-run shard folders and SQLite caches (several GB in total). The summary files above are derived from them.
* Six out-of-fold prediction tables, `experiments/diagnostics/results/main/reports/C_oof_*.csv` (about 24 MB each). They are regenerated by `run_experiment.py`.
* Qualitative case sheets that contain crops of dataset images.

