# Mean Area Coverage (mAC) and Visual Completeness (VC)

Code, configurations and result summaries for the manuscript

> **Complementing Instance Matching with Collective Regional Coverage: Mean Area Coverage and Visual Completeness**
> Han Cai, Huiyu Zhan, Mehrdad R. Kermani. Submitted to *Pattern Recognition*.

mAC and VC are regional diagnostics that complement instance-matching metrics (mAP, F1, LRP) for object detection.

* **mAC** (Mean Area Coverage) measures how much of each annotated target is covered by the union of retained, class-consistent predictions.
* **VC** (Visual Completeness) assigns every retained prediction to at most one same-class target (highest IoU; ties broken by intersection area, then target id). It then computes the Dice agreement between each target and the union of the predictions it owns, and aggregates over targets plus orphan predictions:

  `VC(theta) = sum_i Dice(V_i, g_i) / (N + N_orphan)`

The reference implementation is `region_eval/vc_v1.py`, which matches Eqs. (1)-(16) of the paper.

> **Note on earlier versions.** Earlier, unpublished versions of this project, including the previous contents of this repository and the `mac-vc-epi` package, defined "VC" as the harmonic mean of mAP and mAC and also included an EPI metric. **That definition is not the one used in the paper.** Only the ownership-aware definition above is used.

Historical code and results remain in `mAC/`, `antigaming/`, `BTD/`, `cifar100/`, `MSCOCO/`, `VisDrone/`, `Results/`, and the root `train.py`, `config.py`, and `environment.yml`. They are retained for reference; use `region_eval/` and the entry points below for the current paper.

## Repository layout

```
region_eval/                   core library: geometry, mAC/VC (vc_v1.py), F1/AP/LRP (matching_analysis.py),
                               COCO and VisDrone loading and ignore/crowd filtering (coco.py, visdrone.py),
                               and one module per analysis
run_vc_v1_*.py                 entry points for the analyses listed below
tests/                         unit and property tests (pytest)
examples/region_eval/          small synthetic input for the pure-function API
experiments/
  real_output_diagnostics_v1/  residual support, foreground enrichment, translation reference,
                               fixed-threshold incremental R^2, annotation coarsening, Holm family
  downstream_sam_v1/           frozen SAM ViT-B box-prompted segmentation and threshold selection
scripts/                       final figure scripts (replot_pr_fig345.py, fig56_dense_20261023/)
outputs/vc_v1/                 summary CSV/JSON/PDF files for each analysis (per-image caches removed)
plot/MSCOCO, plot/VisDrone     the 35 archived training runs: args.yaml, results.csv, hparams.json,
                               weights_paths.txt (checkpoints are not included)
*.md (root)                    protocol notes for individual analyses
```

## Where each result in the paper comes from

| Paper item | Entry point | Summary files |
|---|---|---|
| Table 2: collective vs. best single prediction | `run_vc_v1_direct_metric_comparison.py` | `outputs/vc_v1/direct_metric_comparison_v1/model_comparison.csv`, `per_run_comparison.csv` |
| Eq. (18), Sec. 4.2.1 supporting breakdowns | `run_vc_v1_natural_collective_gain.py` | `outputs/vc_v1/natural_collective_gain_v1/*.csv` |
| Table 3: box-to-mask coverage validation | `run_vc_v1_mask_grounded_validity_exact.py` | `outputs/vc_v1/mask_grounded_validity_exact_v1/model_summary.csv`, `threshold_by_*.csv`, `size_by_*.csv` |
| Table 4: residual support and foreground enrichment | `experiments/real_output_diagnostics_v1/run_experiment.py --mode formal` | `.../outputs/formal_v1/reports/A_seed_mean_SD.csv`, `B_C_seed_mean_SD.csv`, `main_results.csv` |
| Table 5: detection results (YOLO models) | `run_vc_v1_formal_matrix.py` | `outputs/vc_v1/formal_matrix_v1/aggregate_primary.csv`, `matrix_thresholds.csv` |
| Table 5: RT-DETR-L and four-model table | `run_vc_v1_rtdetr_full.py`, `run_vc_v1_rtdetr_paper_results.py` | `outputs/vc_v1/paper_results_rtdetr_v1/table_main_performance_four_models.csv` |
| mAP50:95 in Table 5 | archived Ultralytics validation logs | `plot/*/*/*/results.csv` |
| Fig. 3: threshold sensitivity | `scripts/replot_pr_fig345.py` | data: `outputs/vc_v1/paper_results_rtdetr_v1/theta_sensitivity_four_models.csv`; figure: `outputs/vc_v1/pr_fig345_readable_v2/fig_ivc_theta_sensitivity.pdf` |
| Sec. 4.2.6, Table S1: operating point and ownership | `run_vc_v1_reviewer_robustness.py` | `outputs/vc_v1/reviewer_robustness_v1/operating_point_summary.csv`, `ownership_per_seed.csv` |
| Table 6(a), Tables S3-S4: frozen SAM threshold selection | `experiments/downstream_sam_v1/run_experiment.py` | `.../outputs/formal_v1/reports/model_summary.csv`, `paired_differences.csv`, `cross_model_selection.csv` |
| Table 6(b): fixed-threshold incremental R^2 | `experiments/real_output_diagnostics_v1/run_experiment.py` | `.../outputs/formal_v1/reports/C_folds_*.csv`, `main_results.csv` |
| Fig. 4: collective fragmentation | `run_vc_v1_collective_fragmentation.py`, `scripts/replot_pr_fig345.py` | `outputs/vc_v1/collective_fragmentation_v1/fragmentation_aggregate.csv`; figure: `outputs/vc_v1/pr_fig345_readable_v2/fig5_collective_fragmentation_dense.pdf` |
| Fig. 5: controlled tests of VC components | `run_vc_v1_incremental_value.py`, `scripts/replot_pr_fig345.py` | `outputs/vc_v1/incremental_value_v1/canonical_response.csv`; figure: `outputs/vc_v1/pr_fig345_readable_v2/fig6_vc_incremental_value_dense.pdf` |
| Sec. 5.3, Sec. S6: scaling and duplicate flooding | `run_vc_v1_gaming_full_ap.py` | `outputs/vc_v1/gaming_full_ap_v1/gaming_aggregate.csv` |
| Sec. 5.4, Sec. S5: annotation coarsening | `experiments/real_output_diagnostics_v1/run_experiment.py` | `.../outputs/formal_v1/reports/D2_per_seed.csv` |
| Secs. S2-S3: residual-support strata, translation reference | same | `.../reports/A3_strata_*.csv`, `A_per_seed.csv`, `B_C_per_seed.csv`, `B_area_shift_audit_*.csv` |
| Sec. S8, Table S7: 12-test Holm family | same | `.../reports/primary_tests_holm.csv` |
| Sec. S7: training settings (Table S6) | archived runs | `plot/MSCOCO/*/*/args.yaml`, `plot/VisDrone/*/*/args.yaml`, `results.csv` |
| Secondary SAM endpoints (mask AP, foreground precision/recall) | `experiments/downstream_sam_v1/run_experiment.py` | `.../outputs/formal_v1/mask_ap/summary.csv` |

The figure files in `outputs/vc_v1/pr_fig345_readable_v2/` are named after their numbering in an earlier draft. `fig5_*` is paper Fig. 4, and `fig6_*` is paper Fig. 5.

`outputs/vc_v1/mask_grounded_validity_v1/` is an earlier raster (OpenCV) version of the mask audit. The paper reports the exact polygon version in `mask_grounded_validity_exact_v1/`.

## Data and models

* **Datasets.** MS-COCO 2017 validation (5,000 images, with instance annotations) and VisDrone2019-DET validation (548 images). Both are publicly available from their providers under their own terms. No dataset images are redistributed here.
* **Dataset paths.** Paths are currently hard-coded to the authors' machine (`E:\Project\Database\MS-COCO`, `E:\Project\Database\VisDrone`). Edit them in:
  `run_vc_v1_formal_matrix.py`, `run_vc_v1_rtdetr_full.py`, `run_vc_v1_smoke.py`, `region_eval/fixed_checkpoint_analysis.py`, `region_eval/gaming_from_best.py`, `experiments/downstream_sam_v1/downstream/protocol.py`.
* **Detectors.** Five archived runs per setting: YOLOv8n, YOLO11n and YOLO11s on MS-COCO and VisDrone, and RT-DETR-L on VisDrone, trained with Ultralytics. The 35 run configurations and training logs are in `plot/`. Model checkpoints are not included.
* **Inference settings for the prediction cache** (Supplementary Sec. S7): image size 640, confidence floor 0.001, class-aware NMS at IoU 0.7, at most 1,000 detections, batch size 1, FP32, no test-time augmentation. RT-DETR-L uses its native predictor without extra NMS.
* **Frozen SAM.** `segment-anything` at commit `dca509fe793f601edb92606367a655c15ac00fdf`, ViT-B checkpoint `sam_vit_b_01ec64.pth`. The SHA-256 of the checkpoint is recorded in `experiments/downstream_sam_v1/weights/weight_manifest.json`; download it from the URL given there. Mask handling uses `pycocotools==2.0.11` (see `experiments/downstream_sam_v1/requirements.txt`).

## Environment

The recorded formal runs used Python 3.11.15, numpy 2.2.6, torch 2.11.0+cu128 and pycocotools 2.0.11. A package version was not recorded for every historical training run (see Supplementary Sec. S7). Install Ultralytics, PyTorch, numpy, pandas, matplotlib, pycocotools and pytest, then:

```bash
python -m pytest tests
```

The historical mAC parity test is skipped when its two original source files are unavailable in this artifact. The remaining tests validate the bundled implementation.

## What is not included, and why

* Model checkpoints and SAM weights, which are too large and are published by their providers.
* Per-image prediction caches, per-run shard folders and SQLite caches (several GB in total). The summary files above are derived from them.
* Six out-of-fold prediction tables, `experiments/real_output_diagnostics_v1/outputs/formal_v1/reports/C_oof_*.csv` (about 24 MB each). They are regenerated by `run_experiment.py`.
* Qualitative case sheets that contain crops of dataset images.

## License

Code is released under the MIT License (see `LICENSE`). MS-COCO, VisDrone, Ultralytics and Segment Anything are subject to their own licenses.

## Citation

Citation details will be added on publication.
