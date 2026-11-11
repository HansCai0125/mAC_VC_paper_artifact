# Reviewer robustness analyses

Run from E:\Project\mAC with C:\Users\hansc\anaconda3\envs\yolo\python.exe run_vc_v1_reviewer_robustness.py.
The script reuses 35 full-validation post-NMS prediction caches and completed 19-threshold CSVs. It performs no training or inference. Re-running resumes any completed ownership seed without overwriting the source results.

## 1. Ownership-rule ablation

At theta = 0.25, the alternative picks the same-class positive-overlap GT with the largest intersection area. The reference rule picks the largest IoU. Both retain deterministic tie breaks. The candidate set, prediction scores, box geometry, orphan definition, and VC aggregation remain fixed.

ownership_per_seed.csv contains the 35 paired comparisons. ownership_summary.csv gives five-seed mean and sample SD for each of seven settings. ownership_table.tex is a compact tabular fragment.

The alternative lowers mean VC by 0.00076–0.00943 across settings. Among predictions with two or more possible same-class GT owners, the owner changes in 2.28%–8.55% of cases (five-seed means). All seven within-dataset model ranks remain unchanged. This supports rank robustness, but VC's absolute value remains protocol-dependent. Orphan counts are invariant by construction because the candidate-owner definition is unchanged.

The max-IoU result and assigned/orphan counts were checked against the existing production result for every one of the 35 runs.

## 2. F1-optimal operating points

For each dataset–model setting, theta_F1 maximizes mean standard F1 across the five stored seeds on the existing theta = 0.05,...,0.95 grid. Exact ties select the smaller theta. The same selected theta is then used to summarize F1, mAC, and VC (mean and sample SD over five runs). This is a post-hoc descriptive analysis on the same validation split, not an unbiased retuned benchmark.

operating_point_per_seed.csv, operating_point_summary.csv, and operating_point_table.tex contain all results. Selected thresholds are MS COCO: YOLOv8n 0.25, YOLO11n 0.20, YOLO11s 0.30; VisDrone: YOLOv8n 0.20, YOLO11n 0.20, YOLO11s 0.25, RT-DETR-L 0.45. F1, mAC, and VC within-dataset rankings remain unchanged from theta = 0.25. Absolute scores can move substantially: for VisDrone RT-DETR-L, mean mAC changes from 0.7009 to 0.6033 while VC changes from 0.4605 to 0.4887.

Both analyses should be described as robustness checks of the reported protocol. They do not establish metric superiority or independence from threshold and ownership choices.
