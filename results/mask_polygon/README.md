# Exact MS COCO Polygon-Grounded Validity Analysis

This analysis uses continuous COCO polygons and saved full-validation predictions from three YOLO architectures and five configured seed runs. No training or model inference is performed.

Continuous polygon areas are audited against COCO annotation areas before any coverage result is accepted.

## Five-run results at delta = 0.05

- YOLOv8n: r(BoxAC, MaskAC)=0.994; P(box gain>0.05)=4.61%; foreground-confirmed=2.02%; conditional confirmation=43.77%.
- YOLO11n: r(BoxAC, MaskAC)=0.994; P(box gain>0.05)=4.69%; foreground-confirmed=2.09%; conditional confirmation=44.62%.
- YOLO11s: r(BoxAC, MaskAC)=0.993; P(box gain>0.05)=5.51%; foreground-confirmed=2.37%; conditional confirmation=42.93%.

## Paper thresholds

- delta = 0.02
  - YOLOv8n: confirmation=44.76%, confirmed prevalence=3.38%.
  - YOLO11n: confirmation=45.15%, confirmed prevalence=3.42%.
  - YOLO11s: confirmation=43.66%, confirmed prevalence=3.95%.
- delta = 0.05
  - YOLOv8n: confirmation=43.77%, confirmed prevalence=2.02%.
  - YOLO11n: confirmation=44.62%, confirmed prevalence=2.09%.
  - YOLO11s: confirmation=42.93%, confirmed prevalence=2.37%.
- delta = 0.10
  - YOLOv8n: confirmation=42.95%, confirmed prevalence=1.15%.
  - YOLO11n: confirmation=43.04%, confirmed prevalence=1.16%.
  - YOLO11s: confirmation=43.78%, confirmed prevalence=1.41%.

## Outputs

- box_vs_mask_relationships.pdf: paired box/polygon density plots.
- collective_confirmation_curve.pdf: gain-threshold sensitivity.
- mask_validity_by_size.pdf: small/medium/large analysis.
- mask_validity_rates.pdf: delta=0.05 prevalence summary.
- mask_grounded_cases.pdf: qualitative examples.
- mask_grounded_validity_table.tex: paper-ready table.
- opencv_vs_exact_summary.csv: change from the raster analysis.
