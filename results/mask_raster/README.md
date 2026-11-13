# MS COCO Foreground-Mask Validity Analysis

This analysis uses saved full-validation predictions from three YOLO architectures and five training seeds. No training or model inference is performed.

`MaskAC` measures how much of the annotated COCO foreground polygon is covered by the retained same-class prediction boxes. `MaskCollectiveGain` subtracts the best single-box foreground coverage.

## Five-seed results

- YOLOv8n: r(BoxAC, MaskAC)=0.994; P(box gain>0.05)=4.61%; foreground-confirmed=2.11%; conditional confirmation=45.77%.
- YOLO11n: r(BoxAC, MaskAC)=0.994; P(box gain>0.05)=4.69%; foreground-confirmed=2.19%; conditional confirmation=46.67%.
- YOLO11s: r(BoxAC, MaskAC)=0.993; P(box gain>0.05)=5.51%; foreground-confirmed=2.46%; conditional confirmation=44.70%.

## Interpretation

Agreement supports the interpretation that box-level AC reflects genuine foreground support. Disagreement is also informative: a box annotation contains both object and background, so box coverage and foreground-mask coverage are related but intentionally non-identical quantities.

## Outputs

- `box_vs_mask_relationships.pdf`: paired box/mask density plots.
- `mask_validity_rates.pdf`: five-seed collective-case validation rates.
- `mask_grounded_cases.pdf`: auditable qualitative examples (not included: contains dataset image crops).
- `mask_grounded_validity_table.tex`: paper-ready table.
- `run_summary.csv`, `model_summary.csv`, and `runs/.../per_gt_mask_validity.csv`: numerical audit trail (the per-run files are not included).
