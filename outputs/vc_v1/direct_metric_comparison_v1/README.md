# Direct Metric Comparison

All values are computed from saved full-validation predictions. No training or inference was performed.

`BestIoGT` is the closest pairwise directional-coverage baseline; `ABO` is the best pairwise IoU; `BestPredIoA` measures the reverse directional containment.

## Five-seed results

- MS COCO / YOLOv8n: P(gain>0.05)=4.61%, union-only 0.5 crossing=0.39%, AC-pass/ABO-fail=6.82%, r(AC,BestIoGT)=0.997.
- MS COCO / YOLO11n: P(gain>0.05)=4.69%, union-only 0.5 crossing=0.44%, AC-pass/ABO-fail=6.91%, r(AC,BestIoGT)=0.997.
- MS COCO / YOLO11s: P(gain>0.05)=5.51%, union-only 0.5 crossing=0.45%, AC-pass/ABO-fail=6.95%, r(AC,BestIoGT)=0.996.
- VisDrone / YOLOv8n: P(gain>0.05)=6.52%, union-only 0.5 crossing=0.42%, AC-pass/ABO-fail=4.02%, r(AC,BestIoGT)=0.996.
- VisDrone / YOLO11n: P(gain>0.05)=6.36%, union-only 0.5 crossing=0.38%, AC-pass/ABO-fail=3.95%, r(AC,BestIoGT)=0.996.
- VisDrone / YOLO11s: P(gain>0.05)=6.67%, union-only 0.5 crossing=0.40%, AC-pass/ABO-fail=4.11%, r(AC,BestIoGT)=0.996.
- VisDrone / RT-DETR-L: P(gain>0.05)=11.88%, union-only 0.5 crossing=0.55%, AC-pass/ABO-fail=5.91%, r(AC,BestIoGT)=0.991.

## Outputs

- `direct_metric_relationships.pdf`: paired ground-truth density plots.
- `paired_disagreement_rates.pdf`: five-seed disagreement rates.
- `qualitative_collective_cases.pdf`: VC-retained and VC-attenuated cases.
- `direct_metric_comparison_table.tex`: paper-ready table source.
- `per_run_comparison.csv` and `model_comparison.csv`: auditable values.
