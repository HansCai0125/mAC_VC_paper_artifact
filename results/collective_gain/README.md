# Natural Collective Gain Analysis

This report is generated from saved full-validation prediction caches.
No model training or inference is performed.

Primary confidence threshold: `0.25`

## Main outputs

- `model_summary.csv`: five-seed model summaries.
- `run_summary.csv`: one row per model seed.
- `collective_gain_exceedance.pdf`: distribution of natural gain.
- `collective_gain_breakdowns.pdf`: object-size and density breakdowns.
- `top_collective_gain_cases.csv`: candidates for qualitative review.
- `runs/.../per_gt_collective_gain.csv`: auditable per-GT values.

## Interpretation rule

`BestIoGT` is the best single ground-truth-normalized IoA. `CollectiveGain = AC - BestIoGT` isolates coverage supplied by the prediction union beyond the best single prediction.

## Five-seed groups

- MS COCO / YOLO11n: mean gain=0.008320, P(gain>0.05)=0.0469.
- MS COCO / YOLO11s: mean gain=0.009931, P(gain>0.05)=0.0551.
- MS COCO / YOLOv8n: mean gain=0.008220, P(gain>0.05)=0.0461.
- VisDrone / RT-DETR-L: mean gain=0.018908, P(gain>0.05)=0.1188.
- VisDrone / YOLO11n: mean gain=0.010098, P(gain>0.05)=0.0636.
- VisDrone / YOLO11s: mean gain=0.010397, P(gain>0.05)=0.0667.
- VisDrone / YOLOv8n: mean gain=0.010222, P(gain>0.05)=0.0652.
