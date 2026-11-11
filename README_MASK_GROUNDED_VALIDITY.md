# Third-round foreground-mask validity analysis

Run `run_vc_v1_mask_grounded_validity.py` in PyCharm with the existing
`yolo` interpreter.

The script evaluates 15 saved MS COCO runs: YOLOv8n, YOLO11n, and YOLO11s,
each with training seeds 0--4. It uses COCO instance polygons to compare
box-level AC and collective gain with foreground-mask coverage. It performs no
training and no model inference.

The run is resumable at the model/seed level. Outputs are written to:

`outputs/vc_v1/mask_grounded_validity_v1`

Primary outputs include five-seed CSV summaries, paired relationship figures,
foreground-validation rates, qualitative examples, and a LaTeX table.
