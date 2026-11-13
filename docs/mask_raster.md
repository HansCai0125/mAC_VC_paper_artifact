# Third-round foreground-mask validity analysis

Run `python experiments/mask_raster.py` from the repository root.

The script evaluates 15 saved MS COCO runs: YOLOv8n, YOLO11n, and YOLO11s,
each with training seeds 0--4. It uses COCO instance polygons to compare
box-level AC and collective gain with foreground-mask coverage. It performs no
training and no model inference.

The run is resumable at the model/seed level. Outputs are written to:

`results/mask_raster`

Primary outputs include five-seed CSV summaries, paired relationship figures,
foreground-validation rates, qualitative examples, and a LaTeX table.
