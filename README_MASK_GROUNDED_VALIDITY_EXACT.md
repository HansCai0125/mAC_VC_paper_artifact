# Exact continuous-polygon validity analysis

Run `run_vc_v1_mask_grounded_validity_exact.py` in PyCharm with the existing
`yolo` interpreter.

This version replaces native-resolution OpenCV rasterization with continuous
COCO polygon clipping. It evaluates the same 15 saved MS COCO runs and does
not perform training or model inference.

In addition to the paired box/polygon analysis, it reports collective-gain
threshold sensitivity over 0.01--0.20, the paper thresholds 0.02, 0.05, and
0.10, and small/medium/large object breakdowns.

Outputs are written to:

`outputs/vc_v1/mask_grounded_validity_exact_v1`

The earlier OpenCV-raster outputs are preserved and an exact-versus-raster
comparison CSV is generated automatically.
