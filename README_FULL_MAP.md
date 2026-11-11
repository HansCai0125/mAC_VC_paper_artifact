# Continuous diagnostic mAP anti-gaming curves

The formal anti-gaming experiment intentionally calculated diagnostic AP at
only a few representative attack levels. That is why the existing mAP points
cannot be connected without adding measurements. The prediction caches are
complete; the missing values can be calculated offline without retraining or
model inference.

## PyCharm run

Open and run:

`E:\Project\mAC\run_vc_v1_gaming_full_ap.py`

Keep `PREFLIGHT_ONLY = False` for the actual calculation. Set it to `True`
only for a quick input-integrity check. The calculation is resumable at the
dataset/model/seed level and writes to:

`E:\Project\mAC\outputs\vc_v1\gaming_full_ap_v1`

It does not overwrite `gaming_formal_v1`.

After the calculation finishes, the same entry point automatically creates
the connected figures. The figure script can also be rerun by itself:

`E:\Project\mAC\run_vc_v1_gaming_full_map_figures.py`

The connected response figure and its anchor heatmap are written to:

`E:\Project\mAC\outputs\vc_v1\paper_gaming_v5`

The connected mAP curve is diagnostic mAP50:95 evaluated at the fixed
confidence threshold after each controlled manipulation. It should be
described separately from official dataset mAP.
