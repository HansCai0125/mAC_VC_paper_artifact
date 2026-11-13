# Continuous diagnostic mAP anti-gaming curves

The formal anti-gaming experiment intentionally calculated diagnostic AP at
only a few representative attack levels. That is why the existing mAP points
cannot be connected without adding measurements. The prediction caches are
complete; the missing values can be calculated offline without retraining or
model inference.

## PyCharm run

Open and run:

`experiments/perturbations.py`

Keep `PREFLIGHT_ONLY = False` for the actual calculation. Set it to `True`
only for a quick input-integrity check. The calculation is resumable at the
dataset/model/seed level and writes to:

`results/perturbations`

It does not overwrite `gaming_main`.

The entry point writes the result tables. The historical standalone plotting
script for these curves is not included in this repository.

The connected mAP curve is diagnostic mAP50:95 evaluated at the fixed
confidence threshold after each controlled manipulation. It should be
described separately from official dataset mAP.
