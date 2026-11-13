# Natural Collective Gain

Run `python experiments/collective_gain.py` from the repository
root.  The script reads the completed prediction
caches for 30 paired YOLO runs and five VisDrone RT-DETR-L runs.  It does not
train a model and does not perform inference.

The primary output is:

`results/collective_gain`

The analysis computes AC, Best-IoGT (best ground-truth-normalized IoA), ABO,
best prediction-normalized IoA, collective gain, and deterministic marginal
contributor counts for every ground-truth box.  Model summaries are computed
from five seed-level statistics rather than by pooling seeds as independent
ground-truth observations.

The run is resumable.  A completed dataset/model/seed directory is reused only
when its analysis version and source run fingerprint match.  Set
`PREFLIGHT_ONLY = True` in the entry point to validate the 35 source
caches without loading the prediction JSON files.
