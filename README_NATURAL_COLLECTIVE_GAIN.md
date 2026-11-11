# Natural Collective Gain

Run `run_vc_v1_natural_collective_gain.py` directly in PyCharm with the
existing `yolo` interpreter.  The script reads the completed prediction
caches for 30 paired YOLO runs and five VisDrone RT-DETR-L runs.  It does not
train a model and does not perform inference.

The primary output is:

`outputs/vc_v1/natural_collective_gain_v1`

The analysis computes AC, Best-IoGT (best ground-truth-normalized IoA), ABO,
best prediction-normalized IoA, collective gain, and deterministic marginal
contributor counts for every ground-truth box.  Model summaries are computed
from five seed-level statistics rather than by pooling seeds as independent
ground-truth observations.

The run is resumable.  A completed dataset/model/seed directory is reused only
when its analysis version and source run fingerprint match.  Set
`PREFLIGHT_ONLY = True` in the PyCharm entry point to validate the 35 source
caches without loading the prediction JSON files.
