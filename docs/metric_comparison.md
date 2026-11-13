# Direct Metric Comparison

Run `experiments/compare_metrics.py` directly in PyCharm with the
existing `yolo` interpreter. The script reads the completed first-round
per-ground-truth results and saved prediction caches. It performs no training
and no model inference.

The output directory is:

`results/metric_comparison`

The analysis directly compares union Area Coverage (AC) with Best-IoGT, ABO,
and reverse prediction-normalized IoA. It reports five-seed disagreement
rates, paired density plots, a paper-ready LaTeX table, and four qualitative
cases showing both VC-retained and VC-attenuated collective coverage.

Keep `PREFLIGHT_ONLY = False` for the normal PyCharm run. Set it to `True` only
to validate all inputs without performing the comparison.
