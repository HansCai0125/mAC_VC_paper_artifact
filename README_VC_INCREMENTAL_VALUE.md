# VC incremental-value experiment

Run `run_vc_v1_incremental_value.py` directly in PyCharm.

The experiment uses the production VC evaluator and the frozen 18-run formal
matrix. It performs no training and no model inference. It tests equal-mAC box
expansion, orphan injection, and single-owner responsibility, then writes CSV,
PDF/PNG, LaTeX, protocol, and completion-audit files to
`outputs/vc_v1/incremental_value_v1`.

