# VC incremental-value experiment

Run `experiments/vc_components.py` directly in PyCharm.

The experiment uses the production VC evaluator and the frozen 18-run formal
matrix. It performs no training and no model inference. It tests equal-mAC box
expansion, orphan injection, and single-owner responsibility, then writes CSV,
PDF/PNG, LaTeX, protocol, and completion-audit files to
`results/vc_components`.

