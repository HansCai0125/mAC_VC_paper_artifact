# VC incremental-value experiment

Run `python experiments/vc_components.py` from the repository root.

The experiment uses the production VC evaluator and the frozen 18-run formal
matrix. It performs no training and no model inference. It tests equal-mAC box
expansion, orphan injection, and single-owner responsibility, then writes CSV,
PDF/PNG, LaTeX, protocol, and completion-audit files to
`results/vc_components`.

