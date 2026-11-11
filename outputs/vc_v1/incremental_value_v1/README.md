# VC incremental-value experiment

This controlled suite isolates what VC contributes beyond collective coverage. All canonical cases are scored by the production VC v1 implementation. The real-data orphan curves are exact projections from the frozen 18-run formal matrix.

No training or model inference is performed.

## Key anchors

- Equal-mAC expansion at s=1.41: mAC=1.000, VC=0.669, F1=1.000.
- Four added orphans with a fixed perfect union: mAC=1.000, local VC=1.000, dataset VC=0.200.
- One prediction shared by two adjacent targets: mAC=1.000, VC=0.333, F1=0.667.

## Interpretation

mAC supplies a coverage signal. VC does not inherit validity from mAC; it adds spatial specificity, single-owner responsibility, and an orphan-aware dataset denominator. Exact duplicates remain a known union-metric invariance and must be exposed by the reported instance metrics.

## Outputs

- vc_incremental_value_response.pdf: canonical three-property figure.
- real_orphan_projection.pdf: exact projection over 18 formal runs.
- vc_incremental_value_table.tex: compact paper table.
- metric_property_matrix.tex: metric-role summary.
- canonical_response.csv: auditable controlled values.
- orphan_projection_by_run.csv and orphan_projection_by_model.csv: real-run projections.

Protocol fingerprint: `2dae762a3b0a807fbd6daaa50a498850d53e8e6fc56b98a7029cf2ed6b725687`

Real projection rows: 90
