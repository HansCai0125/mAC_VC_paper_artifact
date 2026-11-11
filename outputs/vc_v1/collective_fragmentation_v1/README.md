# Controlled collective-fragmentation experiment

This output evaluates whether mAC and VC retain graded collective-support
information after standard one-to-one matching loses individual valid boxes.
It uses completed prediction caches only. No training or inference is run.

- Source runs: 18 (three fixed MS COCO checkpoints and fifteen VisDrone seed runs)
- Per-run rows: 162
- Aggregate rows: 54
- Fragment counts: [1, 4, 9, 16]
- Fixed-4x4 retention levels: [1.0, 0.8, 0.6, 0.4, 0.2]
- Primary theta: 0.25
- Matching IoU: 0.5
- Protocol fingerprint: `5883e4535901516f90bbc9ddd3b3171d18c8e274d87cc8765a6dcac29d2cb76c`

The experiment is a controlled metric-property test. It does not redefine
fragmented predictions as correct object instances. Diagnostic AP is computed
at the fixed theta on the transformed cached outputs and is distinct from
official dataset AP.
