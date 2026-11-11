# Collective fragmentation experiment

This experiment tests whether mAC and VC retain graded information about
collective regional support after one-to-one detection metrics enter a
fragmented matching-failure regime.

It reads the completed formal prediction caches. It does not train a model,
load a checkpoint, or run model inference.

## Run in PyCharm

Open and run:

`E:\Project\mAC\run_vc_v1_collective_fragmentation.py`

The normal setting is:

```python
PREFLIGHT_ONLY = False
```

Set it to `True` only for a fast cache and protocol check.

## Output

Results are written to:

`E:\Project\mAC\outputs\vc_v1\collective_fragmentation_v1`

The experiment is resumable per dataset/model/seed run. Rerunning the same
entry point skips completed runs that match the frozen protocol.

The final directory contains:

- `fragmentation_runs.csv`: all 162 source-run rows;
- `fragmentation_aggregate.csv`: 54 dataset/model aggregate rows;
- `protocol.json`: frozen transformation and evaluation protocol;
- `collective_fragmentation_complete.json`: completion and file hashes;
- `runs/`: per-run metrics, geometry/ownership audit, and provenance.

## Conditions

- Full-union fragmentation: `1`, `4`, `9`, and `16` fragments.
- Fixed-4x4 coverage retention: `1.0`, `0.8`, `0.6`, `0.4`, and `0.2`.
- Primary confidence threshold: `theta=0.25`.
- Matching IoU: `0.50`.
- Diagnostic mAP50:95 is recomputed at every condition on the controlled
  cached output. It is distinct from official dataset AP.

The transformation starts from the intersection of each frozen baseline TP
and its matched GT. All fragments therefore remain inside the controlled
support. The output records exact retained-area error, fragment count,
ownership changes caused by overlapping same-class GTs, and orphan counts.

## Runtime

The work is CPU-heavy because diagnostic AP is recomputed for eight unique
conditions in each of 18 source runs. A reasonable estimate on the current
machine is roughly one to two hours. Interruptions are safe between completed
source runs; rerun the same file to resume.

The frozen scientific design and success criteria are documented in:

`E:\Project\mAC\COLLECTIVE_FRAGMENTATION_PROTOCOL.md`
