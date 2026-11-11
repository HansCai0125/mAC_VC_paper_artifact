# Collective Fragmentation Experiment Protocol

## Purpose

This experiment tests the specific information supplied by mAC and VC when
one-to-one instance matching no longer distinguishes different degrees of
collective regional support. It is a controlled metric-property experiment,
not a claim that fragmented boxes are correct instance detections.

The predeclared hypothesis is:

> Once every individual fragment falls below the one-to-one IoU matching
> threshold, AP/F1/LRP identify the instance-detection failure but largely
> lose the ability to order different levels of remaining collective support.
> mAC and VC should retain a graded response to that support.

## Frozen experimental scope

- Source: the completed post-NMS formal prediction caches.
- Datasets: MS COCO 2017 validation and VisDrone2019-DET validation.
- Models: YOLOv8n, YOLO11n, and YOLO11s.
- Statistical units:
  - MS COCO: the same predeclared fixed checkpoint per architecture.
  - VisDrone: five training seeds per architecture.
- Confidence threshold: `theta = 0.25`.
- One-to-one matching IoU: `0.50`.
- Training: none.
- Model inference: none.
- All transformations are deterministic and applied offline.

## Frozen target selection

For every source run, freeze the baseline one-to-one true-positive pairs at
`theta = 0.25` and IoU `0.50`. Each target prediction has one matched
ground-truth owner.

For matched pair `(p_i, g_i)`, define the controlled support box

\[
S_i = p_i \cap g_i.
\]

Because the original pair is a true positive, `S_i` has positive area and
its area relative to `g_i` is at least the matching IoU. Replacing `p_i` by
`S_i` removes over-extension outside the owner GT before fragmentation. This
isolates collective support from the large-box and orphan-prediction effects
already studied in the controlled stress tests.

The `q=1` transformed document is the reference condition for this
experiment. Non-target predictions and all ground-truth boxes remain
unchanged in every condition.

## Experiment A: representation fragmentation

Partition every support box `S_i` into a regular `q x q` grid:

\[
q \in \{1,2,3,4\}, \qquad K=q^2\in\{1,4,9,16\}.
\]

Replace the parent target by the `K` non-overlapping tiles. Every tile
inherits the parent's class and confidence. At every `q`, the tile union is
exactly `S_i` up to floating-point tolerance.

This experiment changes representation granularity while holding collective
support fixed.

### Predeclared expectation

- mAC should remain invariant because the covered GT area is unchanged.
- VC should remain invariant because the assigned union and its spatial
  precision are unchanged.
- At `q >= 2`, each individual tile has insufficient IoU for a standard
  one-to-one match, so diagnostic mAP/F1/1-LRP should deteriorate sharply.
- Any departure of mAC/VC from invariance must be audited for ownership
  changes or numerical union errors rather than removed from the results.

## Experiment B: coverage-retention discrimination

Fix `q=4` (`K=16`). Within every grid cell, shrink its tile about the cell
centre by side factor `sqrt(rho)`, using

\[
\rho \in \{1.0,0.8,0.6,0.4,0.2\}.
\]

The fragment count, class labels, confidences, and spatial distribution are
fixed. The total retained tile area is `rho * |S_i|`, with evenly distributed
gaps between fragments. Thus only collective support completeness changes.

### Predeclared expectation

- Diagnostic mAP/F1/1-LRP should already be in their fragmented regime and
  should change little across `rho` because no tile reaches IoU `0.50`.
- mAC should decrease monotonically with retained GT coverage.
- VC should decrease monotonically with retained support and should reflect
  the Dice balance between the retained union and the GT region.
- The central evidence is separation among `rho` conditions after the
  conventional one-to-one metrics have saturated.

## Metrics and reporting

At every condition report:

- diagnostic mAP50:95 at fixed `theta = 0.25`;
- fixed-threshold F1 and LRP;
- mAC;
- VC;
- prediction count, assigned count, orphan count, and ownership-change rate.

All response curves are normalized by the `q=1, rho=1` controlled reference
within the same source run. Raw scores must also be saved.

MS COCO is reported as a fixed-checkpoint point estimate. VisDrone is reported
as mean plus sample standard deviation over five seeds. The main plot may use
an equal-weight descriptive mean over the six dataset-model settings, but the
per-setting results and VisDrone seed standard deviations must remain
available in the result tables.

## Required audit checks

For every run and condition, record and verify:

1. the frozen target-pair count;
2. the expected number of generated fragments;
3. finite, clipped, positive-area fragment coordinates;
4. exact union-area conservation for Experiment A;
5. retained-area error relative to the declared `rho` for Experiment B;
6. the number and fraction of fragments whose VC owner differs from the
   frozen baseline owner;
7. no change to GTs or non-target predictions;
8. no training or model inference;
9. complete resumable provenance for all 18 formal source runs.

## Predeclared success and falsification criteria

The result supports the collective-coverage claim only if all of the
following hold across both datasets and all three architectures:

1. At full support (`rho=1`), increasing `K` leaves mAC and VC approximately
   invariant while conventional one-to-one metrics deteriorate.
2. At fixed `K=16`, mAC and VC provide a monotonic ordering of the five
   `rho` levels.
3. At fixed `K=16`, the conventional metrics show materially less separation
   among `rho` levels than mAC and VC.
4. The result is not explained by ownership changes, clipping errors, or
   incomplete source caches.

If these conditions fail, the paper must not claim that mAC/VC retain useful
collective-support information after one-to-one matching saturates.

No post-hoc removal of models, datasets, seeds, targets, or severity levels is
allowed because their response is unfavourable.

## Planned figure

The main collective-coverage figure will contain three panels:

1. **Controlled construction:** one support box, a complete `4 x 4`
   partition, and a gapped `4 x 4` partition.
2. **Fragmentation response:** normalized scores versus
   `K = 1, 4, 9, 16` at full union support.
3. **Coverage-retention response:** normalized scores versus
   `rho = 0.2, 0.4, 0.6, 0.8, 1.0` at `K=16`.

The main figure will show diagnostic mAP50:95, mAC, VC, and F1. LRP and the
full per-setting curves will be reported in a table or supplementary figure.

## Intended interpretation

The experiment does not redefine fragmented detections as correct object
instances. AP/F1/LRP remain the appropriate measures of instance correctness.
The experiment asks an additional question: how much of an annotated region
is collectively supported when no individual prediction satisfies the
instance-matching rule?

The intended evidence chain is:

1. mAC retains a graded measure of collective coverage;
2. VC supplements that coverage with spatial-precision and orphan penalties;
3. AP/F1/LRP retain instance-level matching and duplicate sensitivity;
4. joint reporting exposes information that no single metric supplies.

Claims about weak, coarse, or group-level annotations must be presented as
the target evaluation setting of this controlled experiment. MS COCO and
VisDrone remain standard instance-level benchmarks and must not be described
as direct validation datasets for naturally occurring group-level labels.
