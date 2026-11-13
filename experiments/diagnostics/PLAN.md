# Frozen analysis specification v1

Source: the authors' frozen supplementary-experiment plan. This implementation adds evidence from existing predictions; it does not change mAC/VC definitions or claim a successful result in advance.

## Sources and populations

- A: the paired YOLO cache registry, 3 architectures × 5 seeds × 2 datasets. COCO contains 5,000 images/run, VisDrone 548 images/run. RT-DETR is outside these paired comparisons; this does not remove it from the manuscript's prior results.
- B: eligible non-crowd COCO polygon targets from the existing continuous polygon audit. A target with no usable polygon is retained in A's box denominator and excluded from mask endpoints.
- C: the 4,000 held-out images from the previously frozen SAM split. The 1,000 calibration images do not enter this analysis. The confidence threshold and global prompt budget are fixed at 0.25 and 20, not selected again.
- D2: all retained non-crowd COCO targets, including singleton groups and all available segmentation encodings. Uses pycocotools masks. The original crowd filtering policy remains part of the source protocol.
- Inputs are read-only. Prediction checksums are verified. Source SAM annotation and prediction hashes must agree with the registered sources. Input sizes/timestamps are checked again at completion. The SAM SQLite identity uses size/mtime plus source protocol linkage, not a claimed full database content hash.

## A: residual support among unmatched targets

Confidence filtering precedes score-ordered, class-consistent greedy one-to-one matching at IoU >= 0.5. Ties use intersection area then GT ID, matching the existing evaluator. K counts positive-area, same-class intersections, not all predictions in the image.

A1: fraction of FN targets with K >= 2. A2: among mask-eligible FN with K >= 2, report (i) MaskCollectiveGain > .05, (ii) MaskAC >= .5 and MaskBestCoverage < .5, (iii) MaskAC >= .5. The best mask coverage is maximized independently from the box-best prediction. All denominators and excluded polygon targets remain visible.

A3 primary: within prespecified strata, compare mean MaskCollectiveGain between high and low box-CollectiveGain terciles. Strata are BestIoGT [0,.3), [.3,.5), [.5,.7), [.7,.9), K=2/3–4/5+, class, and COCO size bin. BestIoGT >= .9 is outside this primary contrast because of the coverage ceiling; target-level output retains these targets.

Cutoffs are pooled across five seeds within each model/stratum, using only box gain. Low <= lower tercile; high >= upper tercile. A stratum requires >=60 distinct (image,GT) targets and >=20 distinct images. Equal tercile cutoffs are excluded, never split by arbitrary row order. No outcome-dependent stratum merging. Fixed weights equal eligible unique-target counts, normalized across strata. Bootstrap holds strata, weights and cutoffs fixed. A seed contrast is unavailable if any retained stratum lacks that seed's high or low group. This is a controlled descriptive association, not a causal effect or complete confound removal.

## B: added-region foreground enrichment

Clip retained same-class rectangles to GT. Select best B by intersection area, score then prediction ID. Define E=(union of clipped rectangles) minus B. Exclude K<2, no remaining GT area, or no positive E. Numerical area tolerance is max(1, GT area)*1e-9.

Continuous foreground areas reuse the historical polygon-clipping convention and polygon-part representation from the previous audit. This preserves comparability with its MaskAC values. Impossible area bounds are separately excluded and logged, not silently clipped into plausible densities.

B1: ratio of summed observed foreground in E to summed area-based expectation. The target expectation is |E| times remaining foreground density in GT\B. Targets with effectively zero expectation are separately counted and excluded from the primary ratio; an all-target ratio is also reported.

B2: keep B fixed. Independently translate every other clipped rectangle uniformly within GT while retaining its width and height. Form E' from their new union minus B. Use 200 draws per target; random seeds are deterministic from experiment/run/image/GT. Ratio of summed observed foreground to summed mean foreground under translation is the primary shift enrichment. Zero-expectation exclusions and all-target sensitivity are reported.

IMPORTANT: translation preserves individual rectangle sizes but not added union area. The report therefore also provides observed/expected added areas, density references, zero-area draw counts, and per-target Monte Carlo standard errors. R_shift is not an equal-added-area experiment, nor a claim of semantic understanding.

B3: target foreground-density excess relative to the area reference; report mean with image-cluster interval and median, overall and by size. B uses all eligible targets; FN status is retained for later descriptive examination.

## C: incremental VC information

Unit: image × class × detector seed. Outcomes are cached SAM foreground union IoU (primary) and precision (secondary), with the original class-specific ignored-pixel policy. Exclude both-empty class unions. If exactly one side is empty, IoU=0; precision is defined as 0 for no predicted pixels. Record exclusions.

Features are recomputed from the exact globally retained top-20 boxes and the same GT/filtering protocol. Within a class, VC=sum(local owned-union Dice)/(GT count+orphan count). Empty-class feature conventions are zero except LRP=1 when predictions or GT are present but unmatched. Union Dice compares unions of all class GT boxes and prediction boxes.

OLS models:

- Compact: union Dice + prompt count.
- Extended: compact + mAC + F1 + LRP + confidence mean/max/population SD + BestIoGT.
- Extended+VC: extended + VC.

For each architecture, pool its five seeds. Five image-group folds, repeated ten times. The image-to-fold map is identical across architectures and seeds. All classes and seeds of an image stay together. Standardize features using training rows only; add intercept; fit numpy least squares with no tuning or outcome clipping. Export matrix rank for collinearity audit.

Primary C1 is extended+VC minus extended held-out R². Compute mean squared error across repetitions; do not evaluate an ensemble formed by averaging repeated predictions. Bootstrap reweights image-level sufficient statistics of these fixed out-of-fold losses. This interval is conditional on fitted folds; it does not include CV-refitting, detector retraining or calibration uncertainty. The same images have participated in previous research analyses, so this is a supplementary validation analysis, not a newly untouched test set.

Mechanism checks: per-GT SAM foreground recall from the entire same-class SAM union is ownership independent. Compare OOF regressions on AC/ABO/BestIoGT with and without local VC. Also report owned-mask IoU using max-IoU and max-intersection ownership. These latter diagnostics depend on allocation and are not independent validation of it. For per-GT owned-mask IoU, other objects count outside that target and there is no additional crowd-pixel exclusion; only the primary image-class outcome uses the frozen SAM ignore policy. Mechanism outputs are descriptive.

## D2: controlled annotation coarsening

Construct a same-class GT graph. Expand each box by delta*sqrt(box area) on each side, clip to image, and connect boxes whose expanded supports intersect or touch. delta=0 is primary; .25 and .5 are sensitivity conditions. A connected component is represented by the enclosing box of its original GT boxes and the union of original masks. Singletons stay in the evaluator. No prediction-dependent grouping.

Report component sizes, incomplete-clique/transitive-chain frequency, foreground occupancy, outside-enclosing-box mask pixels, mean signed and absolute box-versus-mask coverage gaps. Continuous box AC is retained, but the primary box-mask contrast uses the same COCO raster convention on both sides. Report continuous-versus-raster box discrepancy. Recompute hard ownership against the complete grouped target set. Aggregate VC with each image's orphan count once.

D2 is a controlled annotation transformation, not evidence from a naturally coarse-annotated deployment dataset. D1 is gated because registered detector caches already exclude crowd-overlapping predictions; unfiltered outputs would be needed for a new natural-crowd claim.

## Statistics, reporting and stopping

Formal bootstrap B=2,000, resampling images, keeping seed/target/class observations together. For A/B, use all 5,000 COCO images, including images with zero eligible targets. Identical RNG streams provide paired resample weights across models. C uses the common 4,000-image held-out universe. Sample SD across five seeds (ddof=1) is separate from image-cluster uncertainty; neither establishes checkpoint independence by itself.

Prespecified primary family: A3 (>0), B1 (>1), B2 (>1), C1 IoU ΔR² (>0), each for three architectures: 12 tests. One-sided recentered-bootstrap approximate p-values; Holm correction over all 12, retaining unestimable tests as p=1 in the correction. Percentile intervals are pointwise and conditional, not simultaneous confidence bands. Secondary endpoint p-values are not confirmatory evidence. A missing contrast is reported as missing. No favorable-result stopping, no automatic threshold search.

All target-level results, exclusions and seed directions are exported. Small-sample smoke mode is clearly marked and cannot supply paper values. Final scientific interpretation awaits the user's complete run. No manuscript rewrite or new performance claim is generated automatically.
