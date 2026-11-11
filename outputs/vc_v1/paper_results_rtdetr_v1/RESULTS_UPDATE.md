# Four-model VC v1 result update

## Experimental setup draft

Experiments were conducted on MS COCO 2017 and VisDrone2019-DET using their official training and validation splits. All detectors were trained on the corresponding training sets and evaluated on the complete validation sets, comprising 5,000 images for MS COCO and 548 images for VisDrone. Four detector architectures were considered: YOLOv8n, YOLO11n, YOLO11s, and RT-DETR-L. For MS COCO, uncertainty for each reported trained model was estimated using 10,000 image-level bootstrap replicates over the complete validation set. For VisDrone, each architecture was trained with five independent random seeds, and the results are reported as the mean and sample standard deviation. VC, mAC, F1, and LRP were evaluated at the common primary confidence threshold $\theta=0.25$ and matching IoU 0.5. To examine operating-point sensitivity, all metrics were additionally reported at 19 confidence thresholds from 0.05 to 0.95. YOLO uses its standard post-NMS detections, whereas RT-DETR uses its native set-based postprocessing.

## RT-DETR-L primary results

On MS COCO, RT-DETR-L obtained VC 0.3989 [0.3914, 0.4064], mAC 0.7068 [0.6966, 0.7168], F1 0.5014 [0.4948, 0.5081], and LRP 0.7707 [0.7662, 0.7751]. On VisDrone, its five-seed results were VC 0.4605 ± 0.0024, mAC 0.7009 ± 0.0031, F1 0.6129 ± 0.0026, and LRP 0.7447 ± 0.0021. The VisDrone VC mean and sample standard deviation were 0.4605 and 0.0024, respectively.

The VC-maximizing RT-DETR-L threshold was 0.45 on MS COCO and 0.40 on VisDrone. At the common $\theta=0.25$ operating point, the relative gaps from the per-dataset VC maxima were 17.06% and 6.97%. This shift toward higher thresholds shows that confidence calibration differs across detector families. We therefore retain $\theta=0.25$ only as the predeclared common operating point for the primary comparison, without claiming that it is optimal for every architecture, and report the complete curves and model-specific optima as sensitivity results.

## Reporting notes

- `table_main_performance_four_models.csv` is the updated eight-row primary table.
- `table_theta_optima_four_models.csv` and the corresponding figure include all four architectures on both datasets.
- MS COCO uncertainty is estimated by image-level bootstrap over the complete validation set for each reported trained model.
- VisDrone uncertainty reflects variation across five independent training seeds and is reported as the sample standard deviation.
- `diagnostic_AP50_95` is computed from the shared prediction cache and is retained for audit only. Use official dataset-evaluator AP as the paper's standard AP result.
- The existing controlled Gaming analysis remains a separate three-YOLO-architecture experiment until RT-DETR Gaming is explicitly added.
