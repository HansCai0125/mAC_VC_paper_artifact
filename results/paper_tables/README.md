# Paper tables

`table5_detection_results.csv` holds the values printed in paper Table 5: mean and sample SD over five archived runs per setting, at theta = 0.25. Rebuild it offline with `python scripts/build_table5.py`.

## RT-DETR extension (earlier draft)

The remaining files are an offline merge of the completed RT-DETR-L suite with the YOLO results. No training or inference is performed.

* `rtdetr_aggregate_*.csv` and `rtdetr_run_*.csv` are the RT-DETR-L sources. The five-run VisDrone row feeds Table 5.
* `theta_sensitivity_four_models.csv` and `table_theta_optima_four_models.csv` hold the threshold-sensitivity data.
* `table_main_performance_four_models.csv`, `coco_rtdetr_bootstrap.csv` and `RESULTS_UPDATE.md` are kept from an earlier draft. Their MS-COCO rows use one fixed checkpoint per model with image-bootstrap intervals and include an MS-COCO RT-DETR-L row. The paper reports five-run means for the YOLO models and does not report RT-DETR-L on MS-COCO, so these files do not match the printed Table 5.
