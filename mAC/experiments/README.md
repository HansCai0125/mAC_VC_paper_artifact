README — mAC Evaluation Experiments

This file reproduces the experiments reported in “Beyond mAP: A Coverage–Efficiency–Aware Evaluation Framework for Real-Time Object Detection Systems”.
It compares the proposed Mean Area Coverage (mAC) metric with traditional mAP/IoU-based metrics and evaluates their correlation with Visual Completeness (VC) under weakly annotated and small-object scenarios.


1. Objective

	Quantify how closely mAC aligns with human-perceived visual completeness.
	Validate that mAC shows **12–18 % higher correlation** with VC than mAP/IoU metrics.
	Demonstrate that the combined **Efficiency–Precision Index (EPI)** improves interpretability by balancing accuracy and efficiency.

2. Datasets

    Weed Dataset (Exsiting)

    Remote Sensing Dataset (Not found yet)

3. Models

Evaluate 3 YOLO variants 

	YOLOv8, YOLOv11 n, YOLOv11s

Each model is trained on the same training/validation split with identical random seed and image size settings.


4. Metrics and Formulas

    mAC mAP EPI

5.  Experimental Procedure

       Step 1: Environment and package setup
    Step 2: Train and Predict
    Step 3: Compute AC mAC mAP----Produces per-GT coverage values and overall mAC score.
    Step 4: Collect Visual Completeness and EPI Evaluation


6.  Expected Results

       Metric: mAP@0.5,mAC@0.5,mAC@0.7,EPI
    Calculation: Each one Corr with Dataset1,2 and compare

→ Average ??% higher correlation with VC than mAP/IoU.


7.  Interpretability and Visualization

        Scatter plots: Metric score vs VC rating (Regression lines + R²).
        Ranking Heatmap**: Model ranking similarity between metric and VC
        Qualitative Examples: mAP fails (0.49 IoU) but mAC detects (AC@50 = 0.68) and High mAC but excessive boxes → EPI penalization.

8.  Statistical Validation
 	
	To be discussed


9.   Implementation Notes
	
    Randon Seed(42)