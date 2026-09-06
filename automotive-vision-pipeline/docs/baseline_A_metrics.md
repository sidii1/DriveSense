# Model A (Crash-Only Baseline) Benchmark Metrics

> **Status:** FROZEN — Comparison Anchor for Weeks 1–7

## 1. Overview
Model A is an **XGBoost Classifier** trained strictly on historical macro-level crash predictors from the NHTSA CRSS domain (speed limits, road geometry, atmospheric weather, lighting conditions, and driver/vehicle demographics). It has no access to real-time dashcam computer vision or instantaneous kinematic telemetry.

## 2. Quantitative Performance Anchor

| Metric | Model A (Crash Baseline) | Benchmark Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | `0.8800` | Baseline overall classification accuracy |
| **Precision** | `0.8201` | True positive predictive value |
| **Recall (Sensitivity)** | `0.7599` | Proportion of actual high-risk situations detected |
| **F1-Score** | `0.7889` | Harmonic mean of precision and recall |
| **ROC-AUC** | `0.9423` | Area under Receiver Operating Characteristic curve |
| **PR-AUC** | `0.8892` | Area under Precision-Recall curve |

## 3. Confusion Matrix ($N_{test} = 2400$)

```
                Predicted Safe (0)   Predicted Risky (1)
Actual Safe (0)      TN = 1574           FP = 118   
Actual Risky (1)     FN = 170            TP = 538   
```

## 4. Feature Importance Breakdown

| Rank | Feature | Importance Weight |
| :--- | :--- | :--- |
| 1 | `driver_impairment_flag` | `0.2902` (29.0%) |
| 2 | `light_condition` | `0.1314` (13.1%) |
| 3 | `road_surface_condition` | `0.1218` (12.2%) |
| 4 | `weather_condition` | `0.1075` (10.7%) |
| 5 | `vehicle_type` | `0.0932` (9.3%) |
| 6 | `road_alignment` | `0.0923` (9.2%) |
| 7 | `speed_limit_mph` | `0.0748` (7.5%) |
| 8 | `driver_age_group` | `0.0575` (5.8%) |
| 9 | `historical_segment_risk` | `0.0313` (3.1%) |

## 5. Key Baseline Limitations (Why Vision Fusion is Required)
1. **Zero Real-Time Awareness:** Cannot detect if a vehicle is braking sharply ahead, drifting out of its lane, or if an unhelmeted motorcycle is cutting in.
2. **Static Prior Bias:** Assigns identical risk to two cars on the same highway at night, even if one is driving on an empty open road and the other is tailgating at 0.5s TTC.
3. **Missed Near-Miss Incidents:** High false-negative rate for sudden, dynamic hazards.
