# Technical Interview Defense & Architectural Q&A Guide

This document provides rigorous, engineering-grounded answers to core design, algorithmic, and architectural questions for automotive perception and risk estimation interviews.

---

### Q1. Why XGBoost for Risk Estimation instead of a Deep Neural Network (MLP / TabNet)?
**Answer:**
1. **Tabular Feature Efficiency & Small Data Regimes:** Decision tree ensembles (XGBoost / LightGBM) consistently outperform deep architectures on heterogeneous tabular data with mixed continuous (TTC, lane offset) and ordinal/categorical features (road surface, lighting code), as established by Shwartz-Ziv & Armon (2022).
2. **Deterministic Edge Latency:** Model C evaluates in $< 0.4\text{ ms}$ on embedded CPU cores without requiring GPU tensor core acceleration, leaving GPU compute entirely free for YOLOv8 perception.
3. **Exact TreeSHAP Explainability:** TreeSHAP computes exact, polynomial-time Shapley values for XGBoost ($O(TLD^2)$), whereas deep models require sampling-based approximations (KernelSHAP) which are too slow for real-time per-frame attribution.
4. **Handling Missing Telemetry Gracefully:** XGBoost natively routes default split paths when visual markers drop out (e.g., lane markings unobserved), preventing hard inference failures.

---

### Q2. Why Canny + Hough Lines instead of Deep Lane Segmentation (e.g., Ultra-Fast-Lane-Detection / SCNN)?
**Answer:**
1. **Computational Footprint on Edge Hardware:** Canny/Hough runs in $< 2\text{ ms}$ using standard OpenCV CPU vectorized primitives, enabling $> 60\text{ fps}$ throughput on resource-constrained automotive ECUs.
2. **Explicit Geometric Interpretability:** Extrapolated linear/polynomial Hough models provide closed-form lateral offset ($d_{\text{offset}}$) and curvature ($\kappa = 1/R$) directly in metric units, rather than requiring post-hoc morphological parsing of deep probability heatmaps.
3. **Honest Trade-off & Failure Mode:** In well-marked highway corridors, Canny/Hough is exceptionally reliable; under unstructured Indian roads without lane paint, it degrades. We explicitly detect this via `lane_confidence` and gracefully downweight lane departure risk in Model C.

---

### Q3. Why YOLOv8 + ByteTrack for Multi-Object Tracking?
**Answer:**
1. **Low-Confidence Detection Association (ByteTrack Novelty):** Unlike SORT or DeepSORT which discard bounding boxes below confidence threshold $\tau_{\text{high}}$ (losing occluded/distant objects), ByteTrack performs a secondary association matching low-score detections ($0.1 \le \text{conf} < \tau_{\text{high}}$) with existing tracklets using motion similarity (Kalman filter IoU). This dramatically reduces ID switches in dense traffic.
2. **Zero Re-ID Neural Network Overhead:** ByteTrack relies purely on spatial-temporal Kalman kinematics rather than extracting deep appearance embeddings per box (as in DeepSORT), running tracking at $> 120\text{ fps}$ on CPU.
3. **Persistent ID Kinematics:** Persistent tracking IDs allow temporal differentiation of bounding box scale, enabling calculation of approach rates ($\Delta h / \Delta t$) and Time-to-Collision ($\text{TTC}$).

---

### Q4. What is your Baseline and what is genuinely novel in this project?
**Answer:**
1. **The Baseline (Model A):** A crash-only XGBoost model trained strictly on macro-level historical records (NHTSA CRSS). It acts as an actuarial prior (e.g., "high speed limit + night + wet road = elevated risk"), but has zero real-time situational awareness.
2. **The Novelty (Model C Multi-Modal Fusion):**
   - We bridge the gap between static actuarial priors and dynamic edge perception.
   - We extract continuous kinematic risk drivers ($TTC$, lateral offset, closure rate, VRU ratio) and fuse them with macro priors.
   - We provide real-time TreeSHAP local explainability for every frame, decomposing whether the risk is driven by sudden kinematics (TTC = 1.2s) vs macro conditions (unlit wet highway).

---

### Q5. What are your core physical assumptions for Monocular Distance & TTC?
**Answer:**
1. **Pinhole Camera Formulation:** $d = \frac{f \cdot H_{\text{class}}}{h_{\text{bbox}}}$.
2. **Explicit Assumptions:**
   - Calibrated vertical focal length $f$ (assumed $720\text{ px}$ for standard 720p FOV).
   - Known rigid class heights (Car $\approx 1.5\text{m}$, Truck $\approx 3.2\text{m}$, Motorbike $\approx 1.2\text{m}$, Pedestrian $\approx 1.7\text{m}$).
   - Flat roadway ground plane without extreme pitch oscillations.
3. **Closing Rate & TTC:** $\dot{d} = \frac{d_{t - \Delta t} - d_t}{\Delta t}$, $\text{TTC} = \frac{d_t}{\dot{d}}$.
4. **Engineering Defense:** While monocular depth is an approximation compared to LiDAR/Stereo, its relative rate of change ($\dot{d}$) is robust for collision mitigation and provides high-utility signals to Model C.

---

### Q6. What is the biggest limitation under Indian Road Conditions?
**Answer:**
1. **Unmarked / Potholed Roads:** Hough lines fail when painted lane boundaries are absent. Model C handles this by penalizing `lane_confidence` and relying on object distance/TTC.
2. **Heterogeneous Non-Standard Vehicles:** Auto-rickshaws, modified trucks, and overloaded tractors deviate from standard class bounding box dimensions, introducing up to $\pm 15\%$ distance estimation error.
3. **Mitigation:** Skewing depth calibration conservatively ensures the system underestimates rather than overestimates distance, ensuring safety-critical warnings fire early.
