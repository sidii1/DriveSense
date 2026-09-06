# Qualitative Audit & Known Limitations under Indian Road Conditions

> **Domain Shift Study:** Evaluating Classical/Deep Edge Perception & Risk Fusion on Unstructured Driving Environments (IDD / DriveIndia / ThirdEye contexts).

## 1. Executive Summary
Deploying ADAS perception and risk models trained on Western, highly structured highway datasets (e.g., NHTSA CRSS, KITTI, NuScenes) to Indian road conditions introduces significant domain shifts. Rather than presenting unrealistic perfection, this audit transparently characterizes where the pipeline excels, where it breaks, and how the multi-modal fusion architecture fails safely.

## 2. Qualitative Stress-Test Matrix

### 1. Unmarked Arterial & Rural Roads (No Painted Markings)
- **Perception Module Behavior:** Hough lines fail or latch onto road edges, curbs, and asphalt tar strips. Lane confidence drops below 0.20.
- **Model C Fused Risk Behavior:** Model C falls back to object tracking and TTC without asserting false lane departure warnings. Risk score increases proportionally with lane confidence penalty (+0.12).
- **Failure Mode Severity:** `Moderate (Graceful Degradation)`
- **Engineering Mitigation:** Incorporate drivable area segmentation (e.g. lightweight YOLOP or Fast-SCNN) instead of color/edge Hough lines.

### 2. High VRU Density & Lane Filtering (2-Wheelers Weaving)
- **Perception Module Behavior:** Motorcycles weave rapidly between cars. Bounding box IoU changes rapidly causing ByteTrack ID switching during severe occlusion.
- **Model C Fused Risk Behavior:** VRU ratio surges to 0.75+, triggering elevated situational risk even when TTC is temporarily benign. Captures impending cut-in hazards accurately.
- **Failure Mode Severity:** `Low to Moderate`
- **Engineering Mitigation:** Add optical flow velocity vectors to predict lateral cut-in trajectories before bounding box centroids shift.

### 3. Extreme High-Beam Headlight Glare & Monsoonal Rain
- **Perception Module Behavior:** Saturated camera sensor (>15% glare index) wipes out contrast. Rain reflections on wet asphalt create phantom Hough line candidates.
- **Model C Fused Risk Behavior:** Scene photometry flags glare and rain scatter index. Fused model increases risk baseline by +0.35 due to visual sensory degradation.
- **Failure Mode Severity:** `High (Optical Blindness)`
- **Engineering Mitigation:** Hardware-level HDR camera sensors, polarizing lens filters, and temporal contrast normalization.

### 4. Non-Standard Vehicle Geometry (Auto-Rickshaws & Overloaded Trucks)
- **Perception Module Behavior:** Pinhole distance assumes standard 1.5m car or 3.2m truck height. Auto-rickshaws (1.7m) classified as car lead to ~12% distance underestimation.
- **Model C Fused Risk Behavior:** Underestimated distance yields conservative (earlier) TTC warnings — fails safe rather than fails dangerous.
- **Failure Mode Severity:** `Low (Fails Safe)`
- **Engineering Mitigation:** Train fine-tuned YOLO on Indian driving datasets (IDD / DriveIndia / ThirdEye) with explicit 'auto_rickshaw', 'tempo', and 'tractor' classes.

## 3. Key Takeaway for System Architecture & Interviews
1. **Multi-Modal Redundancy:** When vision degrades (e.g. washed out lanes in rain), Model C does not output 0% risk; it defaults to the macro road history prior while alerting the driver of low perceptual confidence.
2. **Conservative Calibration:** Pinhole depth estimation errors on customized local vehicles skew toward underestimating distance, maintaining a positive safety margin for collision mitigation.
