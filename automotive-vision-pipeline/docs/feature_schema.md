# Fused Multi-Modal Feature Schema Specification

## 1. Executive Summary & Research Goal

**Core Research Question:**
> *"Does adding real-time visual/environmental features to a crash-history model improve context-aware risk estimation?"*

To evaluate this question systematically without data leakage or moving targets, this document establishes the formal, frozen feature schema across three modeling paradigms:
1. **Model A (Crash-Only Baseline):** Tabular NHTSA environmental, roadway, and demographic attributes.
2. **Model B (Vision-Only Model):** Real-time kinematic, optical, and perceptual features extracted directly from forward dashcam video.
3. **Model C (Multi-Modal Fusion):** Concatenated, context-aware feature space combining historical priors with instant perceptual telemetry.

---

## 2. Tabular Feature Schema (Model A: Crash Baseline)

These features represent the static or macro-level priors typically available from insurance telematics, municipal DOT records, and NHTSA Crash Report Sampling System (CRSS) datasets.

| Feature Name | Data Type | Permissible Range / Enum | Physical Unit | Description & Justification |
| :--- | :--- | :--- | :--- | :--- |
| `speed_limit_mph` | `float` | $[15.0, 85.0]$ | Miles/hour | Posted roadway speed limit governing baseline kinetic energy risk. |
| `road_alignment` | `int` | `0`: Straight<br>`1`: Curve | Categorical (Binary) | Curvature geometry of the road segment (static GIS / map prior). |
| `road_surface_condition` | `int` | `0`: Dry<br>`1`: Wet / Ice / Snow | Categorical (Binary) | Surface friction coefficient prior. |
| `light_condition` | `int` | `0`: Daylight<br>`1`: Dark (Lighted)<br>`2`: Dark (Unlit) | Categorical (Ordinal) | Ambient time-of-day lighting infrastructure classification. |
| `weather_condition` | `int` | `0`: Clear<br>`1`: Rain / Fog / Mist<br>`2`: Severe Storm / Snow | Categorical (Ordinal) | Atmospheric weather state prior from regional meteorological feeds. |
| `driver_age_group` | `int` | `0`: Young ($<25$)<br>`1`: Middle ($25-64$)<br>`2`: Senior ($65+$) | Categorical (Ordinal) | Demographic actuarial risk bracket. |
| `driver_impairment_flag` | `int` | `0`: Normal<br>`1`: Impaired / Distracted | Categorical (Binary) | Behavioral risk indicator. |
| `vehicle_type` | `int` | `0`: Passenger Car<br>`1`: SUV / Truck<br>`2`: Motorcycle | Categorical (Nominal) | Vehicle mass and passive safety rating class. |
| `historical_segment_risk` | `float` | $[0.0, 1.0]$ | Probability $[0, 1]$ | Empirical historical crash incidence rate per million vehicle-miles traveled. |

---

## 3. Real-Time Vision & Kinematic Feature Schema (Model B)

These features are computed on-the-fly at $30\text{ fps}$ by the edge perception pipeline combining Canny/Hough lane geometry, YOLOv8 + ByteTrack object tracking, monocular pinhole depth/TTC estimation, and OpenCV scene photometry.

| Feature Name | Data Type | Valid Range | Physical Unit | Extraction Technique & Mathematical Definition |
| :--- | :--- | :--- | :--- | :--- |
| `lane_offset_m` | `float` | $[-2.5, +2.5]$ | Meters ($\text{m}$) | Lateral displacement of camera center from lane center ($d_{\text{center}} - d_{\text{ego}}$). Negative = left bias, Positive = right bias. |
| `lane_curvature_rad_m` | `float` | $[-0.05, +0.05]$ | $\text{m}^{-1}$ ($1/R$) | Inverse radius of lane curvature from 2nd-order polynomial fit ($x = ay^2 + by + c \implies \kappa = 2a$). |
| `lane_confidence` | `float` | $[0.0, 1.0]$ | Confidence $[0, 1]$ | Composite score of Hough line support, segment continuity, and left/right lane parallelism. |
| `min_ttc_sec` | `float` | $[0.1, 10.0]$ | Seconds ($\text{s}$) | $\text{TTC} = \frac{d_{\text{lead}}}{\dot{d}_{\text{closing}}}$, computed for the nearest relevant forward vehicle/obstacle. |
| `nearest_obj_distance_m`| `float` | $[1.0, 100.0]$ | Meters ($\text{m}$) | Monocular pinhole depth: $d = \frac{f \cdot H_{\text{class}}}{h_{\text{bbox}}}$ using class-specific assumed heights. |
| `max_approach_rate_mps` | `float` | $[-20.0, +40.0]$ | $\text{m}/\text{s}$ | Maximum closing velocity $(\Delta d / \Delta t)$ across active tracks. Positive = rapidly closing in. |
| `traffic_density` | `int` | $[0, 50]$ | Count / Frame | Total number of active persistent tracked bounding boxes in forward FOV. |
| `vru_ratio` | `float` | $[0.0, 1.0]$ | Ratio $[0, 1]$ | Vulnerable Road User proportion: $\frac{N_{\text{motorcycle}} + N_{\text{pedestrian}} + N_{\text{bicycle}}}{N_{\text{total}} + \epsilon}$. |
| `scene_luminance_mean` | `float` | $[0.0, 255.0]$ | 8-bit Luma | Spatial average grayscale channel intensity: $\frac{1}{HW}\sum I(x, y)$. |
| `scene_contrast_rms` | `float` | $[0.0, 128.0]$ | RMS Intensity | Root-mean-square contrast: $\sqrt{\frac{1}{HW}\sum (I(x,y) - \mu)^2}$. |
| `glare_index` | `float` | $[0.0, 1.0]$ | Fraction $[0, 1]$ | Ratio of saturated over-exposed pixels ($I(x,y) \ge 245$) indicating direct headlight/sun glare. |
| `rain_fog_scatter_index`| `float` | $[0.0, 1.0]$ | Normalized Index | Ratio of attenuated high-frequency Laplacian gradient energy indicating fog/rain haze scattering. |

---

## 4. Fused Multi-Modal Feature Schema (Model C)

Model C takes the unified concatenated vector of dimension $D = 9 + 12 = 21$:

$$\mathbf{x}_{\text{fused}} = \left[ \mathbf{x}_{\text{tabular}}^{(9)} \,\|\, \mathbf{x}_{\text{vision}}^{(12)} \right] \in \mathbb{R}^{21}$$

### Vector Element Ordering:
```python
FUSED_FEATURE_VECTOR = [
    # --- Tabular NHTSA Priors (Index 0-8) ---
    0: "speed_limit_mph",
    1: "road_alignment",
    2: "road_surface_condition",
    3: "light_condition",
    4: "weather_condition",
    5: "driver_age_group",
    6: "driver_impairment_flag",
    7: "vehicle_type",
    8: "historical_segment_risk",

    # --- Real-Time Vision & Telemetry (Index 9-20) ---
    9: "lane_offset_m",
    10: "lane_curvature_rad_m",
    11: "lane_confidence",
    12: "min_ttc_sec",
    13: "nearest_obj_distance_m",
    14: "max_approach_rate_mps",
    15: "traffic_density",
    16: "vru_ratio",
    17: "scene_luminance_mean",
    18: "scene_contrast_rms",
    19: "glare_index",
    20: "rain_fog_scatter_index"
]
```

---

## 5. Target Variable & Risk Classification Thresholds

- **Target Variable ($y$):** `binary_risk_label` $\in \{0, 1\}$
  - `0 (Safe / Normal Driving)`: Contextual driving conditions within standard safety envelopes ($\text{TTC} > 3.0\text{s}$, stable lane tracking, benign environmental hazards).
  - `1 (Elevated / Imminent Crash Risk)`: Critical driving situation requiring active driver intervention or automated collision mitigation ($\text{TTC} < 2.2\text{s}$, rapid closure rate, extreme lane departure, or high-density VRU conflict under poor visibility).
- **Continuous Risk Score ($\hat{p}$):** Calibrated predicted probability $P(y=1 \mid \mathbf{x}) \in [0.0, 1.0]$.
  - **Low Risk:** $\hat{p} < 0.35$ (Green HUD)
  - **Moderate / Elevated:** $0.35 \le \hat{p} < 0.70$ (Yellow HUD Warning)
  - **High / Critical Risk:** $\hat{p} \ge 0.70$ (Red Flashing Alert + Collision Warning)
