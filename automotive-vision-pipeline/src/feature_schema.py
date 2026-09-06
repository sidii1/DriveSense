"""
Feature Schema Definitions for Multi-Modal Crash Risk Intelligence Platform.

This module formalizes:
1. Tabular NHTSA Crash Baseline Features (Model A)
2. Real-Time Vision & Kinematic Features (Model B)
3. Fused Multi-Modal Feature Vector (Model C)
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import numpy as np


# =====================================================================
# 1. TABULAR FEATURES (NHTSA Crash Baseline - Model A)
# =====================================================================
TABULAR_FEATURE_NAMES = [
    "speed_limit_mph",          # e.g., 25, 35, 45, 55, 65, 75
    "road_alignment",           # 0 = Straight, 1 = Curve
    "road_surface_condition",   # 0 = Dry, 1 = Wet/Snow/Ice
    "light_condition",          # 0 = Daylight, 1 = Dark/Dusk/Dawn, 2 = Dark Unlit
    "weather_condition",        # 0 = Clear, 1 = Rain/Snow/Fog, 2 = Severe Storm
    "driver_age_group",         # 0 = Young (<25), 1 = Middle (25-64), 2 = Senior (65+)
    "driver_impairment_flag",   # 0 = None, 1 = Suspected Alcohol/Distraction
    "vehicle_type",             # 0 = Passenger Car, 1 = SUV/Truck, 2 = Motorcycle
    "historical_segment_risk",  # Continuous 0.0 - 1.0 historical crash rate
]


# =====================================================================
# 2. REAL-TIME VISION & KINEMATICS FEATURES (Model B)
# =====================================================================
VISION_FEATURE_NAMES = [
    "lane_offset_m",            # Lateral offset from lane center in meters (- left, + right)
    "lane_curvature_rad_m",     # 1/Radius of curvature in 1/meters (0 = straight)
    "lane_confidence",          # 0.0 - 1.0 confidence of lane boundary detection
    "min_ttc_sec",              # Minimum Time-to-Collision among forward objects (clamped [0.1, 10.0])
    "nearest_obj_distance_m",   # Pinhole estimated distance to closest lead vehicle/obstacle (meters)
    "max_approach_rate_mps",    # Maximum closing velocity (rate of distance decrease in m/s)
    "traffic_density",          # Total active tracked objects in forward FOV
    "vru_ratio",                # Vulnerable Road User ratio: (pedestrians + 2-wheelers) / total_objects
    "scene_luminance_mean",     # Mean grayscale intensity [0 - 255]
    "scene_contrast_rms",       # RMS contrast (std dev of grayscale intensities)
    "glare_index",              # High-intensity saturated pixel ratio [0.0 - 1.0]
    "rain_fog_scatter_index",   # High-frequency edge degradation proxy [0.0 - 1.0]
]


# =====================================================================
# 3. FUSED MULTI-MODAL FEATURE VECTOR (Model C)
# =====================================================================
FUSED_FEATURE_NAMES = TABULAR_FEATURE_NAMES + VISION_FEATURE_NAMES


@dataclass
class LaneTelemetry:
    lane_offset_m: float = 0.0
    lane_curvature_rad_m: float = 0.0
    lane_confidence: float = 0.0
    lane_detected: bool = False
    left_line: Optional[List[List[int]]] = None
    right_line: Optional[List[List[int]]] = None


@dataclass
class TrackedObjectTelemetry:
    id: int
    class_name: str
    bbox: List[int]             # [x1, y1, x2, y2]
    confidence: float
    persistence_frames: int = 1
    distance_m: float = 50.0
    approach_rate_mps: float = 0.0
    ttc_sec: float = 10.0


@dataclass
class SceneTelemetry:
    luminance_mean: float = 128.0
    contrast_rms: float = 50.0
    glare_index: float = 0.0
    rain_fog_scatter_index: float = 0.0
    traffic_density: int = 0
    vru_ratio: float = 0.0


@dataclass
class FrameTelemetry:
    frame_id: int
    timestamp_sec: float
    lane: LaneTelemetry = field(default_factory=LaneTelemetry)
    objects: List[TrackedObjectTelemetry] = field(default_factory=list)
    scene: SceneTelemetry = field(default_factory=SceneTelemetry)

    def to_vision_feature_vector(self) -> np.ndarray:
        """Converts frame telemetry to a normalized 1D numpy array matching VISION_FEATURE_NAMES."""
        # Minimum TTC and closest distance
        if self.objects:
            valid_ttcs = [o.ttc_sec for o in self.objects if o.ttc_sec > 0]
            min_ttc = min(valid_ttcs) if valid_ttcs else 10.0
            nearest_dist = min([o.distance_m for o in self.objects])
            max_approach = max([o.approach_rate_mps for o in self.objects])
        else:
            min_ttc = 10.0
            nearest_dist = 60.0
            max_approach = 0.0

        vec = [
            float(self.lane.lane_offset_m),
            float(self.lane.lane_curvature_rad_m),
            float(self.lane.lane_confidence),
            float(np.clip(min_ttc, 0.1, 10.0)),
            float(np.clip(nearest_dist, 1.0, 100.0)),
            float(np.clip(max_approach, -20.0, 40.0)),
            float(self.scene.traffic_density),
            float(np.clip(self.scene.vru_ratio, 0.0, 1.0)),
            float(self.scene.luminance_mean),
            float(self.scene.contrast_rms),
            float(self.scene.glare_index),
            float(self.scene.rain_fog_scatter_index),
        ]
        return np.array(vec, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
