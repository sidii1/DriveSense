"""
Scene Environmental & Contextual Feature Extraction Module.

Extracts:
1. Photometric Features via OpenCV:
   - scene_luminance_mean: Average brightness level [0-255] (Day/Night discriminator)
   - scene_contrast_rms: Grayscale standard deviation (Dynamic range / Fog proxy)
   - glare_index: Fraction of saturated pixels (>245) (Headlight / Sun glare)
   - rain_fog_scatter_index: Attenuated high-frequency edge energy
2. Dynamic Scene Composition:
   - traffic_density: Total active object count in forward camera frustum
   - vru_ratio: Vulnerable Road User proportion (motorcycles + pedestrians + bicycles) / total
"""

import cv2
import numpy as np
from typing import List, Dict, Any


class SceneFeatureExtractor:
    VRU_CLASSES = {"person", "bicycle", "motorbike"}

    def __init__(self):
        pass

    def extract_features(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extracts environmental photometric and composition telemetry from frame.
        """
        # Downscale for fast CPU computation
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 1. Mean Luminance & RMS Contrast
        mean_lum = float(np.mean(gray))
        contrast_rms = float(np.std(gray))

        # 2. Glare Index: fraction of saturated high-intensity pixels
        saturated_pixels = np.count_nonzero(gray >= 245)
        glare_index = float(saturated_pixels / float(gray.size))

        # 3. Rain / Fog Scatter Index: Laplacian gradient energy proxy
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())
        # Lower Laplacian variance indicates heavy haze / blur / rainy attenuation
        scatter_index = float(1.0 / (1.0 + np.exp((lap_var - 150.0) / 50.0)))

        # 4. Traffic Density & VRU Ratio
        total_objects = len(tracked_objects)
        vru_count = sum(1 for obj in tracked_objects if obj.get("class") in self.VRU_CLASSES)
        vru_ratio = float(vru_count / max(1, total_objects)) if total_objects > 0 else 0.0

        return {
            "luminance_mean": round(mean_lum, 2),
            "contrast_rms": round(contrast_rms, 2),
            "glare_index": round(glare_index, 4),
            "rain_fog_scatter_index": round(scatter_index, 4),
            "traffic_density": int(total_objects),
            "vru_ratio": round(vru_ratio, 3),
            "vru_count": int(vru_count)
        }
