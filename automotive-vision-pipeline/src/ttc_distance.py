"""
Monocular Pinhole Distance & Time-to-Collision (TTC) Module.

Assumptions & Formulation:
1. Pinhole Camera Model:
   distance (m) = (focal_length_px * assumed_height_m) / bbox_height_px
   Assumed object physical heights:
     - car: 1.50 m
     - truck: 3.20 m
     - bus: 3.40 m
     - motorbike: 1.20 m
     - person: 1.70 m
     - bicycle: 1.40 m
2. Time-to-Collision (TTC):
   Closing Rate = (distance_prev - distance_curr) / delta_t  [m/s]
   TTC = distance_curr / Closing_Rate  [seconds]
   - Clamped to [0.1s, 10.0s]. If object is moving away or stationary (closing_rate <= 0), TTC = 10.0s (safe).
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional


class DistanceTTCEstimator:
    # Empirical standard heights for road objects in meters
    ASSUMED_HEIGHTS_M = {
        "car":           1.50,
        "truck":         3.20,
        "bus":           3.40,
        "motorbike":     1.20,
        "person":        1.70,
        "bicycle":       1.40,
        "traffic light": 1.00
    }
    DEFAULT_HEIGHT_M = 1.60

    def __init__(self, focal_length_px: float = 720.0):
        """
        Args:
            focal_length_px: Effective vertical focal length in pixels (typical 720p dashcam ~700-800px).
        """
        self.focal_length_px = focal_length_px
        # Store previous distances for rate-of-closure: {track_id: (timestamp, distance_m)}
        self.distance_cache: Dict[int, List[tuple]] = {}
        self.max_cache_len = 10

    def estimate_distance_and_ttc(
        self,
        tracked_objects: List[Dict[str, Any]],
        timestamp_sec: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Enriches tracked objects with distance_m, approach_rate_mps, and ttc_sec.
        """
        enriched_objects = []

        for obj in tracked_objects:
            obj_id = obj["id"]
            cls_name = obj["class"]
            x1, y1, x2, y2 = obj["bbox"]
            bbox_h = max(4, y2 - y1)

            # 1. Pinhole Distance Estimation
            real_h_m = self.ASSUMED_HEIGHTS_M.get(cls_name, self.DEFAULT_HEIGHT_M)
            dist_m = (self.focal_length_px * real_h_m) / float(bbox_h)
            dist_m = float(np.clip(dist_m, 1.5, 120.0))

            # 2. Track distance history for closing rate
            if obj_id not in self.distance_cache:
                self.distance_cache[obj_id] = []

            cache = self.distance_cache[obj_id]
            cache.append((timestamp_sec, dist_m))
            if len(cache) > self.max_cache_len:
                cache.pop(0)

            # 3. Calculate Approach Rate & TTC
            approach_rate_mps = 0.0
            ttc_sec = 10.0  # Default benign TTC

            if len(cache) >= 3:
                t_prev, d_prev = cache[-3]
                dt = max(0.01, timestamp_sec - t_prev)
                if dt <= 0.01:
                    dt = 3.0 / 30.0  # Fallback for 30fps

                # Closing rate = - d(dist)/dt (positive when getting closer)
                closing_rate = (d_prev - dist_m) / dt
                approach_rate_mps = float(closing_rate)

                if closing_rate > 0.4:  # Approaching faster than 1.5 km/h
                    ttc_raw = dist_m / closing_rate
                    ttc_sec = float(np.clip(ttc_raw, 0.1, 10.0))
                else:
                    ttc_sec = 10.0  # Stationary or receding

            obj_copy = dict(obj)
            obj_copy["distance_m"] = round(dist_m, 1)
            obj_copy["approach_rate_mps"] = round(approach_rate_mps, 2)
            obj_copy["ttc_sec"] = round(ttc_sec, 2)
            enriched_objects.append(obj_copy)

        return enriched_objects

    def draw_distance_ttc_hud(self, frame: np.ndarray, enriched_objects: List[Dict[str, Any]]) -> np.ndarray:
        """Renders distance tags, TTC warnings, and risk halos on objects."""
        output = frame.copy()

        for obj in enriched_objects:
            x1, y1, x2, y2 = obj["bbox"]
            dist_m = obj["distance_m"]
            ttc_sec = obj["ttc_sec"]
            cls_name = obj["class"]

            # Color coding by TTC severity
            if ttc_sec <= 2.0:
                badge_color = (0, 0, 255)       # CRITICAL RED ALERT
                warning_txt = f"CRITICAL TTC {ttc_sec:.1f}s | {dist_m:.0f}m"
                # Pulsing box outline
                cv2.rectangle(output, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 3, cv2.LINE_AA)
            elif ttc_sec <= 4.0:
                badge_color = (0, 165, 255)     # WARNING ORANGE
                warning_txt = f"TTC {ttc_sec:.1f}s | {dist_m:.0f}m"
            else:
                badge_color = (0, 200, 0)       # SAFE GREEN
                warning_txt = f"{dist_m:.0f}m"

            # Badge background
            (bw, bh), _ = cv2.getTextSize(warning_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y = max(bh + 4, y1 - 24)
            cv2.rectangle(output, (x1, badge_y - bh - 4), (x1 + bw + 10, badge_y + 4), badge_color, -1)
            cv2.putText(output, warning_txt, (x1 + 5, badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return output

if __name__ == "__main__":
    import cv2
    from src.object_detection import ObjectTracker

    tracker = ObjectTracker()
    ttc_est = DistanceTTCEstimator()

    img = cv2.imread('data/kitti_raw/image_02/data/0000000000.png')
    tracked = tracker.track(img, timestamp_sec=0.0)
    enriched = ttc_est.estimate_distance_and_ttc(tracked, timestamp_sec=0.0)

    print(f"Objects with distance/TTC: {len(enriched)}")
    for obj in enriched:
        print(obj)

    annotated = tracker.draw_tracked_objects(img, tracked)
    annotated = ttc_est.draw_distance_ttc_hud(annotated, enriched)
    cv2.imshow("Distance & TTC", annotated)
    cv2.waitKey(0)