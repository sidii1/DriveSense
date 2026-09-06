"""
Object Detection & Multi-Object Tracking (MOT) Module via YOLOv8 + ByteTrack.

Provides:
- Persistent Object IDs across frames via ByteTrack.
- Tracking history and persistence duration.
- Approach Rate calculation (bounding box height growth rate as closing velocity proxy).
- Kinematic object telemetry formatting.
"""

import os
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO


class ObjectTracker:
    # COCO Class IDs relevant to automotive driving
    TARGET_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorbike",
        5: "bus",
        7: "truck",
        9: "traffic light"
    }

    CLASS_COLORS = {
        "car":           (255, 120,   0),    # Blue-Cyan BGR
        "truck":         (255,  50,  50),
        "bus":           (255,   0, 180),
        "motorbike":     (  0, 165, 255),    # Orange
        "person":        (  0, 255, 255),    # Yellow
        "bicycle":       ( 50, 205,  50),    # Green
        "traffic light": (  0, 215, 255),
    }
    DEFAULT_COLOR = (180, 180, 180)

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        # Check local paths for weights
        local_candidates = [
            model_name,
            os.path.join(os.path.dirname(__file__), model_name),
            os.path.join(os.path.dirname(__file__), "..", model_name),
            os.path.join(os.path.dirname(__file__), "..", "models", model_name),
        ]
        chosen_model = model_name
        for cand in local_candidates:
            if os.path.exists(cand):
                chosen_model = os.path.abspath(cand)
                break

        print(f"[INFO] Initializing YOLOv8 Object Tracker with ByteTrack ({chosen_model})...")
        self.model = YOLO(chosen_model)
        
        # Track history store: {track_id: [{"timestamp": t, "bbox": (x1,y1,x2,y2), "h": height, "area": area}]}
        self.track_history: Dict[int, List[Dict[str, Any]]] = {}
        self.max_history_len = 30
        print("[INFO] YOLOv8 + ByteTrack tracker ready.")

    def track(self, frame: np.ndarray, timestamp_sec: float = 0.0) -> List[Dict[str, Any]]:
        """
        Runs YOLO detection + ByteTrack multi-object tracking on frame.

        Returns:
            List of tracked object dicts containing:
            id, class, confidence, bbox [x1, y1, x2, y2], persistence_frames, approach_rate_px_s
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence_threshold,
            verbose=False
        )[0]

        tracked_objects = []
        current_active_ids = set()

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            clss = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                if int(cls_id) not in self.TARGET_CLASSES:
                    continue

                class_name = self.TARGET_CLASSES[int(cls_id)]
                x1, y1, x2, y2 = map(int, box)
                bbox_h = max(1, y2 - y1)
                bbox_w = max(1, x2 - x1)
                bbox_area = bbox_h * bbox_w

                current_active_ids.add(int(track_id))

                # Update track history buffer
                if int(track_id) not in self.track_history:
                    self.track_history[int(track_id)] = []

                history = self.track_history[int(track_id)]
                history.append({
                    "timestamp": timestamp_sec,
                    "bbox": (x1, y1, x2, y2),
                    "h": bbox_h,
                    "w": bbox_w,
                    "area": bbox_area
                })
                if len(history) > self.max_history_len:
                    history.pop(0)

                persistence = len(history)

                # Approach rate: Rate of bounding box height expansion (pixels/sec or normalized proxy)
                approach_rate_px_s = 0.0
                if len(history) >= 3:
                    prev_h = history[-3]["h"]
                    dt = max(0.01, timestamp_sec - history[-3]["timestamp"])
                    if dt <= 0.01:
                        # Fallback if timestamp not strictly incremented (e.g. frame index based at 30 fps)
                        dt = 3.0 / 30.0
                    approach_rate_px_s = (bbox_h - prev_h) / dt

                tracked_objects.append({
                    "id": int(track_id),
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "persistence_frames": persistence,
                    "approach_rate_px_s": round(float(approach_rate_px_s), 2)
                })

        # Prune old inactive tracks from history buffer
        for old_id in list(self.track_history.keys()):
            if old_id not in current_active_ids:
                # Retain for brief occlusion (up to 15 frames) or delete
                if len(self.track_history[old_id]) > 0:
                    self.track_history[old_id].pop(0)
                if len(self.track_history[old_id]) == 0:
                    del self.track_history[old_id]

        return tracked_objects

    def draw_tracked_objects(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> np.ndarray:
        """Draws bounding boxes, tracking IDs, persistence, and approach rate badges."""
        output = frame.copy()

        for obj in tracked_objects:
            obj_id = obj["id"]
            cls_name = obj["class"]
            conf = obj["confidence"]
            x1, y1, x2, y2 = obj["bbox"]
            persistence = obj["persistence_frames"]
            approach = obj.get("approach_rate_px_s", 0.0)
            color = self.CLASS_COLORS.get(cls_name, self.DEFAULT_COLOR)

            # Box outline
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label text
            label = f"ID #{obj_id} {cls_name} ({conf*100:.0f}%)"
            sub_label = f"Age:{persistence}f"
            if approach > 15.0:
                sub_label += f" | CLOSING {approach:.0f}px/s"

            # Badge background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output, (x1, max(0, y1 - 22)), (x1 + lw + 12, y1), color, -1)
            cv2.putText(output, label, (x1 + 6, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Bottom approach indicator if rapidly closing
            if approach > 25.0:
                cv2.rectangle(output, (x1, y2), (x1 + 140, y2 + 18), (0, 0, 255), -1)
                cv2.putText(output, "RAPID APPROACH", (x1 + 4, y2 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        return output