"""
Enhanced Lane Detection Module with Geometry, Curvature, Offset & Confidence Estimation.

Calculates:
- lane_offset_m: Lateral displacement from lane center in meters.
- lane_curvature_rad_m: Inverse curve radius (1/R) in 1/m.
- lane_confidence: Quantitative stability/parallelism confidence score [0.0 - 1.0].
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class LaneDetector:
    # Typical standard highway lane width is ~3.7 meters (12 feet)
    STANDARD_LANE_WIDTH_M = 3.7

    def __init__(self):
        self.left_lane_cache = None
        self.right_lane_cache = None
        self.consecutive_misses = 0
        self.max_cache_frames = 10

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes single dashcam frame for lane detection.

        Returns:
            annotated_frame: Frame with lane polygon and HUD overlays.
            telemetry: Dict containing lane_offset_m, lane_curvature_rad_m, lane_confidence, lane_detected.
        """
        height, width = frame.shape[:2]

        # 1. Color Masking in HLS Color Space (White + Yellow Markings)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # White lane lines (high lightness)
        lower_white = np.array([0, 175, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        # Yellow lane lines (hue in yellow range, high saturation)
        lower_yellow = np.array([10, 30, 70], dtype=np.uint8)
        upper_yellow = np.array([45, 204, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 2. Combined Color Mask + Grayscale Canny Edge Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        color_edges = cv2.Canny(cv2.GaussianBlur(color_mask, (5, 5), 0), 40, 120)
        edges = cv2.bitwise_or(gray_edges, color_edges)

        # 3. Trapezoidal Region of Interest (ROI)
        roi_vertices = np.array([[
            (int(width * 0.02), height),
            (int(width * 0.40), int(height * 0.48)),
            (int(width * 0.60), int(height * 0.48)),
            (int(width * 0.98), height)
        ]], dtype=np.int32)

        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # 4. Probabilistic Hough Lines
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=30,
            maxLineGap=40
        )

        # 5. Lane Fitting and Temporal Exponential Smoothing
        left_line_raw, right_line_raw = self._fit_lanes(lines, height, width)
        left_line, right_line = self._smooth_and_validate(left_line_raw, right_line_raw)

        # 6. Compute Telemetry (Offset, Curvature, Confidence)
        telemetry = self._compute_lane_telemetry(left_line, right_line, width, height)

        # 7. Render Visualization Overlay
        annotated_frame = self._draw_lane_overlay(frame, left_line, right_line, telemetry)

        return annotated_frame, telemetry

    def _fit_lanes(self, lines, height: int, width: int):
        left_pts = []
        right_pts = []

        if lines is None:
            return None, None

        for line in lines:
            pts = line.flatten()
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Left lane usually has negative slope in image coordinates
            if -3.5 < slope < -0.30 and x1 < width * 0.55 and x2 < width * 0.55:
                left_pts.extend([(x1, y1), (x2, y2)])
            # Right lane usually has positive slope
            elif 0.30 < slope < 3.5 and x1 > width * 0.45 and x2 > width * 0.45:
                right_pts.extend([(x1, y1), (x2, y2)])

        y_bottom = height
        y_top = int(height * 0.58)

        left_extrapolated = self._extrapolate_line(left_pts, y_bottom, y_top)
        right_extrapolated = self._extrapolate_line(right_pts, y_bottom, y_top)

        if left_extrapolated is not None and right_extrapolated is None:
            (lx1, ly1), (lx2, ly2) = left_extrapolated
            nominal_w = int(width * 0.45)
            right_extrapolated = ((lx1 + nominal_w, ly1), (lx2 + nominal_w, ly2))
        elif right_extrapolated is not None and left_extrapolated is None:
            (rx1, ry1), (rx2, ry2) = right_extrapolated
            nominal_w = int(width * 0.45)
            left_extrapolated = ((rx1 - nominal_w, ry1), (rx2 - nominal_w, ry2))

        return left_extrapolated, right_extrapolated

    def _extrapolate_line(self, pts, y_bottom: int, y_top: int):
        if len(pts) < 2:
            return None

        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]

        try:
            poly = np.polyfit(y_coords, x_coords, 1)
            slope, intercept = poly[0], poly[1]

            x_bottom = int(slope * y_bottom + intercept)
            x_top = int(slope * y_top + intercept)
            return ((x_bottom, y_bottom), (x_top, y_top))
        except Exception:
            return None

    def _smooth_and_validate(self, left_line, right_line):
        alpha = 0.35  # Smoothing factor (weight of current frame)

        # Update left lane cache
        if left_line is not None:
            if self.left_lane_cache is None:
                self.left_lane_cache = left_line
            else:
                (x1_c, y1_c), (x2_c, y2_c) = self.left_lane_cache
                (x1_n, y1_n), (x2_n, y2_n) = left_line
                self.left_lane_cache = (
                    (int(x1_c * (1 - alpha) + x1_n * alpha), int(y1_c * (1 - alpha) + y1_n * alpha)),
                    (int(x2_c * (1 - alpha) + x2_n * alpha), int(y2_c * (1 - alpha) + y2_n * alpha))  # was x2_c/x2_n
                )
        
        # Update right lane cache
        if right_line is not None:
            if self.right_lane_cache is None:
                self.right_lane_cache = right_line
            else:
                (x1_c, y1_c), (x2_c, y2_c) = self.right_lane_cache
                (x1_n, y1_n), (x2_n, y2_n) = right_line
                self.right_lane_cache = (
                     (int(x1_c * (1 - alpha) + x1_n * alpha), int(y1_c * (1 - alpha) + y1_n * alpha)),
                     (int(x2_c * (1 - alpha) + x2_n * alpha), int(y2_c * (1 - alpha) + y2_n * alpha))  # was x2_c/x2_n
)

        if left_line is None and right_line is None:
            self.consecutive_misses += 1
            if self.consecutive_misses > self.max_cache_frames:
                self.left_lane_cache = None
                self.right_lane_cache = None
        else:
            self.consecutive_misses = 0

        return self.left_lane_cache, self.right_lane_cache

    def _compute_lane_telemetry(self, left_line, right_line, width: int, height: int) -> Dict[str, Any]:
        if left_line is None or right_line is None:
            return {
                "lane_offset_m": 0.0,
                "lane_curvature_rad_m": 0.0,
                "lane_confidence": 0.0,
                "lane_detected": False,
                "left_line": None,
                "right_line": None
            }

        left_bottom_x = left_line[0][0]
        right_bottom_x = right_line[0][0]
        lane_width_px = max(10, right_bottom_x - left_bottom_x)

        # Scale factor: meters per pixel at bottom of hood
        meters_per_pixel = self.STANDARD_LANE_WIDTH_M / lane_width_px

        # Vehicle center is assumed at camera center (width / 2)
        vehicle_center_x = width / 2.0
        lane_center_x = (left_bottom_x + right_bottom_x) / 2.0

        # Lateral lane offset in meters (Negative = Left bias, Positive = Right bias)
        lane_offset_m = (vehicle_center_x - lane_center_x) * meters_per_pixel

        # Lane curvature approximation via top/bottom slope divergence
        left_top_x = left_line[1][0]
        right_top_x = right_line[1][0]
        top_lane_center_x = (left_top_x + right_top_x) / 2.0

        # Curvature proxy in rad/m
        dx = (top_lane_center_x - lane_center_x) * meters_per_pixel
        dy = (height * 0.42) * meters_per_pixel
        lane_curvature_rad_m = float(np.clip(dx / (dy**2 + 1e-4), -0.05, 0.05))

        # Lane confidence score based on lane width sanity and detection quality
        width_ratio = lane_width_px / width
        width_sanity = 1.0 - np.clip(abs(width_ratio - 0.70) / 0.35, 0.0, 1.0)
        cache_penalty = max(0.0, 1.0 - (self.consecutive_misses / float(self.max_cache_frames)))
        lane_confidence = float(np.clip(0.5 * width_sanity + 0.5 * cache_penalty, 0.0, 1.0))

        return {
            "lane_offset_m": round(float(lane_offset_m), 3),
            "lane_curvature_rad_m": round(float(lane_curvature_rad_m), 5),
            "lane_confidence": round(float(lane_confidence), 3),
            "lane_detected": True,
            "left_line": [list(left_line[0]), list(left_line[1])],
            "right_line": [list(right_line[0]), list(right_line[1])]
        }

    def _draw_lane_overlay(self, frame: np.ndarray, left_line, right_line, telemetry: Dict[str, Any]) -> np.ndarray:
        output = frame.copy()
        height, width = frame.shape[:2]

        if telemetry["lane_detected"] and left_line is not None and right_line is not None:
            # Filled drivable area polygon
            pts = np.array([
                left_line[0],
                left_line[1],
                right_line[1],
                right_line[0]
            ], dtype=np.int32)

            poly_overlay = output.copy()
            # Color tone based on offset: Green if centered, Yellow/Red if drifting
            offset_abs = abs(telemetry["lane_offset_m"])
            if offset_abs < 0.35:
                poly_color = (0, 200, 50)     # Benign Green
            elif offset_abs < 0.75:
                poly_color = (0, 215, 255)    # Moderate Yellow
            else:
                poly_color = (0, 70, 255)     # Warning Orange/Red

            cv2.fillPoly(poly_overlay, [pts], poly_color)
            cv2.addWeighted(poly_overlay, 0.28, output, 0.72, 0, dst=output)

            # Draw lane boundaries
            cv2.line(output, left_line[0], left_line[1], (0, 0, 255), 6, cv2.LINE_AA)
            cv2.line(output, right_line[0], right_line[1], (0, 0, 255), 6, cv2.LINE_AA)

            # Draw center guidance notch
            lane_center_bottom_x = int((left_line[0][0] + right_line[0][0]) / 2)
            cv2.circle(output, (lane_center_bottom_x, height - 20), 8, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.line(output, (int(width / 2), height - 35), (int(width / 2), height - 5), (0, 255, 0), 3, cv2.LINE_AA)

        return output

if __name__ == "__main__":
    detector = LaneDetector()
    img = cv2.imread('data/kitti_raw/image_02/data/0000000000.png')
    annotated, telemetry = detector.process_frame(img)
    print(telemetry)
    cv2.imshow("Lane Detection", annotated)
    cv2.waitKey(0)
