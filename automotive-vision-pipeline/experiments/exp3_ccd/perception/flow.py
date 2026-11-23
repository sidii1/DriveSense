"""Dense optical flow (OpenCV Farneback) -> looming / divergence cues, no calibration needed.

Recovered by matching the stored `flow_*` columns of the v2 pilot run (results_v1_backup/exp3_v2_test):
grayscale frames downscaled 4x (INTER_AREA), Farneback(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
poly_n=5, poly_sigma=1.2), divergence = d(u)/dx + d(v)/dy via central differences, flow in low-res pixels.
Central ("ahead of the ego-vehicle") region: rows 40-90 %, columns 30-70 % of the frame.
"""
import cv2
import numpy as np

SCALE = 0.25
FARNEBACK = (0.5, 3, 15, 3, 5, 1.2, 0)   # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
CENTER = (0.4, 0.9, 0.3, 0.7)            # y0, y1, x0, x1 as fractions of height / width
NAN_ROW = {"flow_div_center": np.nan, "flow_div_center_p90": np.nan,
           "flow_div_full": np.nan, "flow_mag_center": np.nan}


class FlowEstimator:
    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None

    def update(self, frame_bgr):
        g = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), None, fx=SCALE, fy=SCALE,
                       interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, g
        if prev is None:
            return dict(NAN_ROW)
        fl = cv2.calcOpticalFlowFarneback(prev, g, None, *FARNEBACK)
        div = np.gradient(fl[..., 0], axis=1) + np.gradient(fl[..., 1], axis=0)
        H, W = g.shape
        y0, y1, x0, x1 = CENTER
        sl = (slice(int(y0 * H), int(y1 * H)), slice(int(x0 * W), int(x1 * W)))
        mag = np.linalg.norm(fl, axis=2)
        return {"flow_div_center": float(div[sl].mean()),
                "flow_div_center_p90": float(np.percentile(div[sl], 90)),
                "flow_div_full": float(div.mean()),
                "flow_mag_center": float(mag[sl].mean())}
