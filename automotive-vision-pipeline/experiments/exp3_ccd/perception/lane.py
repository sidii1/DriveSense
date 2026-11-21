"""Classical (Canny + Hough) lane detection. Expect it to fail often on crash footage; report the success rate."""
import cv2
import numpy as np

NAN = float("nan")


def detect_lane(frame):
    H, W = frame.shape[:2]
    s = 640.0 / W if W > 640 else 1.0
    img = cv2.resize(frame, None, fx=s, fy=s) if s != 1.0 else frame
    h, w = img.shape[:2]
    out = {"lane_ok": False, "lane_offset": NAN, "lane_width_px": NAN, "lane_curv": NAN,
           "lane_n_left": 0, "lane_n_right": 0, "geom": None}

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    y_top, y_bot = int(0.60 * h), int(0.85 * h)          # ignore sky and bonnet
    mask = np.zeros_like(edges)
    poly = np.array([[(int(0.02 * w), y_bot), (int(0.42 * w), y_top),
                      (int(0.58 * w), y_top), (int(0.98 * w), y_bot)]], np.int32)
    cv2.fillPoly(mask, poly, 255)
    edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=int(0.06 * w), maxLineGap=int(0.04 * w))
    if lines is None:
        return out

    L, R = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4 or abs(slope) > 4:
            continue
        mx = (x1 + x2) / 2
        if slope < 0 and mx < 0.55 * w:
            L.append((x1, y1, x2, y2))
        elif slope > 0 and mx > 0.45 * w:
            R.append((x1, y1, x2, y2))
    out["lane_n_left"], out["lane_n_right"] = len(L), len(R)
    if not L or not R:
        return out

    def fit(segs):
        pts = np.array([p for x1, y1, x2, y2 in segs for p in ((x1, y1), (x2, y2))], float)
        lin = np.polyfit(pts[:, 1], pts[:, 0], 1)
        quad = None
        if len(pts) >= 6 and np.ptp(pts[:, 1]) > 0.08 * h:
            quad = np.polyfit(pts[:, 1], pts[:, 0], 2)
        return lin, quad

    (ll, lq), (rl, rq) = fit(L), fit(R)
    xl, xr = np.polyval(ll, y_bot), np.polyval(rl, y_bot)
    width = xr - xl
    if not (0.15 * w < width < 0.85 * w):
        return out

    out["lane_ok"] = True
    out["lane_offset"] = float((w / 2 - (xl + xr) / 2) / width)   # >0: camera right of lane centre
    out["lane_width_px"] = float(width / s)
    if lq is not None and rq is not None:
        out["lane_curv"] = float((lq[0] + rq[0]) / 2 * h)            # dimensionless curvature proxy
    out["geom"] = [(np.polyval(ll, y_bot) / s, y_bot / s, np.polyval(ll, y_top) / s, y_top / s),
                   (np.polyval(rl, y_bot) / s, y_bot / s, np.polyval(rl, y_top) / s, y_top / s)]
    return out
