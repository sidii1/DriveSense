"""Approximate distance, scale-based TTC and per-frame traffic summaries."""
import math
from collections import defaultdict
import numpy as np
import config as C


class TrackKinematics:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.f_px = (W / 2) / math.tan(math.radians(C.HFOV_DEG / 2))
        self.hist = defaultdict(list)

    @staticmethod
    def _scale(d):
        x1, y1, x2, y2 = d["xyxy"]
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        return h if d["cls"] in C.VRU_IDS else math.sqrt(w * h)

    def update(self, t, dets):
        """Adds dist_m, in_path, inv_ttc (1/s) and ttc_s to every detection dict."""
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            cx = (x1 + x2) / 2
            d["dist_m"] = self.f_px * C.OBJ_HEIGHT_M[d["cls"]] / max(y2 - y1, 1.0)
            d["in_path"] = bool(abs(cx - self.W / 2) < C.PATH_HALF_WIDTH * self.W
                                and y2 > C.PATH_MIN_BOTTOM * self.H)
            d["inv_ttc"], d["ttc_s"] = 0.0, float("inf")
            if d["id"] < 0:
                continue
            h = self.hist[d["id"]]
            h.append((t, math.log(self._scale(d))))
            recent = [(a, b) for a, b in h if a >= t - C.TTC_WINDOW_S - 1e-6]
            if len(recent) >= C.MIN_TTC_SAMPLES:
                ts = np.array([a for a, _ in recent])
                ls = np.array([b for _, b in recent])
                k = float(np.polyfit(ts, ls, 1)[0])       # d(ln scale)/dt = 1/TTC
                if k > C.INV_TTC_MIN:
                    d["inv_ttc"], d["ttc_s"] = k, 1.0 / k
        return dets


def frame_summary(dets, W, H):
    veh = [d for d in dets if d["cls"] in C.VEHICLE_IDS]
    vru = [d for d in dets if d["cls"] in C.VRU_IDS]
    path = [d for d in dets if d["in_path"]]
    area = sum((d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1]) for d in dets) / (W * H)
    max_path = max([d["inv_ttc"] for d in path], default=0.0)
    return {
        "n_vehicles": len(veh),
        "n_vru": len(vru),
        "n_person": sum(d["cls"] == 0 for d in dets),
        "n_bicycle": sum(d["cls"] == 1 for d in dets),
        "n_motorcycle": sum(d["cls"] == 3 for d in dets),
        "n_truck_bus": sum(d["cls"] in (5, 7) for d in dets),
        "n_tracked": sum(d["id"] >= 0 for d in dets),
        "occupancy": float(min(area, 1.0)),
        "n_path": len(path),
        "max_inv_ttc_path": max_path,
        "max_inv_ttc_all": max([d["inv_ttc"] for d in dets], default=0.0),
        "min_ttc_path_s": (1.0 / max_path) if max_path > 0 else float("nan"),
        "min_dist_path_m": min([d["dist_m"] for d in path], default=float("nan")),
        "min_dist_vru_m": min([d["dist_m"] for d in vru], default=float("nan")),
    }
