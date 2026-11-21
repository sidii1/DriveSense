"""YOLOv8 detection + ByteTrack tracking wrapper."""
import numpy as np
import config as C


class Detector:
    def __init__(self, weights=None, conf=None, imgsz=None, device=None):
        from ultralytics import YOLO  # lazy import so other modules work without torch
        self.model = YOLO(weights or C.YOLO_WEIGHTS)
        self.conf = conf or C.CONF
        self.imgsz = imgsz or C.IMGSZ
        self.device = device

    def reset(self):
        """Call at the start of every video so track IDs / Kalman state do not leak across clips."""
        self.model.predictor = None

    def track(self, frame):
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                             classes=list(C.CLASS_NAMES), conf=self.conf,
                             imgsz=self.imgsz, verbose=False, device=self.device)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        ids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None
               else -np.ones(len(cls), dtype=int))
        for b, c, s, i in zip(xyxy, cls, conf, ids):
            out.append({"xyxy": [float(v) for v in b], "cls": int(c), "conf": float(s), "id": int(i)})
        return out
