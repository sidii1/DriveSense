"""Experiment 3 configuration. Edit paths / thresholds here only."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] if len(ROOT.parents) > 1 else ROOT
CCD_DIR = Path(os.environ.get("EXP3_CCD_DIR", REPO / "data" / "Crash-1500"))
OUT = Path(os.environ.get("EXP3_OUT", ROOT / "results"))
FRAME_DIR = OUT / "frame_features"
DEMO_DIR = OUT / "demo_videos"
FIG_DIR = OUT / "figures"
for _d in (FRAME_DIR, DEMO_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- video / detector ----
FPS_DEFAULT = 10.0            # CCD clips are ~10 fps; the real fps is read from each file
YOLO_WEIGHTS = "yolov8s.pt"
CONF = 0.25
IMGSZ = 960
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VRU_IDS = {0, 1, 3}           # person, bicycle, motorcycle
VEHICLE_IDS = {2, 5, 7}       # car, bus, truck

# ---- distance / TTC ----
# Distance is APPROXIMATE (pinhole + assumed real height + assumed FOV). Dashcam intrinsics are unknown.
OBJ_HEIGHT_M = {0: 1.7, 1: 1.7, 2: 1.5, 3: 1.5, 5: 3.2, 7: 3.0}
HFOV_DEG = 70.0
# TTC is computed from bounding-box scale expansion (tau = 1 / d(ln scale)/dt); needs no calibration.
TTC_WINDOW_S = 0.6
MIN_TTC_SAMPLES = 4
INV_TTC_MIN = 0.02            # 1/s; below this the object is treated as not approaching
PATH_HALF_WIDTH = 0.22        # "in-path" corridor half-width as a fraction of frame width
PATH_MIN_BOTTOM = 0.50        # box bottom must be below this fraction of frame height

# ---- temporal analysis (seconds relative to collision onset) ----
WINDOWS = {"early": (-3.0, -2.0), "mid": (-2.0, -1.0), "late": (-1.0, 0.0)}
MIN_FRAMES_PER_WINDOW = 3
