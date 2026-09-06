"""
DriveSense AI — Unified Multi-Modal Automotive Risk & Vision Intelligence Server.

Unifies:
1. Real-Time ADAS Dashcam Perception & Telemetry Streaming (Lane, ByteTrack, TTC, Photometry)
2. Multi-Modal Risk Estimation (Model C Fused, Model B Vision, Model A Crash Baseline)
3. TreeSHAP Local Feature Explainability & Top-3 Contributing Risk Factors
4. Computer Vision Image Analysis (/analyze-image) with Multi-Signal Weather & Lighting Extraction
5. Indian Road Condition Adaptive Mapping
6. Analytics & Historical Prediction Telemetry
"""

import os
import io
import cv2
import json
import base64
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import xgboost as xgb
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_schema import TABULAR_FEATURE_NAMES, VISION_FEATURE_NAMES, FUSED_FEATURE_NAMES
from explainability import RiskExplainer
from lane_detection import LaneDetector
from object_detection import ObjectTracker
from ttc_distance import DistanceTTCEstimator
from scene_features import SceneFeatureExtractor


app = FastAPI(title="DriveSense AI — Multi-Modal Driving Risk Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
WEB_DIR = os.path.join(ROOT_DIR, "web")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# Mount static web directory
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# ── Load ML Models & Explainers ─────────────────────────────────────────────
model_a_path = os.path.join(MODELS_DIR, "model_a_baseline.json")
model_b_path = os.path.join(MODELS_DIR, "model_b_vision.json")
model_c_path = os.path.join(MODELS_DIR, "model_c_fused.json")

model_a = xgb.XGBClassifier()
model_b = xgb.XGBClassifier()
model_c = xgb.XGBClassifier()

if os.path.exists(model_a_path):
    model_a.load_model(model_a_path)
if os.path.exists(model_b_path):
    model_b.load_model(model_b_path)
if os.path.exists(model_c_path):
    model_c.load_model(model_c_path)

explainer_c = None
try:
    if os.path.exists(model_c_path):
        explainer_c = RiskExplainer(model_c_path)
except Exception as e:
    print(f"[WARN] Explainer initialization deferred: {e}")

# In-memory prediction log store
prediction_logs: List[Dict[str, Any]] = []

# ── Label Maps & Descriptions ───────────────────────────────────────────────
WEATHER_LABELS = {
    1: "Clear", 2: "Rain", 3: "Sleet/Hail", 4: "Snow",
    5: "Fog/Smog", 6: "Crosswinds", 10: "Cloudy",
    11: "Blowing Snow", 12: "Freezing Rain"
}
LIGHT_LABELS = {
    1: "Daylight", 2: "Dark – Not Lighted", 3: "Dark – Lighted",
    4: "Dawn", 5: "Dusk"
}
COLLISION_LABELS = {
    0: "Not Collision", 1: "Front-to-Rear", 2: "Front-to-Front",
    6: "Angle", 7: "Sideswipe (Same Dir)", 8: "Sideswipe (Opposite)"
}


# ── Pydantic Request Models ────────────────────────────────────────────────
class ScenarioPredictRequest(BaseModel):
    model_type: str = "C"           # "A", "B", "C"
    # Tabular features
    hour: int = 14
    month: int = 6
    day_week: int = 3
    weather: int = 1
    lgt_cond: int = 1
    man_coll: int = 0
    rel_road: int = 1
    typ_int: int = 1
    wrk_zone: int = 0
    sch_bus: int = 0
    speed_limit_mph: float = 55.0
    driver_impairment: int = 0
    vehicle_type: int = 0
    # Live Vision Features (Optional overrides)
    lane_offset_m: float = 0.0
    lane_curvature_rad_m: float = 0.001
    lane_confidence: float = 0.90
    min_ttc_sec: float = 6.5
    nearest_obj_distance_m: float = 45.0
    max_approach_rate_mps: float = 0.5
    traffic_density: int = 2
    vru_ratio: float = 0.0


# ── Web UI Root Endpoint ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_html = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_html):
        with open(index_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>DriveSense AI Dashboard</h1><p>web/index.html not found.</p>"


@app.get("/index.css")
async def serve_css():
    return FileResponse(os.path.join(WEB_DIR, "index.css"), media_type="text/css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(os.path.join(WEB_DIR, "app.js"), media_type="application/javascript")


@app.get("/data/real_road.mp4.f401.mp4")
async def serve_demo_video():
    road_path = os.path.join(DATA_DIR, "real_road.mp4.f401.mp4")
    if os.path.exists(road_path):
        return FileResponse(road_path, media_type="video/mp4")
    return JSONResponse({"error": "Video not found"}, status_code=404)

@app.get("/outputs/real_road_risk_web.mp4")
async def serve_output_file():
    file_path = os.path.join(OUTPUTS_DIR, "real_road_risk_web.mp4")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")
    return JSONResponse({"error": "File not found"}, status_code=404)

# ── Prediction Endpoint ────────────────────────────────────────────────────
@app.post("/predict")
def predict_risk(req: ScenarioPredictRequest):
    is_rush_hour = 1 if (7 <= req.hour <= 9) or (17 <= req.hour <= 19) else 0
    is_night = 1 if (req.hour >= 22) or (req.hour <= 5) else 0
    is_winter = 1 if req.month in [12, 1, 2] else 0
    is_weekend = 1 if req.day_week in [1, 7] else 0

    # 1. Construct Tabular Vector (9 features matching TABULAR_FEATURE_NAMES)
    road_alignment = 1 if req.typ_int in [2, 3, 4] or abs(req.lane_curvature_rad_m) > 0.01 else 0
    road_surface = 1 if req.weather in [2, 3, 4, 11, 12] else 0
    light_code = 0 if req.lgt_cond == 1 else (1 if req.lgt_cond == 3 else 2)
    weather_code = 0 if req.weather in [1, 10] else (1 if req.weather in [2, 5] else 2)
    hist_risk = 0.25 if is_rush_hour or is_night else 0.10

    tabular_vec = np.array([
        float(req.speed_limit_mph),
        float(road_alignment),
        float(road_surface),
        float(light_code),
        float(weather_code),
        1.0,  # Middle driver age
        float(req.driver_impairment),
        float(req.vehicle_type),
        float(hist_risk)
    ], dtype=np.float32)

    # 2. Construct Vision Vector (12 features matching VISION_FEATURE_NAMES)
    vision_vec = np.array([
        float(req.lane_offset_m),
        float(req.lane_curvature_rad_m),
        float(req.lane_confidence),
        float(np.clip(req.min_ttc_sec, 0.1, 10.0)),
        float(np.clip(req.nearest_obj_distance_m, 1.0, 100.0)),
        float(np.clip(req.max_approach_rate_mps, -20.0, 40.0)),
        float(req.traffic_density),
        float(np.clip(req.vru_ratio, 0.0, 1.0)),
        140.0 if req.lgt_cond == 1 else 45.0,  # luminance
        48.0,                                 # contrast
        0.05 if req.lgt_cond == 3 else 0.01,  # glare
        0.65 if req.weather in [2, 5] else 0.05 # rain/fog scatter
    ], dtype=np.float32)

    fused_vec = np.concatenate([tabular_vec, vision_vec])

    # 3. Model Inference based on requested modality
    try:
        if req.model_type == "A" and os.path.exists(model_a_path):
            prob = float(model_a.predict_proba(tabular_vec.reshape(1, -1))[0, 1])
        elif req.model_type == "B" and os.path.exists(model_b_path):
            prob = float(model_b.predict_proba(vision_vec.reshape(1, -1))[0, 1])
        elif os.path.exists(model_c_path):
            prob = float(model_c.predict_proba(fused_vec.reshape(1, -1))[0, 1])
        else:
            # Calibrated analytical risk fallback
            kinetic_risk = 0.55 * np.exp(-req.min_ttc_sec / 2.0)
            offset_risk = 0.25 * (abs(req.lane_offset_m) / 1.0)
            env_risk = 0.15 * (1.0 if is_night else 0.0) + 0.15 * (1.0 if req.weather > 1 else 0.0)
            prob = float(np.clip(0.10 + kinetic_risk + offset_risk + env_risk, 0.05, 0.98))
    except Exception:
        prob = 0.25

    risk_score = int(round(prob * 100))

    if risk_score >= 70:
        risk_level, risk_color, risk_label = "HIGH", "#EF4444", "High Risk"
    elif risk_score >= 40:
        risk_level, risk_color, risk_label = "MEDIUM", "#F59E0B", "Moderate Risk"
    else:
        risk_level, risk_color, risk_label = "LOW", "#10B981", "Low Risk"

    # 4. SHAP Factor Decomposition
    top_factors = []
    if explainer_c is not None and req.model_type == "C":
        try:
            explanation = explainer_c.explain_prediction(fused_vec)
            top_factors = explanation.get("top_3_factors", [])
        except Exception:
            pass

    if not top_factors:
        # Fallback dynamic explanation
        if req.min_ttc_sec < 3.0:
            top_factors.append({
                "feature": "min_ttc_sec", "human_name": "Time-to-Collision (TTC)",
                "value": req.min_ttc_sec, "shap_value": 0.35, "direction": "INCREASING_RISK",
                "description": f"Critical TTC of {req.min_ttc_sec:.1f}s to lead object"
            })
        if abs(req.lane_offset_m) > 0.4:
            top_factors.append({
                "feature": "lane_offset_m", "human_name": "Lane Lateral Deviation",
                "value": req.lane_offset_m, "shap_value": 0.22, "direction": "INCREASING_RISK",
                "description": f"Vehicle lateral offset {abs(req.lane_offset_m):.2f}m from center"
            })
        if is_night:
            top_factors.append({
                "feature": "light_condition", "human_name": "Low Ambient Lighting",
                "value": float(light_code), "shap_value": 0.15, "direction": "INCREASING_RISK",
                "description": "Nighttime conditions reduce driver reaction time"
            })

    # Log prediction
    record = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "probability": round(prob, 4),
        "model_type": req.model_type,
        "hour": req.hour,
        "weather": req.weather,
        "is_night": bool(is_night),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    prediction_logs.append(record)
    if len(prediction_logs) > 500:
        prediction_logs.pop(0)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_label": risk_label,
        "risk_color": risk_color,
        "probability": round(prob, 4),
        "factors": [f["description"] for f in top_factors],
        "top_3_factors": top_factors,
        "weather_label": WEATHER_LABELS.get(req.weather, "Unknown"),
        "light_label": LIGHT_LABELS.get(req.lgt_cond, "Unknown"),
        "collision_label": COLLISION_LABELS.get(req.man_coll, "Unknown"),
    }


# ── Computer Vision Image Upload Analysis ──────────────────────────────────
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Brightness → Lighting condition
    brightness = float(np.mean(gray))
    brightness_norm = brightness / 255.0

    if brightness_norm > 0.55:
        lgt_cond, light_label = 1, "Daylight"
        light_note = f"Bright scene ({brightness:.0f}/255)"
    elif brightness_norm > 0.35:
        lgt_cond, light_label = 5, "Dusk/Dawn"
        light_note = f"Low light ({brightness:.0f}/255)"
    elif brightness_norm > 0.18:
        lgt_cond, light_label = 3, "Dark – Lighted"
        light_note = f"Dark, street lighted ({brightness:.0f}/255)"
    else:
        lgt_cond, light_label = 2, "Dark – No Lighting"
        light_note = f"Very dark unlit ({brightness:.0f}/255)"

    # 2. Multi-Signal Weather Extraction
    mean_sat = float(np.mean(hsv[:, :, 1]))
    blue_ratio = float(np.mean(img[:, :, 0])) / (float(np.mean(img)) + 1e-5)

    gray_f = gray.astype(np.float32)
    local_mean = cv2.blur(gray_f, (15, 15))
    local_sq_mean = cv2.blur(gray_f ** 2, (15, 15))
    local_var = local_sq_mean - local_mean ** 2
    mean_local_var = float(np.mean(local_var))

    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_pixel_pct = float(np.sum(bright_mask > 0)) / (h * w)
    laplacian_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

    if mean_sat < 25 and mean_local_var < 400:
        weather, weather_label = 5, "Fog / Smog"
        weather_note = f"Low saturation ({mean_sat:.0f}) + low variance ({mean_local_var:.0f})"
    elif (mean_local_var > 800 and bright_pixel_pct > 0.08 and laplacian_var > 500):
        weather, weather_label = 2, "Rain"
        weather_note = f"High local variance ({mean_local_var:.0f}) + bright droplet spots"
    elif mean_sat < 55 and brightness_norm < 0.42:
        weather, weather_label = 2, "Rain"
        weather_note = f"Overcast/rainy dull scene ({brightness:.0f}/255)"
    elif blue_ratio > 1.12 and brightness_norm < 0.55:
        weather, weather_label = 10, "Cloudy"
        weather_note = f"Blue overcast cast ({blue_ratio:.2f})"
    else:
        weather, weather_label = 1, "Clear"
        weather_note = f"Clear ambient illumination ({brightness:.0f}/255)"

    # 3. Road Complexity & Intersection Proxy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    road_region = edges[int(h * 0.4):, :]
    edge_density = float(np.sum(road_region > 0)) / road_region.size

    if edge_density > 0.12:
        typ_int, road_label = 2, "Complex / Intersection"
        road_note = f"High edge density ({edge_density:.2%})"
    elif edge_density > 0.06:
        typ_int, road_label = 3, "Moderate complexity"
        road_note = f"Moderate edge density ({edge_density:.2%})"
    else:
        typ_int, road_label = 1, "Open Road"
        road_note = f"Low edge density ({edge_density:.2%})"

    # 4. Generate Annotated Image
    annotated = img.copy()
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    for i, (text, col) in enumerate([
        (f"Light:   {light_label}", (255, 255, 255)),
        (f"Weather: {weather_label}", (180, 210, 255)),
        (f"Road:    {road_label}", (160, 255, 180)),
    ]):
        cv2.putText(annotated, text, (14, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "autofill": {
            "weather": weather,
            "lgt_cond": lgt_cond,
            "typ_int": typ_int,
        },
        "analysis": {
            "light": {"label": light_label, "note": light_note, "value": lgt_cond},
            "weather": {"label": weather_label, "note": weather_note, "value": weather},
            "road": {"label": road_label, "note": road_note, "value": typ_int},
            "brightness_raw": round(brightness, 1),
            "edge_density": round(edge_density, 4),
        },
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
    }


# ── Telemetry & Analytics Endpoints ────────────────────────────────────────
@app.get("/api/telemetry")
def get_telemetry():
    telemetry_file = os.path.join(OUTPUTS_DIR, "demo_telemetry.json")
    if os.path.exists(telemetry_file):
        with open(telemetry_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/analytics")
def get_analytics():
    records = prediction_logs if prediction_logs else [
        {"risk_score": 18, "risk_level": "LOW", "hour": 14, "weather": 1, "is_night": False},
        {"risk_score": 68, "risk_level": "MEDIUM", "hour": 23, "weather": 2, "is_night": True},
        {"risk_score": 84, "risk_level": "HIGH", "hour": 2, "weather": 5, "is_night": True}
    ]

    scores = [r["risk_score"] for r in records]
    levels = [r["risk_level"] for r in records]

    hourly = {}
    for r in records:
        h = str(r.get("hour", 0))
        hourly.setdefault(h, []).append(r["risk_score"])
    hourly_avg = {h: round(sum(v) / len(v)) for h, v in hourly.items()}

    weather_risk = {}
    for r in records:
        w = WEATHER_LABELS.get(r.get("weather", 1), "Unknown")
        weather_risk.setdefault(w, []).append(r["risk_score"])
    weather_avg = {w: round(sum(v) / len(v)) for w, v in weather_risk.items()}

    return {
        "total": len(records),
        "avg_risk_score": round(sum(scores) / max(1, len(scores))),
        "high_risk_count": levels.count("HIGH"),
        "medium_risk_count": levels.count("MEDIUM"),
        "low_risk_count": levels.count("LOW"),
        "hourly_avg_risk": hourly_avg,
        "weather_avg_risk": weather_avg,
        "recent": records[-10:]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": {
            "model_a_loaded": os.path.exists(model_a_path),
            "model_b_loaded": os.path.exists(model_b_path),
            "model_c_loaded": os.path.exists(model_c_path)
        },
        "server_time": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting DriveSense AI Unified Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
