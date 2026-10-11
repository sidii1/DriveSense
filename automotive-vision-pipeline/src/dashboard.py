"""
DriveSense AI — Unified Multi-Modal Automotive Risk & Vision Intelligence Server.

Serves:
  GET  /                  dashboard UI
  GET  /api/telemetry     per-frame telemetry from the last pipeline.py run
  POST /predict           Model A / B / C risk + SHAP top-3 factors
  POST /analyze-image     weather / lighting / road-complexity from an uploaded photo
  GET  /analytics, /health
  /outputs/*              static files (clean dashboard video, annotated video, telemetry)
"""

import os
import sys
import json
import base64
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_schema import TABULAR_FEATURE_NAMES, VISION_FEATURE_NAMES
from explainability import RiskExplainer


app = FastAPI(title="DriveSense AI — Multi-Modal Driving Risk Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
WEB_DIR = os.path.join(ROOT_DIR, "web")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# Dev server: never let the browser cache the HTML/JS/CSS/telemetry
NO_CACHE = {"Cache-Control": "no-store, max-age=0"}

# Serve everything in outputs/ (clean video, annotated video, telemetry json)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ── Models & explainers (one explainer per model, so A/B/C all get SHAP) ────
model_a_path = os.path.join(MODELS_DIR, "model_a_baseline.json")
model_b_path = os.path.join(MODELS_DIR, "model_b_vision.json")
model_c_path = os.path.join(MODELS_DIR, "model_c_fused.json")


def load_explainer(path: str, feature_names: Optional[List[str]] = None):
    if not os.path.exists(path):
        print(f"[WARN] Model file missing: {path}")
        return None
    try:
        return RiskExplainer(path, feature_names=feature_names)
    except Exception as e:
        print(f"[WARN] Could not load explainer for {path}: {e}")
        return None


EXPLAINERS = {
    "A": load_explainer(model_a_path, TABULAR_FEATURE_NAMES),
    "B": load_explainer(model_b_path, VISION_FEATURE_NAMES),
    "C": load_explainer(model_c_path),  # defaults to the fused 21-feature schema
}

prediction_logs: List[Dict[str, Any]] = []

# ── Label maps ──────────────────────────────────────────────────────────────
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

# One risk scheme for every model (same thresholds RiskExplainer uses)
LEVEL_STYLE = {
    "LOW":      ("Low Risk",      "#10B981"),
    "MODERATE": ("Moderate Risk", "#F59E0B"),
    "HIGH":     ("High Risk",     "#EF4444"),
    "CRITICAL": ("Critical Risk", "#DC2626"),
}


def classify(prob: float) -> str:
    if prob >= 0.70:
        return "CRITICAL"
    if prob >= 0.45:
        return "HIGH"
    if prob >= 0.25:
        return "MODERATE"
    return "LOW"


# ── Request model ───────────────────────────────────────────────────────────
class ScenarioPredictRequest(BaseModel):
    model_type: str = "C"           # "A", "B", "C"
    # Tabular context
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
    # Live vision features
    lane_offset_m: float = 0.0
    lane_curvature_rad_m: float = 0.001
    lane_confidence: float = 0.90
    min_ttc_sec: float = 6.5
    nearest_obj_distance_m: float = 45.0
    max_approach_rate_mps: float = 0.5
    traffic_density: int = 2
    vru_ratio: float = 0.0
    # Real scene photometry (optional; derived from lighting/weather if omitted)
    luminance_mean: Optional[float] = None
    contrast_rms: Optional[float] = None
    glare_index: Optional[float] = None
    rain_fog_scatter_index: Optional[float] = None


# ── UI routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_html = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_html):
        with open(index_html, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read(), headers=NO_CACHE)
    return HTMLResponse("<h1>DriveSense AI Dashboard</h1><p>web/index.html not found.</p>")


@app.get("/index.css")
async def serve_css():
    return FileResponse(os.path.join(WEB_DIR, "index.css"), media_type="text/css", headers=NO_CACHE)


@app.get("/app.js")
async def serve_js():
    return FileResponse(os.path.join(WEB_DIR, "app.js"), media_type="application/javascript", headers=NO_CACHE)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# ── Prediction ──────────────────────────────────────────────────────────────
@app.post("/predict")
def predict_risk(req: ScenarioPredictRequest):
    is_rush_hour = 1 if (7 <= req.hour <= 9) or (17 <= req.hour <= 19) else 0
    is_night = 1 if (req.hour >= 22) or (req.hour <= 5) else 0

    # 1. Tabular vector (order = TABULAR_FEATURE_NAMES)
    road_alignment = 1 if req.typ_int in [2, 3, 4] or abs(req.lane_curvature_rad_m) > 0.01 else 0
    road_surface = 1 if req.weather in [2, 3, 4, 11, 12] else 0
    light_code = 0 if req.lgt_cond == 1 else (1 if req.lgt_cond == 3 else 2)
    weather_code = 0 if req.weather in [1, 10] else (1 if req.weather in [2, 5] else 2)
    hist_risk = 0.25 if (is_rush_hour or is_night) else 0.12   # 0.12 = pipeline.py prior

    tabular_vec = np.array([
        float(req.speed_limit_mph),
        float(road_alignment),
        float(road_surface),
        float(light_code),
        float(weather_code),
        1.0,                                  # middle driver age group
        float(req.driver_impairment),
        float(req.vehicle_type),
        float(hist_risk),
    ], dtype=np.float32)

    # 2. Vision vector (order = VISION_FEATURE_NAMES)
    lum = req.luminance_mean if req.luminance_mean is not None else (140.0 if req.lgt_cond == 1 else 45.0)
    con = req.contrast_rms if req.contrast_rms is not None else 48.0
    glare = req.glare_index if req.glare_index is not None else (0.05 if req.lgt_cond == 3 else 0.01)
    scatter = (req.rain_fog_scatter_index if req.rain_fog_scatter_index is not None
               else (0.65 if req.weather in [2, 5] else 0.05))

    vision_vec = np.array([
        float(req.lane_offset_m),
        float(req.lane_curvature_rad_m),
        float(req.lane_confidence),
        float(np.clip(req.min_ttc_sec, 0.1, 10.0)),
        float(np.clip(req.nearest_obj_distance_m, 1.0, 100.0)),
        float(np.clip(req.max_approach_rate_mps, -20.0, 40.0)),
        float(req.traffic_density),
        float(np.clip(req.vru_ratio, 0.0, 1.0)),
        float(lum),
        float(con),
        float(glare),
        float(scatter),
    ], dtype=np.float32)

    fused_vec = np.concatenate([tabular_vec, vision_vec])

    # 3. Inference — one code path for all three models
    model_key = req.model_type if req.model_type in ("A", "B") else "C"
    vec = {"A": tabular_vec, "B": vision_vec, "C": fused_vec}[model_key]
    explainer = EXPLAINERS.get(model_key)

    prob: Optional[float] = None
    top_factors: List[Dict[str, Any]] = []
    fallback = False

    if explainer is not None:
        try:
            ex = explainer.explain_prediction(vec)
            prob = float(ex["risk_score"])            # 0..1
            top_factors = ex["top_3_factors"]
        except Exception as e:
            print(f"[WARN] SHAP explanation failed for Model {model_key}: {e}")
            try:
                prob = float(explainer.model.predict_proba(vec.reshape(1, -1))[0, 1])
            except Exception as e2:
                print(f"[WARN] predict_proba failed for Model {model_key}: {e2}")

    if prob is None:
        # Honest analytical fallback — flagged in the response, never passed off as the model
        fallback = True
        kinetic = 0.55 * np.exp(-req.min_ttc_sec / 2.0)
        offset = 0.25 * (abs(req.lane_offset_m) / 1.0)
        env = 0.15 * (1.0 if is_night else 0.0) + 0.15 * (1.0 if req.weather > 1 else 0.0)
        prob = float(np.clip(0.10 + kinetic + offset + env, 0.05, 0.98))
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

    risk_score = int(round(prob * 100))
    risk_level = classify(prob)
    risk_label, risk_color = LEVEL_STYLE[risk_level]

    prediction_logs.append({
        "risk_score": risk_score,
        "risk_level": risk_level,
        "probability": round(prob, 4),
        "model_type": model_key,
        "hour": req.hour,
        "weather": req.weather,
        "is_night": bool(is_night),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    if len(prediction_logs) > 500:
        prediction_logs.pop(0)

    return {
        "risk_score": risk_score,                 # integer percentage, 0-100
        "risk_level": risk_level,                 # LOW / MODERATE / HIGH / CRITICAL
        "risk_label": risk_label,
        "risk_color": risk_color,
        "probability": round(prob, 4),
        "model_used": model_key,
        "fallback": fallback,
        "factors": [f["description"] for f in top_factors],
        "top_3_factors": top_factors,
        "weather_label": WEATHER_LABELS.get(req.weather, "Unknown"),
        "light_label": LIGHT_LABELS.get(req.lgt_cond, "Unknown"),
        "collision_label": COLLISION_LABELS.get(req.man_coll, "Unknown"),
    }


# ── Image upload analysis ───────────────────────────────────────────────────
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

    # 1. Brightness → lighting condition
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

    # 2. Multi-signal weather extraction
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
    elif mean_local_var > 800 and bright_pixel_pct > 0.08 and laplacian_var > 500:
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

    # 3. Road complexity / intersection proxy
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

    # 4. Annotated image
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
        "autofill": {"weather": weather, "lgt_cond": lgt_cond, "typ_int": typ_int},
        "analysis": {
            "light": {"label": light_label, "note": light_note, "value": lgt_cond},
            "weather": {"label": weather_label, "note": weather_note, "value": weather},
            "road": {"label": road_label, "note": road_note, "value": typ_int},
            "brightness_raw": round(brightness, 1),
            "edge_density": round(edge_density, 4),
        },
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
    }


# ── Telemetry, analytics, health ────────────────────────────────────────────
@app.get("/api/telemetry")
def get_telemetry():
    telemetry_file = os.path.join(OUTPUTS_DIR, "demo_telemetry.json")
    if os.path.exists(telemetry_file):
        with open(telemetry_file, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f), headers=NO_CACHE)
    return JSONResponse(content=[], headers=NO_CACHE)


@app.get("/analytics")
def get_analytics():
    records = prediction_logs if prediction_logs else [
        {"risk_score": 18, "risk_level": "LOW", "hour": 14, "weather": 1, "is_night": False},
        {"risk_score": 52, "risk_level": "HIGH", "hour": 23, "weather": 2, "is_night": True},
        {"risk_score": 84, "risk_level": "CRITICAL", "hour": 2, "weather": 5, "is_night": True},
    ]

    scores = [r["risk_score"] for r in records]
    levels = [r["risk_level"] for r in records]

    hourly: Dict[str, List[int]] = {}
    for r in records:
        hourly.setdefault(str(r.get("hour", 0)), []).append(r["risk_score"])
    hourly_avg = {k: round(sum(v) / len(v)) for k, v in hourly.items()}

    weather_risk: Dict[str, List[int]] = {}
    for r in records:
        name = WEATHER_LABELS.get(r.get("weather", 1), "Unknown")
        weather_risk.setdefault(name, []).append(r["risk_score"])
    weather_avg = {k: round(sum(v) / len(v)) for k, v in weather_risk.items()}

    return {
        "total": len(records),
        "avg_risk_score": round(sum(scores) / max(1, len(scores))),
        "critical_risk_count": levels.count("CRITICAL"),
        "high_risk_count": levels.count("HIGH"),
        "moderate_risk_count": levels.count("MODERATE"),
        "low_risk_count": levels.count("LOW"),
        "hourly_avg_risk": hourly_avg,
        "weather_avg_risk": weather_avg,
        "recent": records[-10:],
    }


@app.get("/health")
def health():
    tele_path = os.path.join(OUTPUTS_DIR, "demo_telemetry.json")
    tele_frames = None
    if os.path.exists(tele_path):
        try:
            with open(tele_path, "r", encoding="utf-8") as f:
                tele_frames = len(json.load(f))
        except Exception:
            tele_frames = -1
    return {
        "status": "healthy",
        "models": {k: (v is not None) for k, v in EXPLAINERS.items()},
        "clean_video_present": os.path.exists(os.path.join(OUTPUTS_DIR, "real_road_risk_web.mp4")),
        "telemetry_frames": tele_frames,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting DriveSense AI Unified Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)