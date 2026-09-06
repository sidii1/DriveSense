"""
Integration Test Script for Unified DriveSense AI Dashboard Server.
"""

import sys
import os
import json

# Ensure src/ in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dashboard import app
from fastapi.testclient import TestClient

client = TestClient(app)

print("=" * 60)
print("  TESTING UNIFIED DRIVESENSE AI DASHBOARD ENDPOINTS")
print("=" * 60)

# 1. Health Check
res = client.get("/health")
print(f"1. /health -> Status: {res.status_code}")
print("   Response:", res.json())
assert res.status_code == 200

# 2. Predict Endpoint (Model C Multi-Modal)
payload_c = {
    "model_type": "C",
    "hour": 23,
    "month": 12,
    "day_week": 6,
    "weather": 2,
    "lgt_cond": 2,
    "speed_limit_mph": 65.0,
    "lane_offset_m": -0.45,
    "min_ttc_sec": 1.8,
    "nearest_obj_distance_m": 14.0,
    "max_approach_rate_mps": 7.8,
    "traffic_density": 3,
    "vru_ratio": 0.33
}
res_c = client.post("/predict", json=payload_c)
print(f"\n2. /predict (Model C - High Risk Scenario) -> Status: {res_c.status_code}")
data_c = res_c.json()
print(f"   Risk Score: {data_c['risk_score']}% ({data_c['risk_level']})")
print("   Top Factors:", json.dumps(data_c.get("top_3_factors", []), indent=4))
assert res_c.status_code == 200

# 3. Predict Endpoint (Model A Crash Baseline)
payload_a = dict(payload_c)
payload_a["model_type"] = "A"
res_a = client.post("/predict", json=payload_a)
print(f"\n3. /predict (Model A - Crash Baseline) -> Status: {res_a.status_code}")
data_a = res_a.json()
print(f"   Risk Score: {data_a['risk_score']}% ({data_a['risk_level']})")
assert res_a.status_code == 200

# 4. Telemetry Endpoint
res_telem = client.get("/api/telemetry")
print(f"\n4. /api/telemetry -> Status: {res_telem.status_code}")
telem_list = res_telem.json()
print(f"   Loaded {len(telem_list)} frame records from demo_telemetry.json")
assert res_telem.status_code == 200

# 5. Analytics Endpoint
res_analytics = client.get("/analytics")
print(f"\n5. /analytics -> Status: {res_analytics.status_code}")
print("   Analytics Data:", res_analytics.json())
assert res_analytics.status_code == 200

print("\n" + "=" * 60)
print("  ALL DASHBOARD BACKEND ENDPOINTS PASSED SUCCESSFULLY!")
print("=" * 60)
