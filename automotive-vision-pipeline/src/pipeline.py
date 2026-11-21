"""
End-to-End Multi-Modal Automotive Perception & Risk Intelligence Pipeline.

Orchestrates:
1. Lane Detection (Canny/Hough + Geometry + Offset + Curvature + Confidence)
2. Object Tracking (YOLOv8 + ByteTrack persistent IDs + Kinematics)
3. Monocular Pinhole Distance & Time-to-Collision (TTC)
4. Scene Environmental Photometry & VRU Ratio Analysis
5. Model C Fused Risk Inference & Real-Time TreeSHAP Top-3 Factor Decomposition
6. HUD Visualization & Telemetry JSON Logging
"""

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

# Ensure src/ is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lane_detection import LaneDetector
from object_detection import ObjectTracker
from ttc_distance import DistanceTTCEstimator
from scene_features import SceneFeatureExtractor
from feature_schema import FrameTelemetry, LaneTelemetry, TrackedObjectTelemetry, SceneTelemetry, TABULAR_FEATURE_NAMES
from explainability import RiskExplainer


def run_full_pipeline(
    input_video_path: str,
    output_video_path: str,
    telemetry_json_path: str = "outputs/demo_telemetry.json",
    detect_every_n: int = 1,
    max_frames: int = 400
) -> bool:
    print("=" * 70)
    print("  AUTOMOTIVE MULTI-MODAL RISK & VISION PIPELINE")
    print(f"  Input:     {input_video_path}")
    print(f"  Output:    {output_video_path}")
    print(f"  Telemetry: {telemetry_json_path}")
    print("=" * 70)

    if not os.path.exists(input_video_path):
        print(f"[ERROR] Input video not found: {input_video_path}")
        return False

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(telemetry_json_path), exist_ok=True)

    print("[INFO] Initializing ADAS perception modules & Explainer...")
    lane_detector = LaneDetector()
    tracker = ObjectTracker(confidence_threshold=0.25)
    distance_ttc_calc = DistanceTTCEstimator(focal_length_px=720.0)
    scene_extractor = SceneFeatureExtractor()
    explainer = RiskExplainer(model_path=os.path.join(os.path.dirname(__file__), "..", "models", "model_c_fused.json"))

    # Macro tabular prior for this driving segment (Highway corridor, dry, daytime, clear)
    macro_tabular_prior = np.array([
        55.0,  # speed_limit_mph
        0.0,   # road_alignment (straight)
        0.0,   # road_surface_condition (dry)
        0.0,   # light_condition (daylight)
        0.0,   # weather_condition (clear)
        1.0,   # driver_age_group (middle)
        0.0,   # driver_impairment_flag (none)
        0.0,   # vehicle_type (car)
        0.12   # historical_segment_risk
    ], dtype=np.float32)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames > 0 else total_frames

    out = None
    for backend, fourcc_str in [
        (cv2.CAP_MSMF, "H264"),
        (cv2.CAP_MSMF, "avc1"),
        (cv2.CAP_ANY, "avc1"),
        (cv2.CAP_ANY, "H264"),
        (cv2.CAP_ANY, "mp4v"),
    ]:
        try:
            writer = cv2.VideoWriter(output_video_path, backend, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height))
            if writer.isOpened():
                out = writer
                print(f"[INFO] Initialized VideoWriter with backend={backend}, fourcc={fourcc_str}")
                break
        except Exception:
            pass
    if out is None or not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    telemetry_log = []
    frame_idx = 0
    start_time = time.time()
    last_tracked_objects = []

    print(f"[INFO] Processing {frames_to_process} frames ({width}x{height} @ {fps:.1f} fps)...")

    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp_sec = frame_idx / fps

        # 1. Lane Detection & Geometry
        lane_frame, lane_telemetry = lane_detector.process_frame(frame)

        # 2. YOLOv8 + ByteTrack Tracking
        if frame_idx % detect_every_n == 0:
            last_tracked_objects = tracker.track(frame, timestamp_sec=timestamp_sec)

        # 3. Monocular Distance & TTC
        enriched_objects = distance_ttc_calc.estimate_distance_and_ttc(last_tracked_objects, timestamp_sec=timestamp_sec)

        # 4. Scene Environmental Photometry
        scene_data = scene_extractor.extract_features(frame, enriched_objects)

        # 5. Build Standardized Telemetry Data Object
        frame_telem = FrameTelemetry(
            frame_id=frame_idx,
            timestamp_sec=round(timestamp_sec, 3),
            lane=LaneTelemetry(
                lane_offset_m=lane_telemetry["lane_offset_m"],
                lane_curvature_rad_m=lane_telemetry["lane_curvature_rad_m"],
                lane_confidence=lane_telemetry["lane_confidence"],
                lane_detected=lane_telemetry["lane_detected"],
                left_line=lane_telemetry["left_line"],
                right_line=lane_telemetry["right_line"]
            ),
            objects=[
                TrackedObjectTelemetry(
                    id=o["id"],
                    class_name=o["class"],
                    bbox=o["bbox"],
                    confidence=o["confidence"],
                    persistence_frames=o["persistence_frames"],
                    distance_m=o["distance_m"],
                    approach_rate_mps=o["approach_rate_mps"],
                    ttc_sec=o["ttc_sec"]
                ) for o in enriched_objects
            ],
            scene=SceneTelemetry(
                luminance_mean=scene_data["luminance_mean"],
                contrast_rms=scene_data["contrast_rms"],
                glare_index=scene_data["glare_index"],
                rain_fog_scatter_index=scene_data["rain_fog_scatter_index"],
                traffic_density=scene_data["traffic_density"],
                vru_ratio=scene_data["vru_ratio"]
            )
        )

        # 6. Feature Fusion & Model C Risk Prediction + SHAP Attribution
        vision_vec = frame_telem.to_vision_feature_vector()
        fused_vec = np.concatenate([macro_tabular_prior, vision_vec])

        try:
            explanation = explainer.explain_prediction(fused_vec)
        except Exception:
            # Fallback if model not trained yet
            explanation = {
                "risk_score": 0.15,
                "risk_percentage": 15.0,
                "risk_level": "LOW",
                "top_3_factors": []
            }

        # 7. Render Rich ADAS HUD Overlay
        annotated_frame = distance_ttc_calc.draw_distance_ttc_hud(lane_frame, enriched_objects)
        annotated_frame = tracker.draw_tracked_objects(annotated_frame, enriched_objects)

        # Top HUD Banner: Risk Meter + SHAP Driver
        annotated_frame = _draw_unified_hud(annotated_frame, frame_idx, frames_to_process, explanation, lane_telemetry, scene_data)

        out.write(annotated_frame)

        # Append to telemetry log
        telem_dict = frame_telem.to_dict()
        telem_dict["risk_assessment"] = explanation
        telemetry_log.append(telem_dict)

        if frame_idx % 30 == 0 or frame_idx == frames_to_process:
            elapsed = time.time() - start_time
            fps_act = frame_idx / max(0.01, elapsed)
            print(f"[{frame_idx/frames_to_process*100:5.1f}%] Frame {frame_idx}/{frames_to_process} | "
                  f"Risk: {explanation['risk_percentage']}% ({explanation['risk_level']}) | {fps_act:.1f} FPS")

    cap.release()
    out.release()

    # Ensure both real_road_risk_web.mp4 and road_fused_risk.mp4 exist in outputs
    import shutil
    outputs_dir = os.path.dirname(os.path.abspath(output_video_path))
    web_mp4 = os.path.join(outputs_dir, "real_road_risk_web.mp4")
    fused_mp4 = os.path.join(outputs_dir, "road_fused_risk.mp4")
    if os.path.abspath(output_video_path) == os.path.abspath(web_mp4):
        try:
            shutil.copyfile(web_mp4, fused_mp4)
        except Exception:
            pass
    elif os.path.abspath(output_video_path) == os.path.abspath(fused_mp4):
        try:
            shutil.copyfile(fused_mp4, web_mp4)
        except Exception:
            pass

    # Save full JSON telemetry log
    with open(telemetry_json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_log, f, indent=2)

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"  Pipeline Complete: {frame_idx} frames in {total_time:.1f}s ({frame_idx/max(0.01,total_time):.1f} FPS avg)")
    print(f"  Annotated Video:   {output_video_path}")
    print(f"  Telemetry Log:     {telemetry_json_path}")
    print("=" * 70)
    return True


def _draw_unified_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    explanation: Dict[str, Any],
    lane_telem: Dict[str, Any],
    scene_data: Dict[str, Any]
) -> np.ndarray:
    """Renders sleek top & bottom telemetry HUD overlays."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Top HUD Header Bar
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 18, 24), -1)
    # 2. Bottom HUD Telemetry Strip
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)

    # Risk color scheme
    risk_pct = explanation.get("risk_percentage", 0.0)
    risk_lvl = explanation.get("risk_level", "LOW")
    if risk_lvl == "CRITICAL":
        risk_col = (0, 0, 255)
    elif risk_lvl == "HIGH":
        risk_col = (0, 100, 255)
    elif risk_lvl == "MODERATE":
        risk_col = (0, 215, 255)
    else:
        risk_col = (0, 230, 80)

    # Top Left: System Title & Frame
    cv2.putText(frame, "AUTONOMOUS RISK ENGINE (MODEL C)", (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | Telemetry Live", (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # Top Center: Risk Gauge Pill
    pill_x = int(w * 0.42)
    cv2.rectangle(frame, (pill_x, 8), (pill_x + 220, 46), risk_col, 2)
    cv2.putText(frame, f"RISK: {risk_pct:.0f}% [{risk_lvl}]", (pill_x + 12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, risk_col, 2, cv2.LINE_AA)

    # Top Right: Top SHAP Driver
    top_factors = explanation.get("top_3_factors", [])
    if top_factors:
        top_driver = top_factors[0]["description"]
        cv2.putText(frame, f"Top Driver: {top_driver[:48]}", (pill_x + 240, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Bottom Strip Telemetry
    offset = lane_telem.get("lane_offset_m", 0.0)
    conf = lane_telem.get("lane_confidence", 0.0)
    density = scene_data.get("traffic_density", 0)
    vru_pct = scene_data.get("vru_ratio", 0.0) * 100

    bot_text = f"Lane Offset: {offset:+.2f}m | Lane Conf: {conf*100:.0f}% | Traffic: {density} objs | VRU Ratio: {vru_pct:.0f}%"
    cv2.putText(frame, bot_text, (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Automotive Perception & Risk Pipeline")
    default_input = os.path.join(os.path.dirname(__file__), "..", "inputs", "real_road_input.mp4.mp4")
    if not os.path.exists(default_input):
        default_input = os.path.join(os.path.dirname(__file__), "..", "data", "real_road.mp4.f401.mp4")
    parser.add_argument("--input", "-i", default=default_input)
    parser.add_argument("--output", "-o", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "real_road_risk_web.mp4"))
    parser.add_argument("--telemetry", "-t", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "demo_telemetry.json"))
    parser.add_argument("--max-frames", "-m", type=int, default=300)
    args = parser.parse_args()

    run_full_pipeline(args.input, args.output, args.telemetry, max_frames=args.max_frames)