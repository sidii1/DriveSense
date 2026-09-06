"""
Model Explainability Engine using Native Ultra-Fast TreeSHAP.

Provides:
1. Native C++ TreeSHAP exact feature attributions via XGBoost (pred_contribs=True)
2. Global SHAP Summary Plot generation (outputs/shap_summary.png)
3. Real-Time Local Inference Explanation:
   Extracts Top-3 Risk Drivers per frame with positive/negative attribution and human-readable text.

NOTE: feature_names is configurable per instance (see __init__) so this same
class can explain Model A (tabular-only), Model B (vision-only), and
Model C (fused) — not just the fused 21-feature schema it originally
assumed.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from typing import List, Dict, Any, Tuple

from feature_schema import FUSED_FEATURE_NAMES


class RiskExplainer:
    # Human-friendly descriptions and units for features
    FEATURE_HUMAN_NAMES = {
        "speed_limit_mph": "Speed Limit",
        "road_alignment": "Curved Road Geometry",
        "road_surface_condition": "Wet/Slippery Surface",
        "light_condition": "Low Ambient Light",
        "weather_condition": "Adverse Weather",
        "driver_age_group": "Driver Age Group Risk",
        "driver_impairment_flag": "Driver Impairment / Distraction",
        "vehicle_type": "Vehicle Mass/Class Risk",
        "historical_segment_risk": "High Crash History Zone",
        "lane_offset_m": "Lane Departure Offset",
        "lane_curvature_rad_m": "Sharp Road Curvature",
        "lane_confidence": "Poor Lane Marking Visibility",
        "min_ttc_sec": "Time-to-Collision (TTC)",
        "nearest_obj_distance_m": "Close Proximity to Lead Object",
        "max_approach_rate_mps": "Rapid Object Approach Rate",
        "traffic_density": "High Traffic Density",
        "vru_ratio": "Dense Vulnerable Road Users (VRUs)",
        "scene_luminance_mean": "Dark Scene Illumination",
        "scene_contrast_rms": "Low Scene Contrast",
        "glare_index": "Severe Headlight/Sun Glare",
        "rain_fog_scatter_index": "Rain/Fog Haze Scattering"
    }

    def __init__(self, model_path: str = "models/model_c_fused.json", feature_names: List[str] = None):
        self.model = xgb.XGBClassifier()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            print(f"[INFO] RiskExplainer loaded model from: {model_path}")
        else:
            print(f"[WARN] Model not found at {model_path}. Explainer will require model training first.")

        self.booster = self.model.get_booster()

        # FIX: feature_names is now configurable per model instead of being
        # hardcoded to the 21-feature fused schema. This lets the same class
        # explain Model A (9 tabular features) and Model B (12 vision
        # features) correctly, not just Model C. Defaults to
        # FUSED_FEATURE_NAMES so existing Model C call sites are unaffected.
        self.feature_names = feature_names if feature_names is not None else FUSED_FEATURE_NAMES

    def generate_global_summary_plot(self, df_sample: pd.DataFrame, output_path: str = "outputs/shap_summary.png"):
        """Computes global TreeSHAP values and saves a publication-ready importance plot."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print("[INFO] Computing global TreeSHAP feature attributions...")

        dmat = xgb.DMatrix(df_sample[self.feature_names], feature_names=self.feature_names)
        contribs = self.booster.predict(dmat, pred_contribs=True)[:, :-1]
        mean_abs_shap = np.mean(np.abs(contribs), axis=0)

        # Sort features by mean |SHAP|
        sorted_indices = np.argsort(mean_abs_shap)
        sorted_features = [self.FEATURE_HUMAN_NAMES.get(self.feature_names[i], self.feature_names[i]) for i in sorted_indices]
        sorted_vals = mean_abs_shap[sorted_indices]

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        bars = ax.barh(sorted_features[-12:], sorted_vals[-12:], color="#2ECC71", edgecolor="#27AE60", height=0.65)

        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Log-Odds Risk)", fontsize=11, fontweight="bold", color="#FFFFFF")
        ax.set_title("Top Global Risk Drivers", fontsize=13, fontweight="bold", pad=12, color="#FFFFFF")
        ax.grid(axis="x", linestyle="--", alpha=0.25)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", ha="left", fontsize=9, fontweight="bold", color="#ECF0F1")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[INFO] Global SHAP summary plot saved to: {output_path}")

    def explain_prediction(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Ultra-fast local SHAP explanation for a single 1D or 2D feature vector (< 1ms).
        `feature_vector` must have the same length/order as self.feature_names
        (i.e. tabular_vec for a Model-A explainer, vision_vec for Model B,
        fused_vec for Model C).
        """
        vec_2d = feature_vector.reshape(1, -1)

        if vec_2d.shape[1] != len(self.feature_names):
            raise ValueError(
                f"[RiskExplainer] feature_vector has {vec_2d.shape[1]} values but this "
                f"explainer was configured with {len(self.feature_names)} feature_names "
                f"({self.feature_names}). Pass the matching vector for this model."
            )

        prob = float(self.model.predict_proba(vec_2d)[0, 1])

        # Native C++ exact TreeSHAP computation
        dmat = xgb.DMatrix(vec_2d, feature_names=self.feature_names)
        shap_vals = self.booster.predict(dmat, pred_contribs=True)[0, :-1]

        # Rank features by absolute magnitude of SHAP contribution
        ranked_indices = np.argsort(np.abs(shap_vals))[::-1]

        top_factors = []
        for idx in ranked_indices[:3]:
            feat_name = self.feature_names[idx]
            val = float(feature_vector[idx])
            s_val = float(shap_vals[idx])
            human_name = self.FEATURE_HUMAN_NAMES.get(feat_name, feat_name)
            direction = "INCREASING_RISK" if s_val > 0 else "DECREASING_RISK"

            desc = self._format_factor_description(feat_name, val, s_val)

            top_factors.append({
                "feature": feat_name,
                "human_name": human_name,
                "value": round(val, 2),
                "shap_value": round(s_val, 4),
                "direction": direction,
                "description": desc
            })

        # Risk severity classification
        if prob >= 0.70:
            level = "CRITICAL"
        elif prob >= 0.45:
            level = "HIGH"
        elif prob >= 0.25:
            level = "MODERATE"
        else:
            level = "LOW"

        return {
            "risk_score": round(prob, 4),
            "risk_percentage": round(prob * 100, 1),
            "risk_level": level,
            "top_3_factors": top_factors
        }

    def _format_factor_description(self, feat: str, val: float, s_val: float) -> str:
        if feat == "min_ttc_sec":
            return f"Closing time-to-collision is {val:.1f}s ({'+' if s_val>0 else ''}{s_val:.2f} risk impact)"
        elif feat == "lane_offset_m":
            side = "left" if val < 0 else "right"
            return f"Vehicle lateral offset {abs(val):.2f}m to {side} lane line"
        elif feat == "max_approach_rate_mps":
            return f"Lead obstacle closing velocity is {val:.1f} m/s"
        elif feat == "vru_ratio":
            return f"High vulnerable road user concentration ({val*100:.0f}% of objects)"
        elif feat == "traffic_density":
            return f"{int(val)} active objects detected in forward path"
        elif feat == "lane_confidence":
            return f"Lane marking confidence is low ({val*100:.0f}%)"
        elif feat == "glare_index":
            return f"High optical glare ({val*100:.1f}% saturated pixels)"
        elif feat == "historical_segment_risk":
            return f"Roadway segment baseline crash risk is {val*100:.0f}%"
        else:
            return f"{self.FEATURE_HUMAN_NAMES.get(feat, feat)}: value {val:.1f}"


if __name__ == "__main__":
    from fusion_model import generate_multimodal_dataset
    df = generate_multimodal_dataset(n_samples=2000, seed=42)
    explainer = RiskExplainer("models/model_c_fused.json")
    explainer.generate_global_summary_plot(df, "outputs/shap_summary.png")