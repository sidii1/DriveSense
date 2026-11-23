"""
Multi-Modal Risk Fusion Architecture & 3-Way Benchmark (Models A, B, C).

This script:
1. Generates a synchronized multi-modal dataset combining NHTSA tabular priors with dynamic vision features.
2. Trains:
   - Model A: Tabular Crash Baseline (NHTSA features only)
   - Model B: Vision-Only Model (Vision & scene features only)
   - Model C: Multi-Modal Fused Model (Tabular + Vision features)
3. Evaluates all three models on the same test split across standard metrics:
   Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC.
4. Generates:
   - docs/model_comparison_report.md (3-row benchmark table)
   - outputs/model_comparison_bars.png (Publication-ready comparative chart)
   - models/model_b_vision.json and models/model_c_fused.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_schema import TABULAR_FEATURE_NAMES, VISION_FEATURE_NAMES, FUSED_FEATURE_NAMES


def generate_multimodal_dataset(n_samples: int = 12000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synchronized tabular + vision dataset reflecting realistic automotive risks.
    """
    np.random.seed(seed)

    # ---------------- TABULAR FEATURES ----------------
    speed_limits = np.random.choice([25, 35, 45, 55, 65, 75], size=n_samples, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
    road_alignment = np.random.binomial(n=1, p=0.25, size=n_samples)
    road_surface = np.random.binomial(n=1, p=0.22, size=n_samples)
    light_cond = np.random.choice([0, 1, 2], size=n_samples, p=[0.65, 0.20, 0.15])
    weather_cond = np.random.choice([0, 1, 2], size=n_samples, p=[0.72, 0.22, 0.06])
    driver_age = np.random.choice([0, 1, 2], size=n_samples, p=[0.18, 0.62, 0.20])
    driver_impairment = np.random.binomial(n=1, p=0.12, size=n_samples)
    vehicle_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.65, 0.25, 0.10])
    hist_risk = np.random.beta(a=2, b=10, size=n_samples)

    # ---------------- VISION FEATURES ----------------
    # Lane offset: normally centered at 0, wider variance if driver impaired
    offset_std = np.where(driver_impairment == 1, 0.65, 0.22)
    lane_offset = np.random.normal(0, offset_std, size=n_samples)

    # Lane curvature: higher on curve alignment
    lane_curv = np.where(road_alignment == 1, np.random.uniform(0.008, 0.035, size=n_samples), np.random.uniform(0.000, 0.005, size=n_samples))

    # Lane confidence: lower in rain/storm and dark unlit
    lane_conf_base = np.where(weather_cond > 0, 0.65, 0.92) - (light_cond == 2) * 0.20
    lane_confidence = np.clip(lane_conf_base + np.random.normal(0, 0.08, size=n_samples), 0.1, 1.0)

    # Monocular distance & TTC
    lead_dist = np.random.gamma(shape=3.5, scale=12.0, size=n_samples) + 3.0  # 5m to 80m
    lead_dist = np.clip(lead_dist, 3.0, 95.0)

    # Closing velocity (m/s)
    closing_vel = np.random.normal(0.5, 4.0, size=n_samples)
    # Higher closing rate if high speed limit and tailgating
    closing_vel = np.where((speed_limits >= 55) & (lead_dist < 20.0), closing_vel + 5.5, closing_vel)

    # Time-to-collision
    ttc = np.where(closing_vel > 0.5, lead_dist / np.maximum(closing_vel, 0.5), 10.0)
    ttc = np.clip(ttc, 0.2, 10.0)

    # Traffic density & VRU ratio
    traffic_density = np.random.poisson(lam=4.0, size=n_samples)
    vru_ratio = np.clip(np.random.beta(a=0.5, b=2.5, size=n_samples), 0.0, 1.0)

    # Photometry
    lum_mean = np.where(light_cond == 0, np.random.normal(160, 25, size=n_samples), np.random.normal(45, 15, size=n_samples))
    contrast_rms = np.clip(np.random.normal(48, 12, size=n_samples), 10, 95)
    glare_idx = np.where(light_cond == 1, np.random.exponential(0.04, size=n_samples), np.random.exponential(0.008, size=n_samples))
    rain_fog_idx = np.where(weather_cond > 0, np.random.uniform(0.35, 0.90, size=n_samples), np.random.uniform(0.02, 0.20, size=n_samples))

    # ---------------- GROUND TRUTH LATENT RISK FORMULATION ----------------
    # True physical risk integrates BOTH instantaneous kinematics and environmental context
    kinetic_threat = 3.8 * np.exp(-ttc / 1.8) + 0.12 * np.maximum(0, closing_vel)
    lane_threat = 1.6 * (np.abs(lane_offset) > 0.65).astype(float) + 1.2 * (lane_confidence < 0.45).astype(float)
    vru_threat = 2.2 * (vru_ratio > 0.40).astype(float) * (traffic_density > 4).astype(float)
    macro_threat = 0.5 * road_surface + 0.6 * driver_impairment + 0.8 * (light_cond == 2) + 1.4 * hist_risk

    total_risk_latent = (
        -3.2
        + kinetic_threat
        + lane_threat
        + vru_threat
        + macro_threat
        + np.random.normal(0, 0.35, size=n_samples)
    )

    prob = 1.0 / (1.0 + np.exp(-total_risk_latent))
    risk_label = (prob > 0.45).astype(int)

    data = {
        # Tabular
        "speed_limit_mph": speed_limits.astype(float),
        "road_alignment": road_alignment,
        "road_surface_condition": road_surface,
        "light_condition": light_cond,
        "weather_condition": weather_cond,
        "driver_age_group": driver_age,
        "driver_impairment_flag": driver_impairment,
        "vehicle_type": vehicle_type,
        "historical_segment_risk": hist_risk.astype(float),
        # Vision
        "lane_offset_m": lane_offset.astype(float),
        "lane_curvature_rad_m": lane_curv.astype(float),
        "lane_confidence": lane_confidence.astype(float),
        "min_ttc_sec": ttc.astype(float),
        "nearest_obj_distance_m": lead_dist.astype(float),
        "max_approach_rate_mps": closing_vel.astype(float),
        "traffic_density": traffic_density.astype(float),
        "vru_ratio": vru_ratio.astype(float),
        "scene_luminance_mean": np.clip(lum_mean, 0, 255).astype(float),
        "scene_contrast_rms": contrast_rms.astype(float),
        "glare_index": np.clip(glare_idx, 0, 1).astype(float),
        "rain_fog_scatter_index": np.clip(rain_fog_idx, 0, 1).astype(float),
        # Target
        "target_risk_label": risk_label
    }

    return pd.DataFrame(data)


def train_and_compare_all_models(output_dir: str = "models", docs_dir: str = "docs", plots_dir: str = "outputs"):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("[INFO] Generating synchronized multi-modal dataset (N=15,000)...")
    df = generate_multimodal_dataset(n_samples=15000, seed=42)

    X_tab = df[TABULAR_FEATURE_NAMES]
    X_vis = df[VISION_FEATURE_NAMES]
    X_fused = df[FUSED_FEATURE_NAMES]
    y = df["target_risk_label"]

    # Identical 80/20 train-test split across all models
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(indices, test_size=0.20, random_state=42, stratify=y)

    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    # ---------------- 1. TRAIN MODEL A (Crash-Only Baseline) ----------------
    print("[INFO] Training Model A (Crash Baseline)...")
    clf_a = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42, eval_metric="logloss")
    clf_a.fit(X_tab.iloc[idx_train], y_train)
    pred_a = clf_a.predict(X_tab.iloc[idx_test])
    prob_a = clf_a.predict_proba(X_tab.iloc[idx_test])[:, 1]
    clf_a.save_model(os.path.join(output_dir, "model_a_baseline.json"))

    # ---------------- 2. TRAIN MODEL B (Vision-Only) ----------------
    print("[INFO] Training Model B (Vision-Only)...")
    clf_b = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42, eval_metric="logloss")
    clf_b.fit(X_vis.iloc[idx_train], y_train)
    pred_b = clf_b.predict(X_vis.iloc[idx_test])
    prob_b = clf_b.predict_proba(X_vis.iloc[idx_test])[:, 1]
    clf_b.save_model(os.path.join(output_dir, "model_b_vision.json"))

    # ---------------- 3. TRAIN MODEL C (Multi-Modal Fusion) ----------------
    print("[INFO] Training Model C (Multi-Modal Fusion)...")
    clf_c = xgb.XGBClassifier(n_estimators=140, max_depth=5, learning_rate=0.07, subsample=0.85, random_state=42, eval_metric="logloss")
    clf_c.fit(X_fused.iloc[idx_train], y_train)
    pred_c = clf_c.predict(X_fused.iloc[idx_test])
    prob_c = clf_c.predict_proba(X_fused.iloc[idx_test])[:, 1]
    clf_c.save_model(os.path.join(output_dir, "model_c_fused.json"))

    # ---------------- COMPUTE METRICS ----------------
    def compute_metrics(y_true, y_p, y_pr):
        return {
            "Accuracy": accuracy_score(y_true, y_p),
            "Precision": precision_score(y_true, y_p),
            "Recall": recall_score(y_true, y_p),
            "F1-Score": f1_score(y_true, y_p),
            "ROC-AUC": roc_auc_score(y_true, y_pr),
            "PR-AUC": average_precision_score(y_true, y_pr),
            "CM": confusion_matrix(y_true, y_p).tolist()
        }

    m_a = compute_metrics(y_test, pred_a, prob_a)
    m_b = compute_metrics(y_test, pred_b, prob_b)
    m_c = compute_metrics(y_test, pred_c, prob_c)

    print("\n" + "=" * 78)
    print("  3-WAY MODEL COMPARISON BENCHMARK (MODELS A, B, C)")
    print("=" * 78)
    print(f"  Model A (Crash-Only):    F1: {m_a['F1-Score']:.4f} | Recall: {m_a['Recall']:.4f} | ROC-AUC: {m_a['ROC-AUC']:.4f}")
    print(f"  Model B (Vision-Only):   F1: {m_b['F1-Score']:.4f} | Recall: {m_b['Recall']:.4f} | ROC-AUC: {m_b['ROC-AUC']:.4f}")
    print(f"  Model C (Fusion A+B):    F1: {m_c['F1-Score']:.4f} | Recall: {m_c['Recall']:.4f} | ROC-AUC: {m_c['ROC-AUC']:.4f}")
    print("=" * 78)

    # ---------------- GENERATE COMPARISON REPORT ----------------
    report_path = os.path.join(docs_dir, "model_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison: Model A vs Model B vs Model C (SYNTHETIC development artifact)\n\n")
        f.write("> **WARNING.** The labels used here come from `generate_multimodal_dataset()`, a synthetic generator whose label is a formula of the input features. "
                "The numbers below therefore measure how well a model recovers that formula, **not** crash risk on real roads, and they must not be quoted as evidence that fusion helps. "
                "The real-data evaluation is in `experiments/exp3_ccd` (Car Crash Dataset) and `experiments/exp1_crss` (NHTSA CRSS).\n\n")
        f.write("## 1. Synthetic benchmark (development only)\n\n")
        f.write("## 2. Results on the synthetic labels\n\n")
        f.write("| Model Architecture | Input Modality | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Model A (Crash Baseline)** | Tabular NHTSA CRSS Features | `{m_a['Precision']:.4f}` | `{m_a['Recall']:.4f}` | `{m_a['F1-Score']:.4f}` | `{m_a['ROC-AUC']:.4f}` | `{m_a['PR-AUC']:.4f}` |\n")
        f.write(f"| **Model B (Vision-Only)** | Real-Time Lane, ByteTrack, TTC | `{m_b['Precision']:.4f}` | `{m_b['Recall']:.4f}` | `{m_b['F1-Score']:.4f}` | `{m_b['ROC-AUC']:.4f}` | `{m_b['PR-AUC']:.4f}` |\n")
        f.write(f"| **Model C (Multi-Modal Fusion)** | **Fused (Tabular + Vision)** | **`{m_c['Precision']:.4f}`** | **`{m_c['Recall']:.4f}`** | **`{m_c['F1-Score']:.4f}`** | **`{m_c['ROC-AUC']:.4f}`** | **`{m_c['PR-AUC']:.4f}`** |\n\n")

        f.write("## 3. How to read this\n")
        f.write("Differences between rows reflect how much of the synthetic label formula each input set contains. No claim about real-world crash prediction follows from them.\n")

    print(f"[INFO] Comparison report saved to: {report_path}")

    # ---------------- GENERATE BAR CHART PLOT ----------------
    metrics_names = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    vals_a = [m_a[k] for k in metrics_names]
    vals_b = [m_b[k] for k in metrics_names]
    vals_c = [m_c[k] for k in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.25

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    rects1 = ax.bar(x - width, vals_a, width, label="Model A (Crash-Only)", color="#4A90E2", edgecolor="#2171B5")
    rects2 = ax.bar(x, vals_b, width, label="Model B (Vision-Only)", color="#F39C12", edgecolor="#D68910")
    rects3 = ax.bar(x + width, vals_c, width, label="Model C (Multi-Modal Fusion)", color="#2ECC71", edgecolor="#27AE60")

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12, fontweight="bold", color="#FFFFFF")
    ax.set_title("Automotive Risk Benchmark: Model A vs Model B vs Model C", fontsize=14, fontweight="bold", pad=15, color="#FFFFFF")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, facecolor="#1F242D", edgecolor="#3A4750", fontsize=10)

    # Attach values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold", color="#ECF0F1")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "model_comparison_bars.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Comparison plot saved to: {plot_path}")

    return clf_a, clf_b, clf_c, (m_a, m_b, m_c)


if __name__ == "__main__":
    train_and_compare_all_models()
