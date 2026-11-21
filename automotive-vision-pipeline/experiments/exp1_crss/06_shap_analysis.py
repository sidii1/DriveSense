"""
06_shap_analysis.py
===================
DriveSense Experiment 1 — SHAP Attribution Aggregated by Original Feature

Scientific purpose
------------------
1. Uses TreeSHAP to interpret the primary XGBoost model on Target 2 (Serious 'A' Injury).
2. Uses TreeSHAP to interpret the DriveSense Tier 2 Perception-Proximal model.
3. Aggregates SHAP attributions by ORIGINAL FEATURE (summing one-hot levels)
   so that each real-world variable has a single, unfragmented attribution score.
4. Enforces strict scientific phrasing:
   - Explains that SHAP reflects mathematical feature attribution in the fitted model,
     NOT real-world causality.
   - Avoids speculative causal claims (e.g., does not claim higher person counts
     change individual severity, or that non-intersection causes higher speed;
     reports simply that these variables show strong predictive associations).
5. Generates:
   - shap_summary_tier3.png (Aggregated Beeswarm for Full Model)
   - shap_bar_tier3.png (Aggregated Mean |SHAP| Ranking for Full Model)
   - shap_bar_tier2.png (Aggregated Mean |SHAP| for DriveSense Perception Model)
   - shap_top_features.json (Aggregated feature rankings with scientific notes)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import shap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SCHEMA_JSON = os.path.join(RESULTS_DIR, "feature_schema.json")

FEATURE_DISPLAY_NAMES = {
    "A_WEATHER": "Weather Condition",
    "A_TOD": "Time of Day (Day vs. Night)",
    "A_DOW": "Day of Week (Weekday vs. Weekend)",
    "A_INTER": "Intersection Relation",
    "A_RELRD": "Roadway Relation",
    "A_INTSEC": "Intersection Type",
    "A_JUNC": "Junction Type",
    "A_PED": "Pedestrian Crash Involvement",
    "A_PEDAL": "Cyclist Crash Involvement",
    "A_MC": "Motorcycle Crash Involvement",
    "has_pedestrian": "Pedestrian Present in Scene",
    "has_pedalcyclist": "Cyclist Present in Scene",
    "has_motorcycle": "Motorcycle Present in Scene",
    "has_large_truck": "Large Truck Present in Scene",
    "has_bus": "Bus Present in Scene",
    "A_LT": "Large Truck Crash Flag",
    "dominant_body": "Dominant Vehicle Body Type",
    "n_vehicles": "Number of Vehicles Involved",
    "A_MANCOL": "Manner of Collision",
    "A_CT": "Crash Type",
    "A_SPCRA": "Speeding Crash Involvement",
    "has_speeding_veh": "Speeding Vehicle Present",
    "A_DIST": "Distraction Involvement",
    "A_HR": "Hit-and-Run Involvement",
    "n_persons": "Number of Persons Involved"
}

def aggregate_shap_by_feature(shap_raw, raw_feature_names, original_features):
    """
    Aggregates one-hot SHAP columns back into their original feature.
    Returns:
      shap_agg : (n_samples, n_original_features) array
      mean_abs : array of mean absolute SHAP for each original feature
      net_dir  : array of net mean SHAP for each original feature
    """
    n_samples = shap_raw.shape[0]
    n_orig = len(original_features)
    shap_agg = np.zeros((n_samples, n_orig))

    for orig_idx, orig_col in enumerate(original_features):
        matching_cols = []
        for raw_idx, raw_name in enumerate(raw_feature_names):
            clean_name = raw_name.replace("cat__", "").replace("num__", "").replace("prep__", "")
            if clean_name == orig_col or clean_name.startswith(orig_col + "_"):
                matching_cols.append(raw_idx)
        if matching_cols:
            shap_agg[:, orig_idx] = np.sum(shap_raw[:, matching_cols], axis=1)

    mean_abs = np.mean(np.abs(shap_agg), axis=0)
    net_dir = np.mean(shap_agg, axis=0)
    return shap_agg, mean_abs, net_dir

def main():
    print(f"\n============================================================")
    print("CRSS SHAP Explainability: Feature-Aggregated & Multi-Tier")
    print("============================================================")

    with open(SCHEMA_JSON) as f:
        schema = json.load(f)

    with open(os.path.join(RESULTS_DIR, "test_set_2024.pkl"), "rb") as f:
        df_test = pickle.load(f)

    tier_3_cols = schema["tiers"]["tier_3_full_historical"]
    tier_2_cols = schema["tiers"]["tier_2_perception_upper_bound"]

    # Sample 2,000 cases for fast, consistent TreeSHAP computation
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(df_test), size=min(2000, len(df_test)), replace=False)
    df_sample = df_test.iloc[sample_idx].reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 1. SHAP for Primary Tier 3 XGBoost Model (Target 2 - Serious Injury)
    # -----------------------------------------------------------------------
    print("\n[1/2] Computing SHAP on Tier 3 Full Model (Target 2)...")
    with open(os.path.join(RESULTS_DIR, "model_T2_xgboost.pkl"), "rb") as f:
        pipe_t3 = pickle.load(f)

    prep_t3 = pipe_t3.named_steps["prep"]
    clf_t3 = pipe_t3.named_steps["clf"]

    X_t3_trans = prep_t3.transform(df_sample[tier_3_cols])
    raw_names_t3 = prep_t3.get_feature_names_out()

    explainer_t3 = shap.TreeExplainer(clf_t3)
    shap_raw_t3 = explainer_t3.shap_values(X_t3_trans)
    if isinstance(shap_raw_t3, list):
        shap_raw_t3 = shap_raw_t3[1]

    shap_agg_t3, mean_abs_t3, net_dir_t3 = aggregate_shap_by_feature(shap_raw_t3, raw_names_t3, tier_3_cols)

    # Sort original features by mean absolute SHAP
    sort_t3 = np.argsort(mean_abs_t3)[::-1]
    sorted_features_t3 = [tier_3_cols[i] for i in sort_t3]
    sorted_names_t3 = [FEATURE_DISPLAY_NAMES.get(f, f) for f in sorted_features_t3]
    sorted_vals_t3 = [float(mean_abs_t3[i]) for i in sort_t3]

    top_features_meta = []
    for rank, idx in enumerate(sort_t3[:15]):
        col = tier_3_cols[idx]
        top_features_meta.append({
            "rank": rank + 1,
            "feature_key": col,
            "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
            "mean_abs_shap": round(float(mean_abs_t3[idx]), 4),
            "net_direction": "toward_serious" if net_dir_t3[idx] > 0 else "toward_non_serious"
        })

    # -----------------------------------------------------------------------
    # 2. SHAP for DriveSense Tier 2 Perception Model
    # -----------------------------------------------------------------------
    print("\n[2/2] Computing SHAP on DriveSense Tier 2 Perception Model...")
    with open(os.path.join(RESULTS_DIR, "model_tier_2_perception_upper_bound_xgb.pkl"), "rb") as f:
        pipe_t2 = pickle.load(f)

    prep_t2 = pipe_t2.named_steps["prep"]
    clf_t2 = pipe_t2.named_steps["clf"]

    X_t2_trans = prep_t2.transform(df_sample[tier_2_cols])
    raw_names_t2 = prep_t2.get_feature_names_out()

    explainer_t2 = shap.TreeExplainer(clf_t2)
    shap_raw_t2 = explainer_t2.shap_values(X_t2_trans)
    if isinstance(shap_raw_t2, list):
        shap_raw_t2 = shap_raw_t2[1]

    shap_agg_t2, mean_abs_t2, net_dir_t2 = aggregate_shap_by_feature(shap_raw_t2, raw_names_t2, tier_2_cols)
    sort_t2 = np.argsort(mean_abs_t2)[::-1]
    sorted_features_t2 = [tier_2_cols[i] for i in sort_t2]
    sorted_names_t2 = [FEATURE_DISPLAY_NAMES.get(f, f) for f in sorted_features_t2]
    sorted_vals_t2 = [float(mean_abs_t2[i]) for i in sort_t2]

    # Save JSON
    shap_out = {
        "interpretation_caveats": (
            "SHAP values quantify statistical feature attribution in the fitted XGBoost model, "
            "NOT real-world causal mechanisms. Observational confounders (impact speed, occupant age, "
            "seatbelt/helmet usage, local infrastructure) are not controlled in crash records. "
            "Features with high attribution reflect statistical associations with the crash-level outcome."
        ),
        "target": "Target 2 (Serious 'A' Injury vs Minor/None)",
        "tier_3_full_model_rankings": top_features_meta,
        "tier_2_perception_model_rankings": [
            {
                "rank": r + 1,
                "feature_key": sorted_features_t2[r],
                "display_name": sorted_names_t2[r],
                "mean_abs_shap": round(sorted_vals_t2[r], 4)
            }
            for r in range(min(12, len(sorted_features_t2)))
        ]
    }

    with open(os.path.join(RESULTS_DIR, "shap_top_features.json"), "w", encoding="utf-8") as f:
        json.dump(shap_out, f, indent=2)
    print(f"✓ Saved aggregated SHAP rankings to {os.path.join(RESULTS_DIR, 'shap_top_features.json')}")

    # -----------------------------------------------------------------------
    # Generate Publication Figures
    # -----------------------------------------------------------------------
    print("\nGenerating Aggregated SHAP Plots...")

    # Plot 1: Aggregated Beeswarm Summary (Tier 3)
    plt.figure(figsize=(10, 7.5))
    shap.summary_plot(
        shap_agg_t3[:, sort_t3[:15]],
        features=df_sample[sorted_features_t3[:15]],
        feature_names=sorted_names_t3[:15],
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP Feature Attribution Aggregated by Original Feature\n"
              "Primary Model (Tier 3 XGBoost) on Serious 'A' Injury Target\n"
              "(Non-Causal Statistical Attribution)",
              fontsize=11, pad=15)
    plt.xlabel("Aggregated SHAP Value (Impact on Model Severe Risk Prediction)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ shap_summary.png saved (Aggregated Beeswarm)")

    # Plot 2: Bar Chart Comparison (Tier 3 vs Tier 2 Perception Model)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Ax1: Tier 3 Full Model
    y_pos1 = np.arange(len(sorted_names_t3[:12]))
    ax1.barh(y_pos1, sorted_vals_t3[:12][::-1], color="#E91E63", alpha=0.85)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(sorted_names_t3[:12][::-1], fontsize=9.5)
    ax1.set_xlabel("Mean |SHAP Value| (Predictive Attribution)", fontsize=10.5)
    ax1.set_title("(a) Primary Model: Tier 3 Full Crash Report", fontsize=11.5)
    ax1.grid(axis="x", alpha=0.3)

    # Ax2: Tier 2 Perception Model
    y_pos2 = np.arange(len(sorted_names_t2[:12]))
    ax2.barh(y_pos2, sorted_vals_t2[:12][::-1], color="#9C27B0", alpha=0.85)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(sorted_names_t2[:12][::-1], fontsize=9.5)
    ax2.set_xlabel("Mean |SHAP Value| (Predictive Attribution)", fontsize=10.5)
    ax2.set_title("(b) DriveSense Model: Tier 2 Perception Upper Bound", fontsize=11.5)
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle("Feature Importance Aggregated by Original Variable\nTarget 2 (Serious 'A' Injury Crash)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ shap_bar.png saved (Tier 3 & Tier 2 Dual Bar Charts)")

    print("\n--- Top Aggregated Predictors in Primary Tier 3 Model ---")
    for item in top_features_meta[:10]:
        print(f"  {item['rank']:>2}. {item['display_name']:<35} : Mean|SHAP| = {item['mean_abs_shap']:.4f} ({item['net_direction']})")

    print("\n--- Top Aggregated Predictors in DriveSense Tier 2 Perception Model ---")
    for r in range(min(8, len(sorted_features_t2))):
        print(f"  {r+1:>2}. {sorted_names_t2[r]:<35} : Mean|SHAP| = {sorted_vals_t2[r]:.4f}")

    print("\n✓ Aggregated SHAP analysis complete.")

if __name__ == "__main__":
    main()
