"""
03_feature_engineering.py
==========================
DriveSense Experiment 1 — Feature Engineering & Vision-Link Tier Architecture

Scientific purpose
------------------
1. Prepares model-ready feature matrices for Target 1 and Target 2:
   - Target 1 (T1): Any Injury Crash (1) vs. Property-Damage-Only (0).
   - Target 2 (T2): Serious (A) Injury Crash (1) vs. Minor/Possible/None (0).
     (Zero fatal records exist in CRSS AUX; T2 represents incapacitating/serious injury).
2. Defines the DriveSense Feature Hierarchy:
   - Tier 1: Environmental & Macro-Contextual (weather, time of day [day/night],
     day of week [weekday/weekend], roadway relation, intersection/junction layout).
   - Tier 2a: Tier 1 + Vulnerable Road User (VRU) flags only (pedestrian, cyclist, motorcycle).
   - Tier 2b: Tier 1 + Vehicle Fleet Composition only (body types, large truck, bus).
   - Tier 2 (Full Perception Upper Bound): Tier 1 + Tier 2a + Tier 2b.
     Represents an upper bound on what an on-vehicle vision system could contribute
     if it perfectly detected the involved agent classes before impact.
   - Tier 3: Full Crash-Report Upper Bound (Tier 2 + collision manner, crash type,
     speeding, distraction, hit-and-run, n_vehicles, n_persons).
3. Defines Feature Ablation:
   - Tier 3 excluding occupant (n_persons) and vehicle (n_vehicles) counts,
     testing sensitivity to police report logging artifacts.
4. Quarantines Post-Crash Dynamics:
   - Rollover variables (A_ROLL, has_rollover) are quarantined from all predictive tiers.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
INPUT_CLEANED_CSV = os.path.join(RESULTS_DIR, "crss_cleaned.csv")
OUTPUT_FEATURE_CSV = os.path.join(RESULTS_DIR, "feature_matrix.csv")
OUTPUT_SCHEMA_JSON = os.path.join(RESULTS_DIR, "feature_schema.json")

# ---------------------------------------------------------------------------
# Feature Tier Specifications
# ---------------------------------------------------------------------------

# Tier 1: Environmental & Roadway Context (Static / Map / Weather API)
TIER_1_FEATURES = [
    "A_WEATHER",   # Weather condition (categorical)
    "A_TOD",       # Time of day: 1=Daytime 6am-6pm, 2=Nighttime 6pm-6am (categorical)
    "A_DOW",       # Day of week: 1=Weekday, 2=Weekend (categorical)
    "A_INTER",     # Intersection-related (categorical)
    "A_RELRD",     # Relation to roadway (categorical)
    "A_INTSEC",    # Intersection type (categorical)
    "A_JUNC",      # Junction type (categorical)
]

# Tier 2a: Tier 1 + VRU Flags Only
VRU_FEATURES = [
    "A_PED",            # Pedestrian crash flag (categorical)
    "A_PEDAL",          # Pedalcyclist crash flag (categorical)
    "A_MC",             # Motorcycle crash flag (categorical)
    "has_pedestrian",   # Pedestrian present in person table (categorical)
    "has_pedalcyclist", # Pedalcyclist present in person table (categorical)
    "has_motorcycle",   # Motorcycle present in vehicle table (categorical)
]

# Tier 2b: Tier 1 + Vehicle Fleet / Type Only
VEHICLE_TYPE_FEATURES = [
    "dominant_body",    # Modal vehicle body type (categorical)
    "has_large_truck",  # Large truck present from vehicle table (categorical)
    "A_LT",             # Large truck crash flag from accident table (categorical)
    "has_bus",          # Bus present from vehicle table (categorical)
]

# Tier 3: Additional Police Investigation & Crash Circumstances
TIER_3_ADDITIONAL = [
    "A_MANCOL",         # Manner of collision (categorical)
    "A_CT",             # Crash type (categorical)
    "A_SPCRA",          # Speeding crash flag (categorical)
    "has_speeding_veh", # Speeding vehicle involved (categorical)
    "A_DIST",           # Distraction flag (categorical)
    "A_HR",             # Hit-and-run flag (categorical)
    "n_vehicles",       # Number of vehicles involved (numeric - post-crash count)
    "n_persons",        # Total persons involved (numeric - post-crash count)
]

POST_CRASH_VARS = [
    "A_ROLL",           # Rollover crash flag
    "has_rollover",     # Rollover vehicle flag
]

LEAKAGE_VARS = [
    "A_CRAINJ",
    "n_injured",
    "n_serious",
    "n_fatal",
    "max_injury_severity"
]

def main():
    print(f"\n============================================================")
    print("CRSS Feature Engineering & Multi-Tier Architecture")
    print("============================================================")

    df = pd.read_csv(INPUT_CLEANED_CSV, low_memory=False)
    print(f"Loaded cleaned multi-year crash table: {len(df):,} rows")

    # Feature Tiers
    tier_1 = list(TIER_1_FEATURES)
    tier_2a_vru = tier_1 + list(VRU_FEATURES)
    tier_2b_veh = tier_1 + list(VEHICLE_TYPE_FEATURES)
    tier_2_all = tier_1 + list(VRU_FEATURES) + list(VEHICLE_TYPE_FEATURES)
    tier_3_full = tier_2_all + list(TIER_3_ADDITIONAL)
    ablation_no_counts = [f for f in tier_3_full if f not in ("n_persons", "n_vehicles")]

    numeric_cols = ["n_vehicles", "n_persons"]
    categorical_cols = [c for c in tier_3_full if c not in numeric_cols]

    print(f"\nFeature Tiers Configured:")
    print(f"  Tier 1 (Macro-Contextual / Environmental) : {len(tier_1)} features")
    print(f"  Tier 2a (Tier 1 + VRU Flags Only)         : {len(tier_2a_vru)} features (+{len(VRU_FEATURES)} VRU flags)")
    print(f"  Tier 2b (Tier 1 + Vehicle Types Only)     : {len(tier_2b_veh)} features (+{len(VEHICLE_TYPE_FEATURES)} Vehicle Type flags)")
    print(f"  Tier 2 (Perception Upper Bound)           : {len(tier_2_all)} features (VRU + Fleet Types)")
    print(f"  Tier 3 (Full Historical Crash Report)     : {len(tier_3_full)} features (+{len(TIER_3_ADDITIONAL)} investigative features)")
    print(f"  Ablation (Tier 3 without counts)          : {len(ablation_no_counts)} features")

    # Encode categoricals as string format for OneHotEncoder
    for col in categorical_cols:
        df[col] = df[col].fillna(9).astype(int).astype(str)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1.0).astype(float)

    # Rename target 2 to clearly reflect Serious Injury
    df['target_t2_serious_injury'] = df['target_t2_serious_fatal'].astype(int)

    print(f"\nTarget Distributions Across Multi-Year Dataset (n = {len(df):,}):")
    print(f"  Target 1 (Any Injury):")
    print(f"    Positive (1): {int(df['target_t1_any_inj'].sum()):,} ({df['target_t1_any_inj'].mean()*100:.2f}%)")
    print(f"    Negative (0): {int((df['target_t1_any_inj']==0).sum()):,} ({(df['target_t1_any_inj']==0).mean()*100:.2f}%)")
    print(f"  Target 2 (Serious 'A' Injury):")
    print(f"    Positive (1): {int(df['target_t2_serious_injury'].sum()):,} ({df['target_t2_serious_injury'].mean()*100:.2f}%)")
    print(f"    Negative (0): {int((df['target_t2_serious_injury']==0).sum()):,} ({(df['target_t2_serious_injury']==0).mean()*100:.2f}%)")

    cols_to_save = [
        "CASENUM", "YEAR", "WEIGHT",
        "target_t1_any_inj", "target_t2_serious_injury"
    ] + tier_3_full + POST_CRASH_VARS

    df_out = df[cols_to_save]
    df_out.to_csv(OUTPUT_FEATURE_CSV, index=False)
    print(f"\n✓ Saved feature matrix to {OUTPUT_FEATURE_CSV}")

    schema = {
        "metadata": {
            "total_records": len(df_out),
            "years_included": sorted(df_out['YEAR'].unique().tolist()),
            "leakage_excluded_variables": LEAKAGE_VARS,
            "post_crash_quarantined_variables": POST_CRASH_VARS
        },
        "tiers": {
            "tier_1_environmental": tier_1,
            "tier_2a_vru_only": tier_2a_vru,
            "tier_2b_veh_only": tier_2b_veh,
            "tier_2_perception_upper_bound": tier_2_all,
            "tier_3_full_historical": tier_3_full,
            "ablation_no_counts": ablation_no_counts
        },
        "column_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "targets": ["target_t1_any_inj", "target_t2_serious_injury"],
            "metadata": ["CASENUM", "YEAR", "WEIGHT"]
        }
    }

    with open(OUTPUT_SCHEMA_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"✓ Saved feature schema to {OUTPUT_SCHEMA_JSON}")

if __name__ == "__main__":
    main()
