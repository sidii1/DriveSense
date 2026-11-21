"""
02_clean_and_merge.py
=====================
DriveSense Experiment 1 — CRSS 2023 & 2024 Cleaning and Aggregation

Scientific purpose
------------------
Aggregates relational tables (ACCIDENT, PERSON, VEHICLE) to one row per crash
for both 2023 and 2024 cohorts.

Target Variables Created
-------------------------
1. Target 1 (T1) — Any Injury Crash:
   severe_crash_t1 = 1 if at least one person injured (K, A, B, C, or unknown-injured)
                   = 0 if Property-Damage-Only (no injury)
2. Target 2 (T2) — Serious/Fatal Crash (KA severity):
   severe_crash_t2 = 1 if at least one person sustained a fatal (K) or suspected serious (A) injury
                   = 0 if minor, possible, or no apparent injury

Feature Cleaning Decisions
--------------------------
- Rollover variables (A_ROLL, has_rollover) are tagged as POST-CRASH circumstances
  and excluded from the primary pre-collision / scene-context models.
- Vehicle and person counts (n_vehicles, n_persons) are computed and preserved for ablation.
- Demographic / post-crash injury counts (n_fatal, n_serious, etc.) are separated from features.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    2024: {
        "acc": os.path.join(BASE_DATA_DIR, "raw", "crss_2024", "ACC_AUX.CSV"),
        "per": os.path.join(BASE_DATA_DIR, "raw", "crss_2024", "PER_AUX.CSV"),
        "veh": os.path.join(BASE_DATA_DIR, "raw", "crss_2024", "VEH_AUX.CSV"),
    },
    2023: {
        "acc": os.path.join(BASE_DATA_DIR, "CRSS2023", "acc_aux.csv"),
        "per": os.path.join(BASE_DATA_DIR, "CRSS2023", "per_aux.csv"),
        "veh": os.path.join(BASE_DATA_DIR, "CRSS2023", "veh_aux.csv"),
    }
}

OUTPUT_CLEANED_CSV = os.path.join(RESULTS_DIR, "crss_cleaned.csv")
AUDIT_JSON = os.path.join(RESULTS_DIR, "merge_audit.json")

def process_cohort(year, paths):
    print(f"\nProcessing Cohort Year: {year}")
    df_acc = pd.read_csv(paths["acc"], low_memory=False)
    df_per = pd.read_csv(paths["per"], low_memory=False)
    df_veh = pd.read_csv(paths["veh"], low_memory=False)

    df_acc['YEAR'] = year
    df_per['YEAR'] = year
    df_veh['YEAR'] = year

    # 1. Aggregate Person level
    # A_PERINJ: 1=Fatal, 2=Serious(A), 3=Minor(B), 4=Possible(C), 5=No injury(O), 7=Injured-unknown
    # A_PTYPE: 1=Driver, 2=Passenger, 3=Pedestrian, 4=Pedalcyclist, 5=Other
    per_agg = df_per.groupby('CASENUM').agg(
        n_persons=('PER_NO', 'count'),
        n_injured=('A_PERINJ', lambda s: int(s.isin([1, 2, 3, 4, 7]).sum())),
        n_serious=('A_PERINJ', lambda s: int((s == 2).sum())),
        n_fatal=('A_PERINJ', lambda s: int((s == 1).sum())),
        max_injury_severity=('A_PERINJ', 'min'),
        target_t1_any_inj=('A_PERINJ', lambda s: int(s.isin([1, 2, 3, 4, 7]).any())),
        target_t2_serious_fatal=('A_PERINJ', lambda s: int(s.isin([1, 2]).any())),
        has_pedestrian=('A_PTYPE', lambda s: int((s == 3).any())),
        has_pedalcyclist=('A_PTYPE', lambda s: int((s == 4).any())),
    ).reset_index()

    # 2. Aggregate Vehicle level
    # A_BODY: 1=Car, 2=Light truck/SUV/pickup, 3=Large truck, 4=Motorcycle, 5=Bus, 6=Other/unknown
    # A_VROLL: 1=Rollover, 2=No rollover
    # A_SPVEH: 1=Speeding, 2=Not speeding
    def agg_veh(group):
        bodies = group['A_BODY'].dropna().astype(int).tolist()
        dom_body = group['A_BODY'].mode()[0] if len(group['A_BODY'].mode()) > 0 else 1
        return pd.Series({
            'n_vehicles': len(group),
            'has_large_truck': int((group['A_BODY'] == 3).any()),
            'has_motorcycle': int((group['A_BODY'] == 4).any()),
            'has_bus': int((group['A_BODY'] == 5).any()),
            'has_speeding_veh': int((group['A_SPVEH'] == 1).any()),
            'has_rollover': int((group['A_VROLL'] == 1).any()),
            'dominant_body': int(dom_body)
        })

    veh_agg = df_veh.groupby('CASENUM').apply(agg_veh, include_groups=False).reset_index()

    # 3. Merge with ACC_AUX
    merged = pd.merge(df_acc, per_agg, on='CASENUM', how='inner')
    merged = pd.merge(merged, veh_agg, on='CASENUM', how='inner')

    # Double check T1 with A_CRAINJ:
    # A_CRAINJ in [2, 4] is also Any Injury
    merged['target_t1_any_inj'] = np.where(
        merged['A_CRAINJ'].isin([2, 4]) | (merged['target_t1_any_inj'] == 1),
        1,
        0
    )

    print(f"  Merged crashes for {year}: {len(merged):,}")
    print(f"  T1 (Any Injury) prevalence: {merged['target_t1_any_inj'].mean():.4f}")
    print(f"  T2 (Serious/Fatal KA) prevalence: {merged['target_t2_serious_fatal'].mean():.4f}")

    return merged

def main():
    cohorts = []
    audit = {}

    for year, paths in DATASETS.items():
        df_cohort = process_cohort(year, paths)
        cohorts.append(df_cohort)
        audit[str(year)] = {
            "total_crashes": len(df_cohort),
            "t1_positives": int(df_cohort['target_t1_any_inj'].sum()),
            "t1_prevalence": float(df_cohort['target_t1_any_inj'].mean()),
            "t2_positives": int(df_cohort['target_t2_serious_fatal'].sum()),
            "t2_prevalence": float(df_cohort['target_t2_serious_fatal'].mean()),
            "weighted_t1_rate": float((df_cohort['target_t1_any_inj'] * df_cohort['WEIGHT']).sum() / df_cohort['WEIGHT'].sum()),
            "weighted_t2_rate": float((df_cohort['target_t2_serious_fatal'] * df_cohort['WEIGHT']).sum() / df_cohort['WEIGHT'].sum()),
        }

    full_df = pd.concat(cohorts, ignore_index=True)
    print(f"\n============================================================")
    print(f"Total Combined Crashes (2023 + 2024): {len(full_df):,}")
    print(f"============================================================")

    full_df.to_csv(OUTPUT_CLEANED_CSV, index=False)
    print(f"✓ Saved cleaned multi-year crash table to {OUTPUT_CLEANED_CSV}")

    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"✓ Saved merge audit to {AUDIT_JSON}")

if __name__ == "__main__":
    main()
