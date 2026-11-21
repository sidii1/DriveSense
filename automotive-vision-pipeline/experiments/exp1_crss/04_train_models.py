"""
04_train_models.py
==================
DriveSense Experiment 1 — Model Training & Validation Protocol

Scientific purpose
------------------
1. Trains classifiers across two crash severity targets:
   - Target 1 (T1): Any Injury vs. Property-Damage-Only (PDO).
   - Target 2 (T2): Serious (A) Injury vs. Minor/Possible/None.
     (Primary target for automotive safety baseline).
2. Evaluates the DriveSense Feature Tier Progression on Target 2:
   - Tier 1: Macro-Contextual / Environmental (weather, day/night, road relation).
   - Tier 2a: Tier 1 + VRU flags only (pedestrian, cyclist, motorcycle).
   - Tier 2b: Tier 1 + Vehicle fleet types only (body types, large truck, bus).
   - Tier 2: Perception Upper Bound (Tier 1 + Tier 2a + Tier 2b).
   - Tier 3: Full Historical Crash Report Upper Bound (+collision geometry, speeding, counts).
3. Evaluates Feature Ablation:
   - Tier 3 without n_persons and n_vehicles (tests sensitivity to occupant logging).
4. Conducts Temporal Generalization:
   - Train on 2023 CRSS (n=50,092) -> Evaluated on 2024 CRSS (n=51,641).
5. Tunes F1 threshold on training data via 5-fold cross-validation out-of-fold predictions.
"""

import os
import sys
import json
import pickle
import time
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
INPUT_FEATURE_CSV = os.path.join(RESULTS_DIR, "feature_matrix.csv")
SCHEMA_JSON = os.path.join(RESULTS_DIR, "feature_schema.json")
RANDOM_STATE = 42

def create_preprocessor(cat_cols, num_cols):
    transformers = []
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ))
    if num_cols:
        transformers.append((
            "num",
            StandardScaler(),
            num_cols
        ))
    return ColumnTransformer(transformers=transformers)

def get_pipelines(cat_cols, num_cols, pos_weight=1.0):
    lr_pipe = Pipeline([
        ("prep", create_preprocessor(cat_cols, num_cols)),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE
        ))
    ])

    rf_pipe = Pipeline([
        ("prep", create_preprocessor(cat_cols, num_cols)),
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    xgb_pipe = Pipeline([
        ("prep", create_preprocessor(cat_cols, num_cols)),
        ("clf", xgb.XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    return {
        "LogisticRegression": lr_pipe,
        "RandomForest": rf_pipe,
        "XGBoost": xgb_pipe
    }

def find_best_f1_threshold(y_true, oof_probs):
    best_f1 = 0.0
    best_t = 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (oof_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return round(best_t, 3), round(best_f1, 4)

def split_cat_num(cols):
    nums = [c for c in cols if c in ("n_vehicles", "n_persons")]
    cats = [c for c in cols if c not in nums]
    return cats, nums

def main():
    print(f"\n============================================================")
    print("CRSS Model Training: Multi-Target, Multi-Tier & Temporal")
    print("============================================================")

    df = pd.read_csv(INPUT_FEATURE_CSV, low_memory=False)
    with open(SCHEMA_JSON) as f:
        schema = json.load(f)

    df_2024 = df[df['YEAR'] == 2024].copy().reset_index(drop=True)
    df_2023 = df[df['YEAR'] == 2023].copy().reset_index(drop=True)

    print(f"Cohort sizes: 2024={len(df_2024):,}, 2023={len(df_2023):,}")

    tier_1 = schema["tiers"]["tier_1_environmental"]
    tier_2a_vru = schema["tiers"]["tier_2a_vru_only"]
    tier_2b_veh = schema["tiers"]["tier_2b_veh_only"]
    tier_2_all = schema["tiers"]["tier_2_perception_upper_bound"]
    tier_3_full = schema["tiers"]["tier_3_full_historical"]
    ablation_cols = schema["tiers"]["ablation_no_counts"]

    # 80/20 Stratified Split on 2024 data (Target 2 stratified)
    train_idx, test_idx = train_test_split(
        df_2024.index,
        test_size=0.20,
        stratify=df_2024['target_t2_serious_injury'],
        random_state=RANDOM_STATE
    )

    df_train_2024 = df_2024.iloc[train_idx].reset_index(drop=True)
    df_test_2024 = df_2024.iloc[test_idx].reset_index(drop=True)

    print(f"\n2024 Benchmark Split: Train={len(df_train_2024):,}, Test={len(df_test_2024):,}")
    print(f"  T1 Train positive rate: {df_train_2024['target_t1_any_inj'].mean()*100:.1f}%")
    print(f"  T2 Train positive rate: {df_train_2024['target_t2_serious_injury'].mean()*100:.1f}%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_summary = {}

    # -----------------------------------------------------------------------
    # 1. Train Target 1 (Any Injury vs PDO) on Full Tier 3
    # -----------------------------------------------------------------------
    print(f"\n[1/6] Training Models for Target 1 (Any Injury vs. PDO)...")
    t1_pos_weight = (len(df_train_2024) - df_train_2024['target_t1_any_inj'].sum()) / df_train_2024['target_t1_any_inj'].sum()
    t1_cats, t1_nums = split_cat_num(tier_3_full)
    t1_pipes = get_pipelines(t1_cats, t1_nums, pos_weight=t1_pos_weight)

    cv_summary["Target_1"] = {}
    for name, pipe in t1_pipes.items():
        t0 = time.time()
        scores = cross_val_score(
            pipe, df_train_2024[tier_3_full], df_train_2024['target_t1_any_inj'],
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        oof_probs = cross_val_predict(
            pipe, df_train_2024[tier_3_full], df_train_2024['target_t1_any_inj'],
            cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        best_t, best_f1 = find_best_f1_threshold(df_train_2024['target_t1_any_inj'].to_numpy(), oof_probs)

        pipe.fit(df_train_2024[tier_3_full], df_train_2024['target_t1_any_inj'])
        elapsed = round(time.time() - t0, 2)
        print(f"  {name:<20}: CV ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f} | Tuned Thr = {best_t:.3f} (OOF F1 = {best_f1:.4f}) [{elapsed}s]")
        cv_summary["Target_1"][name] = {
            "cv_roc_auc_mean": round(float(scores.mean()), 4),
            "cv_roc_auc_std": round(float(scores.std()), 4),
            "tuned_threshold": best_t,
            "oof_f1": best_f1
        }
        with open(os.path.join(RESULTS_DIR, f"model_T1_{name.lower()}.pkl"), "wb") as f:
            pickle.dump(pipe, f)

    # -----------------------------------------------------------------------
    # 2. Train Target 2 (Serious 'A' Injury vs Minor/None) on Full Tier 3
    # -----------------------------------------------------------------------
    print(f"\n[2/6] Training Models for Target 2 (Serious 'A' Injury)...")
    t2_pos_weight = (len(df_train_2024) - df_train_2024['target_t2_serious_injury'].sum()) / df_train_2024['target_t2_serious_injury'].sum()
    t2_cats, t2_nums = split_cat_num(tier_3_full)
    t2_pipes = get_pipelines(t2_cats, t2_nums, pos_weight=t2_pos_weight)

    cv_summary["Target_2"] = {}
    for name, pipe in t2_pipes.items():
        t0 = time.time()
        scores = cross_val_score(
            pipe, df_train_2024[tier_3_full], df_train_2024['target_t2_serious_injury'],
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        oof_probs = cross_val_predict(
            pipe, df_train_2024[tier_3_full], df_train_2024['target_t2_serious_injury'],
            cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        best_t, best_f1 = find_best_f1_threshold(df_train_2024['target_t2_serious_injury'].to_numpy(), oof_probs)

        pipe.fit(df_train_2024[tier_3_full], df_train_2024['target_t2_serious_injury'])
        elapsed = round(time.time() - t0, 2)
        print(f"  {name:<20}: CV ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f} | Tuned Thr = {best_t:.3f} (OOF F1 = {best_f1:.4f}) [{elapsed}s]")
        cv_summary["Target_2"][name] = {
            "cv_roc_auc_mean": round(float(scores.mean()), 4),
            "cv_roc_auc_std": round(float(scores.std()), 4),
            "tuned_threshold": best_t,
            "oof_f1": best_f1
        }
        with open(os.path.join(RESULTS_DIR, f"model_T2_{name.lower()}.pkl"), "wb") as f:
            pickle.dump(pipe, f)

    # -----------------------------------------------------------------------
    # 3. DriveSense Feature Tiers (on Target 2 with XGBoost)
    # -----------------------------------------------------------------------
    print(f"\n[3/6] Training DriveSense Feature Tiers (Target 2 with XGBoost)...")
    tier_specs = {
        "Tier_1_Environmental": tier_1,
        "Tier_2a_VRU_Only": tier_2a_vru,
        "Tier_2b_VehType_Only": tier_2b_veh,
        "Tier_2_Perception_Upper_Bound": tier_2_all,
        "Tier_3_FullReport": tier_3_full
    }

    cv_summary["DriveSense_Tiers"] = {}
    for tier_name, cols in tier_specs.items():
        c_cats, c_nums = split_cat_num(cols)
        pipe = Pipeline([
            ("prep", create_preprocessor(c_cats, c_nums)),
            ("clf", xgb.XGBClassifier(
                n_estimators=250, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                scale_pos_weight=t2_pos_weight, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])
        scores = cross_val_score(
            pipe, df_train_2024[cols], df_train_2024['target_t2_serious_injury'],
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        oof_probs = cross_val_predict(
            pipe, df_train_2024[cols], df_train_2024['target_t2_serious_injury'],
            cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        best_t, best_f1 = find_best_f1_threshold(df_train_2024['target_t2_serious_injury'].to_numpy(), oof_probs)

        pipe.fit(df_train_2024[cols], df_train_2024['target_t2_serious_injury'])
        print(f"  {tier_name:<30}: CV ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f} | Tuned Thr = {best_t:.3f}")
        cv_summary["DriveSense_Tiers"][tier_name] = {
            "cv_roc_auc_mean": round(float(scores.mean()), 4),
            "cv_roc_auc_std": round(float(scores.std()), 4),
            "tuned_threshold": best_t,
            "oof_f1": best_f1
        }
        with open(os.path.join(RESULTS_DIR, f"model_{tier_name.lower()}_xgb.pkl"), "wb") as f:
            pickle.dump(pipe, f)

    # -----------------------------------------------------------------------
    # 4. Feature Ablation (Tier 3 without n_persons and n_vehicles)
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Training Feature Ablation (Tier 3 without counts)...")
    ab_cats, ab_nums = split_cat_num(ablation_cols)
    ab_pipe = Pipeline([
        ("prep", create_preprocessor(ab_cats, ab_nums)),
        ("clf", xgb.XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=t2_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    ab_scores = cross_val_score(
        ab_pipe, df_train_2024[ablation_cols], df_train_2024['target_t2_serious_injury'],
        cv=cv, scoring="roc_auc", n_jobs=-1
    )
    oof_probs_ab = cross_val_predict(
        ab_pipe, df_train_2024[ablation_cols], df_train_2024['target_t2_serious_injury'],
        cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    best_t_ab, best_f1_ab = find_best_f1_threshold(df_train_2024['target_t2_serious_injury'].to_numpy(), oof_probs_ab)

    ab_pipe.fit(df_train_2024[ablation_cols], df_train_2024['target_t2_serious_injury'])
    print(f"  Ablation (no counts)          : CV ROC-AUC = {ab_scores.mean():.4f} +/- {ab_scores.std():.4f}")
    cv_summary["Ablation_no_counts"] = {
        "cv_roc_auc_mean": round(float(ab_scores.mean()), 4),
        "cv_roc_auc_std": round(float(ab_scores.std()), 4),
        "tuned_threshold": best_t_ab,
        "oof_f1": best_f1_ab
    }
    with open(os.path.join(RESULTS_DIR, "model_ablation_no_counts_xgb.pkl"), "wb") as f:
        pickle.dump(ab_pipe, f)

    # -----------------------------------------------------------------------
    # 5. Temporal Model: Train on 2023 CRSS (n=50,092)
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Training Temporal Model (Trained on 2023 CRSS, n={len(df_2023):,})...")
    t2_2023_pos_wt = (len(df_2023) - df_2023['target_t2_serious_injury'].sum()) / df_2023['target_t2_serious_injury'].sum()
    temp_cats, temp_nums = split_cat_num(tier_3_full)
    temp_pipe = Pipeline([
        ("prep", create_preprocessor(temp_cats, temp_nums)),
        ("clf", xgb.XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=t2_2023_pos_wt, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    temp_pipe.fit(df_2023[tier_3_full], df_2023['target_t2_serious_injury'])
    with open(os.path.join(RESULTS_DIR, "model_temporal_xgb.pkl"), "wb") as f:
        pickle.dump(temp_pipe, f)
    print("  ✓ Temporal model fitted on full 2023 cohort.")

    # -----------------------------------------------------------------------
    # 6. Save Evaluation Data
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Saving test sets and CV metadata...")
    with open(os.path.join(RESULTS_DIR, "test_set_2024.pkl"), "wb") as f:
        pickle.dump(df_test_2024, f)

    with open(os.path.join(RESULTS_DIR, "test_set_temporal.pkl"), "wb") as f:
        pickle.dump(df_2024, f)

    with open(os.path.join(RESULTS_DIR, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"✓ Cross-validation results saved to {os.path.join(RESULTS_DIR, 'cv_results.json')}")
    print("✓ All models and evaluation sets saved. Proceed to 05_evaluate_models.py.")

if __name__ == "__main__":
    main()
