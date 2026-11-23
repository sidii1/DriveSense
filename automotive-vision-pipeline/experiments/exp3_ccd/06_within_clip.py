"""STEP 6: source-free early-warning evaluation WITHIN accident clips.

The crash-vs-normal task of 04_anticipation_eval.py is confounded by the data source (encoding properties alone
separate CCD accident from normal clips almost perfectly). Here every clip is its own control: accident clips only,
"far" frames (more than 2 s before the annotated onset) are negatives, "imminent" frames (last 1 s before onset) are
positives, frames in between are ignored. Source, camera, encoder and scene are shared by both classes, so a score
can only separate them through what changes as the accident approaches.

Splits are by clip with the official CCD lists; the operating threshold is fixed on held-out training clips
(FAR = share of far frames above threshold = 10 %); CIs are bootstrap over clips.

python 06_within_clip.py --crash results/frame_features_all.csv --split-dir ../../data/CarCrash/splits --out results/eval --paper-dir ../../../paper
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

spec = importlib.util.spec_from_file_location("ev", str(Path(__file__).resolve().parent / "04_anticipation_eval.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

FAR_T, NEAR, FARFRAME = 0.10, (-1.0, 0.0), -2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--paper-dir", default="")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    cr = pd.read_csv(a.crash)
    cr["label"] = 1
    cr["clip"] = "crash_" + cr["video_id"].map(ev.norm_id)
    df = cr.sort_values(["clip", "t_sec"]).reset_index(drop=True)
    df, cues = ev.base_cues(df)
    clips = df.groupby("clip")["label"].first()
    split, src, _ = ev.read_split(a.split_dir, clips)
    df = df[df["clip"].isin(split.index)].reset_index(drop=True)
    X = ev.causal_features(df, cues)

    pos = (df.t_rel >= NEAR[0]) & (df.t_rel < NEAR[1])
    neg = df.t_rel < FARFRAME
    y = pos.astype(int)
    use = pos | neg
    sp = df["clip"].map(split)
    tr_clips = split[split == "train"].index.intersection(df["clip"].unique())
    fit_c, val_c = train_test_split(tr_clips, test_size=0.2, random_state=ev.SEED)
    is_fit, is_val, is_te = df["clip"].isin(fit_c), df["clip"].isin(val_c), sp == "test"

    def evaluate(name, s):
        """s: frame score Series. Frame-level AUC/AP on test clips, plus clip-level alerting at fixed far-frame rate."""
        s = ev.smooth(df, s.values)
        v_far = s[is_val & neg].values
        thr = float(np.quantile(v_far, 1 - FAR_T)) + 1e-12
        te = df[is_te & use].assign(s=s[is_te & use], y=y[is_te & use])
        te_clip = df[is_te & (df.t_rel < 0)].assign(s=s[is_te & (df.t_rel < 0)])
        ids = te["clip"].unique()

        # per-clip arrays so that the cluster bootstrap resamples whole clips WITH duplicates
        Y = {k: g.y.values for k, g in te.groupby("clip")}
        Sc = {k: g.s.values for k, g in te.groupby("clip")}
        hit = {}
        for k, g in te_clip.groupby("clip"):
            w = g[(g.t_rel >= -2.0) & (g.s >= thr)]
            hit[k] = float(-w.t_rel.min()) if len(w) else np.nan      # lead time of the first alert (s)

        def stats(sel):
            yy = np.concatenate([Y[k] for k in sel])
            ss = np.concatenate([Sc[k] for k in sel])
            far_alert = float((ss[yy == 0] >= thr).mean())
            lead = np.array([hit[k] for k in sel])
            rec = float(np.isfinite(lead).mean())
            mtta = float(np.nanmean(lead)) if np.isfinite(lead).any() else 0.0
            return roc_auc_score(yy, ss), average_precision_score(yy, ss), far_alert, rec, mtta

        base = stats(list(ids))
        rng = np.random.default_rng(ev.SEED)
        bs = np.array([stats(list(rng.choice(ids, len(ids)))) for _ in range(a.boot)])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        return {"AUC": base[0], "AUC_ci": [lo[0], hi[0]], "AP": base[1], "AP_ci": [lo[1], hi[1]],
                "far_frames": base[2], "recall": base[3], "recall_ci": [lo[3], hi[3]], "mTTA": base[4]}

    res = {}
    # rule-style single-cue scores (no training)
    for name, col in [("Rule: box-TTC", "max_inv_ttc_path"), ("Rule: flow-divergence TTC", "inv_tau_div"),
                      ("Flow magnitude (no learning)", "flow_mag_center")]:
        res[name] = evaluate(name, df[col].astype(float).fillna(0.0))
    for name, groups in [("Scene only", ["scene"]), ("Kinematic only", ["kinematic"]), ("Motion only", ["motion"]),
                         ("Motion-TTC only", ["motion_ttc"]), ("Kinematic + motion", ["kinematic", "motion"]),
                         ("Fused (all cues)", ["scene", "kinematic", "motion"])]:
        cols = ev.cols_for(groups, cues, X)
        trn = is_fit & use
        m = ev.fit_xgb(X.loc[trn, cols], y[trn])
        res[name] = evaluate(name, pd.Series(m.predict_proba(X[cols])[:, 1], index=df.index))
    for k, v in res.items():
        print(f"{k:32s} AUC {v['AUC']:.3f} [{v['AUC_ci'][0]:.3f},{v['AUC_ci'][1]:.3f}]  AP {v['AP']:.3f}  far {v['far_frames']:.2f}  "
              f"recall {v['recall']:.2f} [{v['recall_ci'][0]:.2f},{v['recall_ci'][1]:.2f}]  mTTA {v['mTTA']:.2f}")
    meta = {"split_source": src, "n_test_clips": int(is_te.groupby(df["clip"]).first().sum()),
            "n_train_clips": int(len(fit_c)), "n_val_clips": int(len(val_c)), "far_target": FAR_T,
            "positive": "last 1 s before onset", "negative": "frames > 2 s before onset"}
    (out / "within_clip.json").write_text(json.dumps({"meta": meta, "results": res}, indent=2, default=float))
    if a.paper_dir:
        short = {"Rule: box-TTC": "WRuleBox", "Rule: flow-divergence TTC": "WRuleDiv", "Flow magnitude (no learning)": "WFlowMag",
                 "Scene only": "WScene", "Kinematic only": "WKin", "Motion only": "WMotion", "Motion-TTC only": "WMotionTTC",
                 "Kinematic + motion": "WKinMotion", "Fused (all cues)": "WFused"}
        f2 = lambda x: f"{x:.2f}"
        M = {"WNTest": meta["n_test_clips"], "WNTrain": meta["n_train_clips"] + meta["n_val_clips"]}
        for k, v in res.items():
            n = short[k]
            M[n + "AUC"], M[n + "AUCLo"], M[n + "AUCHi"] = f2(v["AUC"]), f2(v["AUC_ci"][0]), f2(v["AUC_ci"][1])
            M[n + "AP"] = f2(v["AP"])
            M[n + "Recall"] = f"{100 * v['recall']:.0f}"
            M[n + "RecallLo"], M[n + "RecallHi"] = f"{100 * v['recall_ci'][0]:.0f}", f"{100 * v['recall_ci'][1]:.0f}"
            M[n + "mTTA"] = f2(v["mTTA"])
            M[n + "FAR"] = f"{100 * v['far_frames']:.0f}"
        pd_ = Path(a.paper_dir)
        with open(pd_ / "generated_within.tex", "w", newline="\n") as f:
            f.write("% AUTO-GENERATED by 06_within_clip.py -- do not edit by hand.\n")
            for k, v in M.items():
                f.write("\\def\\" + k + "{" + str(v) + "}\n")
        with open(pd_ / "generated_tab_within.tex", "w", newline="\n") as f:
            f.write("% AUTO-GENERATED by 06_within_clip.py\n")
            f.write("\\begin{tabular}{@{}lllrr@{}}\n\\toprule\n")
            f.write("Score & AUC [95\\% CI] & AP & Recall [95\\% CI] & mTTA\\\\\n\\midrule\n")
            for k, v in res.items():
                if k == "Scene only":
                    f.write("\\midrule\n")
                f.write(f"{k} & {f2(v['AUC'])} [{f2(v['AUC_ci'][0])}, {f2(v['AUC_ci'][1])}] & {f2(v['AP'])} & "
                        f"{100 * v['recall']:.0f} [{100 * v['recall_ci'][0]:.0f}, {100 * v['recall_ci'][1]:.0f}] & {f2(v['mTTA'])}\\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
