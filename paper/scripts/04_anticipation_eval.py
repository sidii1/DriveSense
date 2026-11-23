"""STEP 4: full anticipation evaluation for the IRIS 2027 paper (crash vs normal CCD clips).

Produces every number in Sections 5.2-5.5 of the paper:
  * leakage test      : can the FIRST SECOND of a clip (before any accident) reveal crash-vs-normal source?
  * control comparison: are late-minus-early changes crash-specific (crash vs normal, same relative windows)?
  * anticipation      : rule-based box-TTC alerts vs explainable XGBoost (scene / kinematic / motion / fused)
                        -> AP, ROC-AUC, recall and mean time-to-accident (mTTA) at a fixed false-alarm rate
  * TreeSHAP          : mean |SHAP| per cue and per group on test crash frames
All CIs are bootstrap over CLIPS. Splits are by CLIP, never by frame.

Typical use (run inside experiments/exp3_ccd):
    # 1) extract normal clips with the SAME extractor used for crash clips, e.g.
    #    EXP3_CCD_DIR=<path>/Normal  EXP3_OUT=results_normal  python 01_extract_features.py --demo 0
    # 2) run this script
    python 04_anticipation_eval.py --crash results/frame_features_all.csv \
        --normal-dir results_normal/frame_features --split-dir <CCD dir with train/test lists> \
        --out results/eval --paper-dir <paper folder>

If no official split lists are found, a seeded, stratified 80/20 clip split is used and this is
recorded in eval_summary.json ("split_source") -- state it in the paper.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
GROUPS = {
    "scene": ["brightness", "contrast", "glare_frac", "sharpness", "dark_channel", "saturation"],
    "kinematic": ["max_inv_ttc_path", "max_inv_ttc_all", "has_ttc", "n_vehicles", "n_vru", "n_path",
                  "occupancy", "has_det", "n_tracked"],
    "motion": ["flow_mag_center", "flow_div_center", "flow_div_center_p90", "flow_div_full"],
}
CONFIGS = {  # name -> groups used
    "Scene only": ["scene"],
    "Kinematic only": ["kinematic"],
    "Motion only": ["motion"],
    "Kinematic + motion": ["kinematic", "motion"],
    "Fused (all cues)": ["scene", "kinematic", "motion"],
}
ROLL_K = 10       # 1.0 s causal context at 10 fps
SMOOTH_K = 3      # causal smoothing of alert scores (frames)
WIN_EARLY, WIN_LATE = (-3.0, -2.0), (-1.0, 0.0)


# ----------------------------------------------------------------------------- data loading
def norm_id(s):
    d = re.sub(r"\D", "", re.sub(r"\.(mp4|avi|mov|mkv|npz|npy)$", "", str(s), flags=re.I))
    return d.lstrip("0") or "0"


def load(args):
    cr = pd.read_csv(args.crash)
    assert "t_event" in cr.columns, "crash CSV must come from 02_build_video_features.py (needs t_event)"
    cr["label"] = 1
    files = sorted(Path(args.normal_dir).glob("*.csv"))
    assert files, f"no normal-clip CSVs in {args.normal_dir}"
    nr = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    nr["label"] = 0
    for d in (cr, nr):
        d["vid"] = d["video_id"].map(norm_id)
        d["clip"] = np.where(d["label"] == 1, "crash_", "normal_") + d["vid"]
    # pseudo-onset for normal clips: deterministic draw from the empirical crash-onset distribution,
    # so normal and crash clips are evaluated over spans of the same length distribution.
    onsets = np.sort(cr.groupby("clip")["t_event"].first().values)
    def pseudo(c):
        h = int(hashlib.md5(c.encode()).hexdigest(), 16)
        return float(onsets[h % len(onsets)])
    nr["t_event"] = nr["clip"].map(pseudo)
    nr["t_rel"] = nr["t_sec"] - nr["t_event"]
    keep = sorted(set(cr.columns) & set(nr.columns))
    df = pd.concat([cr[keep], nr[keep]], ignore_index=True).sort_values(["clip", "t_sec"])
    return df.reset_index(drop=True)


def read_split(split_dir, clips):
    """Look for official list files (train*/test*). Each line must contain a clip id; lines with
    'neg'/'normal' are normal clips, 'pos'/'crash' crash clips. Falls back to a seeded split."""
    if split_dir:
        tr = sorted(Path(split_dir).rglob("train*.txt"))
        te = sorted(Path(split_dir).rglob("test*.txt"))
        if tr and te:
            def parse(paths):
                out = set()
                for p in paths:
                    for line in p.read_text(errors="ignore").splitlines():
                        if not line.strip():
                            continue
                        low = line.lower()
                        kind = "normal_" if ("neg" in low or "normal" in low) else "crash_"
                        tok = re.findall(r"\d+", line.split()[0])
                        if tok:
                            out.add(kind + (tok[-1].lstrip("0") or "0"))
                return out
            trs, tes = parse(tr), parse(te)
            got = {c: ("train" if c in trs else "test" if c in tes else None) for c in clips.index}
            cov = np.mean([v is not None for v in got.values()])
            if cov > 0.95:
                return pd.Series(got), f"official ({tr[0].name}/{te[0].name})"
            print(f"WARNING: official lists cover only {cov:.0%} of extracted clips; using seeded split.")
    tr_idx, te_idx = train_test_split(clips.index, test_size=0.2, stratify=clips.values, random_state=SEED)
    s = pd.Series("train", index=clips.index)
    s[te_idx] = "test"
    return s, "seeded stratified 80/20 by clip"


# ----------------------------------------------------------------------------- features
def base_cues(df):
    cues = [c for g in GROUPS.values() for c in g if c in df.columns]
    df = df.copy()
    for c in cues:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("max_inv_ttc_path", "max_inv_ttc_all"):
        if c in df:
            df[c] = df[c].fillna(0.0)
    return df, cues


def causal_features(df, cues):
    g = df.groupby("clip", sort=False)
    out = {}
    for c in cues:
        s = g[c]
        out[f"{c}__last"] = df[c]
        out[f"{c}__mean1s"] = s.transform(lambda x: x.rolling(ROLL_K, min_periods=1).mean())
        out[f"{c}__max1s"] = s.transform(lambda x: x.rolling(ROLL_K, min_periods=1).max())
        out[f"{c}__slope1s"] = s.transform(lambda x: (x - x.shift(ROLL_K - 1).bfill()) / ((ROLL_K - 1) / 10.0))
    return pd.DataFrame(out, index=df.index)


def cols_for(groups, cues, X):
    base = [c for gname in groups for c in GROUPS[gname] if c in cues]
    return [k for k in X.columns if k.split("__")[0] in base]


# ----------------------------------------------------------------------------- alert metrics
def smooth(df, score):
    return pd.Series(score, index=df.index).groupby(df["clip"]).transform(
        lambda x: x.rolling(SMOOTH_K, min_periods=1).mean())


def clip_table(df, s):
    """Per clip: label, max smoothed score in the evaluated span (t < onset), and the score trace."""
    d = df[["clip", "label", "t_sec", "t_event"]].assign(s=s.values)
    d = d[d.t_sec < d.t_event]
    return d


def threshold_at_far(ct, far):
    neg = ct[ct.label == 0].groupby("clip")["s"].max().values
    return float(np.quantile(neg, 1 - far)) + 1e-12 if len(neg) else np.inf


def alert_stats(ct, thr):
    rows = []
    for clip, g in ct.groupby("clip"):
        hit = g[g.s >= thr]
        first = hit.t_sec.min() if len(hit) else np.nan
        rows.append((clip, g.label.iat[0], g.s.max(), g.t_event.iat[0] - first if len(hit) else np.nan))
    return pd.DataFrame(rows, columns=["clip", "label", "score", "tta"]).set_index("clip")


def metrics(at):
    pos, neg = at[at.label == 1], at[at.label == 0]
    m = {"AP": average_precision_score(at.label, at.score) if at.label.nunique() == 2 else np.nan,
         "AUC": roc_auc_score(at.label, at.score) if at.label.nunique() == 2 else np.nan,
         "FAR": float(neg.tta.notna().mean()) if len(neg) else np.nan,
         "Recall": float(pos.tta.notna().mean()) if len(pos) else np.nan,
         "mTTA": float(pos.tta.mean()) if pos.tta.notna().any() else 0.0,
         "R1s": float((pos.tta >= 1.0).mean()), "R2s": float((pos.tta >= 2.0).mean())}
    return m


def boot(at, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    P, N = at[at.label == 1], at[at.label == 0]
    res = []
    for _ in range(n):
        b = pd.concat([P.iloc[rng.integers(0, len(P), len(P))], N.iloc[rng.integers(0, len(N), len(N))]])
        res.append(fn(b))
    return res


def ci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


# ----------------------------------------------------------------------------- experiments
def fit_xgb(X, y, w=None):
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, eval_metric="logloss",
                          scale_pos_weight=spw, random_state=SEED, n_jobs=-1, tree_method="hist")
    m.fit(X, y, sample_weight=w)
    return m


def leakage_test(df, cues, split, args):
    first = df[df.t_sec < 1.0]
    onset = df.groupby("clip")["t_event"].first()
    ok = onset[(onset >= 2.0) | onset.index.str.startswith("normal_")].index  # first second precedes accident by >= 1 s
    agg = first[first["clip"].isin(ok)].groupby("clip")[cues].mean()
    lab = df.groupby("clip")["label"].first().loc[agg.index]
    out = {}
    for name, groups in {"Scene cues only": ["scene"], "All cues": ["scene", "kinematic", "motion"]}.items():
        cols = [c for gname in groups for c in GROUPS[gname] if c in cues]
        tr, te = split.loc[agg.index] == "train", split.loc[agg.index] == "test"
        m = fit_xgb(agg.loc[tr, cols], lab[tr])
        p = m.predict_proba(agg.loc[te, cols])[:, 1]
        at = pd.DataFrame({"label": lab[te].values, "score": p})
        auc = roc_auc_score(at.label, at.score)
        lo, hi = ci(boot(at, lambda b: roc_auc_score(b.label, b.score)))
        out[name] = {"AUC": auc, "lo": lo, "hi": hi, "n_test": int(te.sum()), "n_train": int(tr.sum())}
    return out


def control_comparison(df, cues):
    def win(lo, hi):
        w = df[(df.t_rel >= lo) & (df.t_rel < hi)]
        c = w.groupby("clip").size()
        return w.groupby("clip")[cues].mean()[c >= 3]
    e, l = win(*WIN_EARLY), win(*WIN_LATE)
    idx = e.index.intersection(l.index)
    d = (l.loc[idx] - e.loc[idx])
    lab = df.groupby("clip")["label"].first().loc[idx]
    rows, rng = [], np.random.default_rng(SEED)
    for c in cues:
        a, b = d.loc[lab == 1, c].dropna().values, d.loc[lab == 0, c].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 2 * u.statistic / (len(a) * len(b)) - 1  # rank-biserial: >0 means crash deltas larger
        bs = [np.median(a[rng.integers(0, len(a), len(a))]) - np.median(b[rng.integers(0, len(b), len(b))])
              for _ in range(1000)]
        rows.append(dict(cue=c, n_crash=len(a), n_normal=len(b), med_delta_crash=np.median(a),
                         med_delta_normal=np.median(b), diff=np.median(a) - np.median(b),
                         lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), r_rb=rbc, p=u.pvalue))
    t = pd.DataFrame(rows)
    if len(t):
        order = np.argsort(t.p.values)
        adj, run = np.empty(len(t)), 0.0
        for rank, i in enumerate(order):
            run = max(run, (len(t) - rank) * t.p.values[i]); adj[i] = min(run, 1.0)
        t["p_holm"] = adj
    return t


def anticipation(df, X, cues, split, args):
    lab = df["label"].values
    clip_split = df["clip"].map(split)
    # hold out 20% of TRAIN clips (stratified) to fix the operating point
    tr_clips = split[split == "train"].index
    tr_lab = df.groupby("clip")["label"].first().loc[tr_clips]
    fit_c, val_c = train_test_split(tr_clips, test_size=0.2, stratify=tr_lab.values, random_state=SEED)
    is_fit, is_val, is_te = df["clip"].isin(fit_c), df["clip"].isin(val_c), (clip_split == "test")
    pre = df.t_sec < df.t_event
    # training frames: normal frames before pseudo-onset (neg); crash frames within the horizon (pos)
    pos = (df.label == 1) & (df.t_rel >= -args.horizon) & (df.t_rel < 0)
    neg = (df.label == 0) & pre
    trn = is_fit & (pos | neg)

    results, at_store, models = {}, {}, {}
    # --- rule-based baselines (no learning) ---
    inv = df["max_inv_ttc_path"].values if "max_inv_ttc_path" in df else np.zeros(len(df))
    s_rule = smooth(df, inv)
    ct_val, ct_te = clip_table(df[is_val], s_rule[is_val]), clip_table(df[is_te], s_rule[is_te])
    thr = threshold_at_far(ct_val, args.far)
    at_store["Rule: box-TTC (tuned)"] = alert_stats(ct_te, thr)
    results["Rule: box-TTC (tuned)"] = {"thr_tau_s": (1 / thr) if thr > 0 else np.inf}
    at_store["Rule: box-TTC < 2 s (fixed)"] = alert_stats(ct_te, 0.5)
    results["Rule: box-TTC < 2 s (fixed)"] = {"thr_tau_s": 2.0}
    # --- learned models ---
    for name, groups in CONFIGS.items():
        cols = cols_for(groups, cues, X)
        if not cols:
            continue
        m = fit_xgb(X.loc[trn, cols], lab[trn])
        s = smooth(df, m.predict_proba(X[cols])[:, 1])
        thr = threshold_at_far(clip_table(df[is_val], s[is_val]), args.far)
        at_store[name] = alert_stats(clip_table(df[is_te], s[is_te]), thr)
        results[name] = {"n_features": len(cols)}
        models[name] = (m, cols)
    cols = cols_for(["scene", "kinematic", "motion"], cues, X)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(X.loc[trn, cols].fillna(0), lab[trn])
    s = smooth(df, lr.predict_proba(X[cols].fillna(0))[:, 1])
    thr = threshold_at_far(clip_table(df[is_val], s[is_val]), args.far)
    at_store["Fused, logistic regression"] = alert_stats(clip_table(df[is_te], s[is_te]), thr)
    results["Fused, logistic regression"] = {"n_features": len(cols)}

    for name, at in at_store.items():
        m = metrics(at)
        b = boot(at, metrics, n=args.boot)
        for k in ("AP", "AUC", "Recall", "mTTA"):
            m[k + "_ci"] = ci([x[k] for x in b])
        results[name].update(m)
    # paired bootstrap: fused XGB vs tuned rule, on AP and recall
    ref, alt = at_store["Rule: box-TTC (tuned)"], at_store.get("Fused (all cues)")
    if alt is not None:
        j = pd.concat([ref.add_suffix("_r"), alt.add_suffix("_f")], axis=1)
        j["label"] = j["label_r"]
        def dap(b):
            return (average_precision_score(b.label, b.score_f) - average_precision_score(b.label, b.score_r))
        bs = boot(j, dap, n=args.boot)
        results["delta_AP_fused_minus_rule"] = {"mean": float(dap(j)), "ci": ci(bs)}
    return results, models, (is_te & (df.label == 1) & (df.t_rel >= -args.horizon) & (df.t_rel < 0))


def shap_summary(models, X, mask, out):
    m, cols = models["Fused (all cues)"]
    contrib = m.get_booster().predict(xgb.DMatrix(X.loc[mask, cols]), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=cols)
    by_cue = imp.groupby(lambda k: k.split("__")[0]).sum().sort_values(ascending=False)
    grp = {c: g for g, cs in GROUPS.items() for c in cs}
    by_group = by_cue.groupby(lambda c: grp.get(c, "other")).sum()
    top = by_cue.head(10)[::-1]
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    hatch = {"scene": "", "kinematic": "////", "motion": "...."}
    ax.barh(range(len(top)), top.values, color="white", edgecolor="black", lw=0.6,
            hatch=[hatch.get(grp.get(c), "") for c in top.index])
    ax.set_yticks(range(len(top)), [c.replace("_", r"\_") if False else c for c in top.index], fontsize=6)
    ax.set_xlabel("mean |TreeSHAP| (log-odds), test crash frames, last 2 s")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch=h, label=g) for g, h in hatch.items()],
              frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_shap.pdf")
    return by_cue, by_group


# ----------------------------------------------------------------------------- LaTeX export
def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def write_latex(S, ctrl, paper):
    paper.mkdir(parents=True, exist_ok=True)
    M = {
        "EvalNCrash": S["n_clips"]["crash"], "EvalNNormal": S["n_clips"]["normal"],
        "EvalNTrain": S["n_clips"]["train"], "EvalNTest": S["n_clips"]["test"],
        "EvalSplit": S["split_source"], "EvalFAR": f"{100 * S['far']:.0f}",
        "EvalHorizon": f"{S['horizon']:.1f}",
    }
    for key, name in [("Scene", "Scene cues only"), ("All", "All cues")]:
        r = S["leakage"][name]
        M[f"Leak{key}AUC"], M[f"Leak{key}Lo"], M[f"Leak{key}Hi"] = f2(r["AUC"]), f2(r["lo"]), f2(r["hi"])
    A = S["anticipation"]
    short = {"Rule: box-TTC (tuned)": "RuleTuned", "Rule: box-TTC < 2 s (fixed)": "RuleFixed",
             "Scene only": "Scene", "Kinematic only": "Kin", "Motion only": "Motion",
             "Kinematic + motion": "KinMotion", "Fused (all cues)": "Fused", "Fused, logistic regression": "FusedLR"}
    for name, k in short.items():
        if name in A:
            r = A[name]
            M[f"{k}AP"], M[f"{k}AUC"] = f2(r["AP"]), f2(r["AUC"])
            M[f"{k}Recall"], M[f"{k}mTTA"] = f"{100 * r['Recall']:.0f}", f2(r["mTTA"])
    if "delta_AP_fused_minus_rule" in A:
        d = A["delta_AP_fused_minus_rule"]
        M["DeltaAP"], M["DeltaAPLo"], M["DeltaAPHi"] = f2(d["mean"]), f2(d["ci"][0]), f2(d["ci"][1])
    if "RuleTuned" in [short.get(n) for n in A]:
        M["RuleTau"] = f2(A["Rule: box-TTC (tuned)"]["thr_tau_s"])
    for g, v in S["shap_by_group_share"].items():
        M[f"Shap{g.capitalize()}Share"] = f"{100 * v:.0f}"
    M["ShapTopOne"] = S["shap_top_cues"][0].replace("_", r"\_")
    M["ShapTopTwo"] = S["shap_top_cues"][1].replace("_", r"\_")
    M["ShapTopThree"] = S["shap_top_cues"][2].replace("_", r"\_")
    with open(paper / "generated_eval.tex", "w") as f:
        f.write("% AUTO-GENERATED by 04_anticipation_eval.py -- do not edit by hand.\n\\EvalAvailabletrue\n")
        for k, v in M.items():
            f.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
    # main table body
    with open(paper / "generated_tab_main.tex", "w") as f:
        f.write("% AUTO-GENERATED by 04_anticipation_eval.py\n")
        for name in short:
            if name not in A:
                continue
            r = A[name]
            if name == "Scene only":
                f.write("\\hline\n")
            f.write(f"{name} & {f2(r['AP'])} [{f2(r['AP_ci'][0])}, {f2(r['AP_ci'][1])}] & {f2(r['AUC'])} & "
                    f"{100 * r['FAR']:.0f} & {100 * r['Recall']:.0f} [{100 * r['Recall_ci'][0]:.0f}, {100 * r['Recall_ci'][1]:.0f}] & "
                    f"{f2(r['mTTA'])} & {100 * r['R1s']:.0f}\\\\\n")
    with open(paper / "generated_tab_control.tex", "w") as f:
        f.write("% AUTO-GENERATED by 04_anticipation_eval.py\n")
        for _, r in ctrl.sort_values("p_holm").head(8).iterrows():
            p = "$<$0.001" if r.p_holm < 1e-3 else f"{r.p_holm:.3f}"
            g = lambda x, fmt=".3g": format(x, fmt).replace("-", "$-$")
            f.write(f"\\texttt{{{r.cue.replace('_', chr(92) + '_')}}} & {r.n_crash}/{r.n_normal} & {g(r.med_delta_crash)} & "
                    f"{g(r.med_delta_normal)} & {g(r['diff'])} [{g(r['lo'])}, {g(r['hi'])}] & {g(r.r_rb, '.2f')} & {p}\\\\\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash", required=True)
    ap.add_argument("--normal-dir", required=True)
    ap.add_argument("--split-dir", default="")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--paper-dir", default="")
    ap.add_argument("--far", type=float, default=0.10)
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = load(args)
    df, cues = base_cues(df)
    clips = df.groupby("clip")["label"].first()
    split, src = read_split(args.split_dir, clips)
    print(f"clips: {int(clips.sum())} crash, {int((clips == 0).sum())} normal | split: {src}")
    X = causal_features(df, cues)

    leak = leakage_test(df, cues, split, args)
    print("leakage:", json.dumps(leak, indent=1))
    ctrl = control_comparison(df, cues)
    ctrl.to_csv(out / "control_comparison.csv", index=False)
    print(ctrl.round(4).to_string())
    res, models, shap_mask = anticipation(df, X, cues, split, args)
    by_cue, by_group = shap_summary(models, X, shap_mask, out)
    by_cue.to_csv(out / "shap_by_cue.csv")

    S = {"split_source": src, "far": args.far, "horizon": args.horizon, "boot": args.boot,
         "n_clips": {"crash": int(clips.sum()), "normal": int((clips == 0).sum()),
                     "train": int((split == "train").sum()), "test": int((split == "test").sum())},
         "cues": cues, "leakage": leak, "anticipation": res,
         "shap_by_group_share": (by_group / by_group.sum()).to_dict(),
         "shap_top_cues": list(by_cue.index[:5])}
    (out / "eval_summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith("_ci")} if isinstance(v, dict) else v
                      for k, v in res.items()}, indent=1, default=float))
    if args.paper_dir:
        write_latex(S, ctrl, Path(args.paper_dir))
        import shutil
        shutil.copy(out / "fig_shap.pdf", Path(args.paper_dir) / "figures" / "fig_shap.pdf")
        print("LaTeX fragments written to", args.paper_dir)


if __name__ == "__main__":
    main()
