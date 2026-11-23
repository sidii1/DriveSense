"""STEP 4: full anticipation evaluation for the IRIS 2027 paper (crash vs normal CCD clips).

Produces every number in Sections 5.2-5.6 of the paper:
  * leakage test      : can the FIRST SECOND of a clip (before any accident) reveal crash-vs-normal source?
                        (nuisance/encoding cues only, scene, kinematic, motion, all) + source-controlled (matched) protocol
  * control comparison: are late-minus-early changes crash-specific (crash vs normal, same relative windows)?
  * anticipation      : rule-based box-TTC and flow-divergence-TTC alerts vs explainable XGBoost
                        (scene / kinematic / motion / motion-TTC / fused) -> AP, ROC-AUC, recall and mean
                        time-to-accident (mTTA) at a fixed false-alarm rate, operating curves, robustness slices
  * TreeSHAP          : mean |SHAP| per cue and per group on test crash frames
  * sanity checks     : clip-disjoint split, test FAR near target, label-shuffle null, rule == smoothed box-TTC
All CIs are bootstrap over CLIPS. Splits are by CLIP, never by frame.

Typical use (run inside experiments/exp3_ccd):
    python 04_anticipation_eval.py --crash results/frame_features_all.csv \
        --normal-dir results_normal/frame_features --split-dir ../../data/CarCrash/splits \
        --nuisance results/nuisance_crash.csv results/nuisance_normal.csv \
        --out results/eval --paper-dir ../../../paper

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
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    # detector-free time-to-contact from flow divergence: tau = 2 / (fps * div)  (Eq. 3 of the paper)
    "motion_ttc": ["inv_tau_div", "tau_div"],
}
CONFIGS = {  # name -> groups used
    "Scene only": ["scene"],
    "Kinematic only": ["kinematic"],
    "Motion only": ["motion"],
    "Motion-TTC only": ["motion_ttc"],
    "Kinematic + motion": ["kinematic", "motion"],
    "Fused (all cues)": ["scene", "kinematic", "motion"],
    "Fused + motion-TTC": ["scene", "kinematic", "motion", "motion_ttc"],
}
NUISANCE_COLS = ["width", "height", "aspect", "fps_file", "n_frames", "codec_fourcc", "file_bytes", "bytes_per_s",
                 "bits_per_pixel", "blockiness_diff", "blockiness_ratio", "noise_sigma", "letterbox_top",
                 "letterbox_bottom", "letterbox_left", "letterbox_right", "framediff_first1s", "framediff_first1s_max"]
ROLL_K = 10       # 1.0 s causal context at 10 fps
SMOOTH_K = 3      # causal smoothing of alert scores (frames)
WIN_EARLY, WIN_LATE = (-3.0, -2.0), (-1.0, 0.0)
CURVE_FARS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


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
    # same derived columns that 02_build_video_features.py adds to the crash frames
    nr["has_det"] = ((nr["n_vehicles"].fillna(0) + nr["n_vru"].fillna(0)) > 0).astype(float)
    nr["has_ttc"] = (nr["max_inv_ttc_path"].fillna(0) > 0).astype(float)
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
    meta = [c for c in ("timing", "weather", "egoinvolve") if c in cr.columns]
    df = pd.concat([cr[keep + [m for m in meta if m not in keep]], nr[keep]], ignore_index=True)
    df = df.sort_values(["clip", "t_sec"]).reset_index(drop=True)
    return df


def load_nuisance(paths):
    if not paths:
        return None
    parts = []
    for p, lab in zip(paths, (1, 0)):
        d = pd.read_csv(p)
        d["clip"] = ("crash_" if lab == 1 else "normal_") + d["video_id"].map(norm_id)
        d["label"] = lab
        parts.append(d)
    n = pd.concat(parts, ignore_index=True).set_index("clip")
    return n


def read_split(split_dir, clips):
    """Official CCD lists: vgg16_features/train.txt, test.txt with lines 'positive/000353.npz 1' / 'negative/001355.npz 0'.
    Falls back to a seeded stratified split (and says so) if the lists are missing or do not cover the clips."""
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
            assert not (trs & tes), "official train and test lists overlap"
            got = {c: ("train" if c in trs else "test" if c in tes else None) for c in clips.index}
            cov = np.mean([v is not None for v in got.values()])
            if cov > 0.95:
                s = pd.Series(got)
                n_drop = int(s.isna().sum())
                return s.dropna(), f"official CCD lists ({tr[0].name}/{te[0].name})", n_drop
            print(f"WARNING: official lists cover only {cov:.0%} of extracted clips; using seeded split.")
    tr_idx, te_idx = train_test_split(clips.index, test_size=0.2, stratify=clips.values, random_state=SEED)
    s = pd.Series("train", index=clips.index)
    s[te_idx] = "test"
    return s, "seeded stratified 80/20 by clip", 0


# ----------------------------------------------------------------------------- features
def base_cues(df):
    df = df.copy()
    # detector-free time-to-contact from flow divergence (per-frame divergence -> per-second via fps)
    D = pd.to_numeric(df["flow_div_center"], errors="coerce").clip(lower=0) * df["fps"]
    df["inv_tau_div"] = (D / 2.0).fillna(0.0)                           # 1/tau  [s^-1]
    df["tau_div"] = (2.0 / np.maximum(D, 1e-3)).clip(0.1, 10.0).fillna(10.0)   # tau [s], clipped
    cues = [c for g in GROUPS.values() for c in g if c in df.columns]
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
    return pd.Series(np.asarray(score), index=df.index).groupby(df["clip"]).transform(
        lambda x: x.rolling(SMOOTH_K, min_periods=1).mean())


def clip_table(df, s):
    """Per frame before onset: clip, label, t_sec, t_event, smoothed score."""
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
    two = at.label.nunique() == 2
    return {"AP": average_precision_score(at.label, at.score) if two else np.nan,
            "AUC": roc_auc_score(at.label, at.score) if two else np.nan,
            "FAR": float(neg.tta.notna().mean()) if len(neg) else np.nan,
            "Recall": float(pos.tta.notna().mean()) if len(pos) else np.nan,
            "mTTA": float(pos.tta.mean()) if pos.tta.notna().any() else 0.0,
            "R1s": float((pos.tta >= 1.0).mean()), "R2s": float((pos.tta >= 2.0).mean())}


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


# ----------------------------------------------------------------------------- models
def fit_xgb(X, y, w=None, seed=SEED):
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, eval_metric="logloss",
                          scale_pos_weight=spw, random_state=seed, n_jobs=-1, tree_method="hist")
    m.fit(X, y, sample_weight=w)
    return m


# ----------------------------------------------------------------------------- leakage (C1)
def clip_feature_table(df, cues, nuis):
    """One row per clip: first-second means of every visual cue (+ nuisance columns when available)."""
    onset = df.groupby("clip")["t_event"].first()
    ok = onset[(onset >= 2.0) | onset.index.str.startswith("normal_")].index   # 1st second ends >= 1 s before onset
    first = df[(df.t_sec < 1.0) & df["clip"].isin(ok)]
    agg = first.groupby("clip")[cues].mean()
    if nuis is not None:
        agg = agg.join(nuis[NUISANCE_COLS], how="inner")
    return agg


def leak_auc(agg, lab, split, cols, n_boot=1000, seed=SEED):
    tr, te = split.loc[agg.index] == "train", split.loc[agg.index] == "test"
    m = fit_xgb(agg.loc[tr, cols], lab[tr], seed=seed)
    p = m.predict_proba(agg.loc[te, cols])[:, 1]
    at = pd.DataFrame({"label": lab[te].values, "score": p})
    lo, hi = ci(boot(at, lambda b: roc_auc_score(b.label, b.score), n=n_boot))
    return {"AUC": float(roc_auc_score(at.label, at.score)), "lo": lo, "hi": hi,
            "n_test": int(te.sum()), "n_train": int(tr.sum()), "n_test_crash": int(at.label.sum())}


def leakage_sets(cues, have_nuis):
    vis = lambda gs: [c for g in gs for c in GROUPS[g] if c in cues]
    sets = {}
    if have_nuis:
        sets["Nuisance (encoding) only"] = NUISANCE_COLS
    sets["Scene cues only"] = vis(["scene"])
    sets["Kinematic cues only"] = vis(["kinematic"])
    sets["Motion cues only"] = vis(["motion"])
    sets["All visual cues"] = vis(["scene", "kinematic", "motion"])
    if have_nuis:
        sets["Visual + nuisance"] = vis(["scene", "kinematic", "motion"]) + NUISANCE_COLS
    return sets


def leakage_test(df, cues, split, nuis, args, keep_clips=None):
    agg = clip_feature_table(df, cues, nuis)
    if keep_clips is not None:
        agg = agg.loc[agg.index.intersection(keep_clips)]
    lab = df.groupby("clip")["label"].first().loc[agg.index]
    out = {}
    for name, cols in leakage_sets(cues, nuis is not None).items():
        out[name] = leak_auc(agg, lab, split, cols, n_boot=args.boot)
    return out


def nuisance_report(nuis, split):
    """Descriptive comparison of encoding properties between classes (C1a)."""
    rows = []
    for c in NUISANCE_COLS:
        a, b = nuis.loc[nuis.label == 1, c].dropna(), nuis.loc[nuis.label == 0, c].dropna()
        if a.nunique() <= 1 and b.nunique() <= 1:
            rows.append(dict(feature=c, crash_median=a.median(), normal_median=b.median(), auc_single=0.5, constant=True))
            continue
        auc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a.values, b.values])
        rows.append(dict(feature=c, crash_median=a.median(), normal_median=b.median(), auc_single=float(auc), constant=False))
    t = pd.DataFrame(rows)
    t["abs_dev"] = (t.auc_single - 0.5).abs()
    res = {"table": t.sort_values("abs_dev", ascending=False)}
    res["res_crash"] = nuis.loc[nuis.label == 1].groupby(["width", "height"]).size().sort_values(ascending=False).head(5).to_dict()
    res["res_normal"] = nuis.loc[nuis.label == 0].groupby(["width", "height"]).size().sort_values(ascending=False).head(5).to_dict()
    res["codec_crash"] = nuis.loc[nuis.label == 1, "codec"].value_counts().to_dict()
    res["codec_normal"] = nuis.loc[nuis.label == 0, "codec"].value_counts().to_dict()
    res["frac_720p_crash"] = float(((nuis.label == 1) & (nuis.height == 720)).sum() / (nuis.label == 1).sum())
    res["frac_720p_normal"] = float(((nuis.label == 0) & (nuis.height == 720)).sum() / (nuis.label == 0).sum())
    res["fps_crash"] = float(nuis.loc[nuis.label == 1, "fps_file"].median())
    res["fps_normal"] = float(nuis.loc[nuis.label == 0, "fps_file"].median())
    return res


def matched_subset(nuis, split, caliper_sd=0.25, seed=SEED):
    """Source-controlled protocol: 1:1 nearest-neighbour matching of normal to accident clips on the
    out-of-fold propensity (logit) of the nuisance classifier, without replacement, within each split."""
    cols = NUISANCE_COLS
    X, y = nuis[cols], nuis["label"].values
    oof = np.zeros(len(nuis))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = fit_xgb(X.iloc[tr], y[tr], seed=seed)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    eps = 1e-4
    logit = np.log(np.clip(oof, eps, 1 - eps) / (1 - np.clip(oof, eps, 1 - eps)))
    nuis = nuis.assign(ps=oof, logit=logit)
    cal = caliper_sd * nuis["logit"].std()
    rng = np.random.default_rng(seed)
    keep, pairs = [], 0
    for sp in ("train", "test"):
        idx = [c for c in nuis.index if split.get(c) == sp]
        P = nuis.loc[[c for c in idx if nuis.loc[c, "label"] == 1]]
        N = nuis.loc[[c for c in idx if nuis.loc[c, "label"] == 0]]
        free = pd.Series(True, index=N.index)
        for c in rng.permutation(P.index):
            d = (N["logit"] - P.loc[c, "logit"]).abs()
            d = d[free]
            if len(d) == 0:
                break
            j = d.idxmin()
            if d[j] <= cal:
                free[j] = False
                keep += [c, j]
                pairs += 1
    return set(keep), {"pairs": pairs, "caliper_logit": float(cal), "propensity_oof_auc": float(roc_auc_score(y, oof))}


# ----------------------------------------------------------------------------- control comparison
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


# ----------------------------------------------------------------------------- anticipation
def anticipation(df, X, cues, split, args, only=None, shuffle_seed=None):
    lab = df["label"].values
    clip_split = df["clip"].map(split)
    tr_clips = split[split == "train"].index.intersection(df["clip"].unique())
    tr_lab = df.groupby("clip")["label"].first().loc[tr_clips]
    fit_c, val_c = train_test_split(tr_clips, test_size=0.2, stratify=tr_lab.values, random_state=SEED)
    is_fit, is_val, is_te = df["clip"].isin(fit_c), df["clip"].isin(val_c), (clip_split == "test")
    pre = df.t_sec < df.t_event
    pos = (df.label == 1) & (df.t_rel >= -args.horizon) & (df.t_rel < 0)
    neg = (df.label == 0) & pre
    trn = is_fit & (pos | neg)

    results, at_store, models, tables = {}, {}, {}, {}

    def register(name, s, fixed_thr=None, extra=None):
        ct_val, ct_te = clip_table(df[is_val], s[is_val]), clip_table(df[is_te], s[is_te])
        thr = fixed_thr if fixed_thr is not None else threshold_at_far(ct_val, args.far)
        at_store[name] = alert_stats(ct_te, thr)
        tables[name] = (ct_val, ct_te)
        results[name] = dict(extra or {})
        return thr

    def want(name):
        return only is None or name in only

    # --- rule-based baselines (no learning) ---
    inv = df["max_inv_ttc_path"].values if "max_inv_ttc_path" in df else np.zeros(len(df))
    s_rule = smooth(df, inv)
    if want("Rule: box-TTC (tuned)"):
        thr = register("Rule: box-TTC (tuned)", s_rule)
        results["Rule: box-TTC (tuned)"]["thr_tau_s"] = (1 / thr) if thr > 0 else np.inf
    if want("Rule: box-TTC < 2 s (fixed)"):
        register("Rule: box-TTC < 2 s (fixed)", s_rule, fixed_thr=0.5, extra={"thr_tau_s": 2.0})
    if want("Rule: flow-divergence TTC (tuned)"):
        s_div = smooth(df, df["inv_tau_div"].values)
        thr = register("Rule: flow-divergence TTC (tuned)", s_div)
        results["Rule: flow-divergence TTC (tuned)"]["thr_tau_s"] = (1 / thr) if thr > 0 else np.inf
    # --- learned models ---
    for name, groups in CONFIGS.items():
        if not want(name):
            continue
        cols = cols_for(groups, cues, X)
        if not cols:
            continue
        m = fit_xgb(X.loc[trn, cols], lab[trn])
        s = smooth(df, m.predict_proba(X[cols])[:, 1])
        register(name, s, extra={"n_features": len(cols)})
        models[name] = (m, cols)
    if want("Fused, logistic regression"):
        cols = cols_for(["scene", "kinematic", "motion"], cues, X)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(X.loc[trn, cols].fillna(0), lab[trn])
        s = smooth(df, lr.predict_proba(X[cols].fillna(0))[:, 1])
        register("Fused, logistic regression", s, extra={"n_features": len(cols)})

    for name, at in at_store.items():
        m = metrics(at)
        b = boot(at, metrics, n=args.boot)
        for k in ("AP", "AUC", "Recall", "mTTA", "FAR"):
            m[k + "_ci"] = ci([x[k] for x in b])
        results[name].update(m)
    ref, alt = at_store.get("Rule: box-TTC (tuned)"), at_store.get("Fused (all cues)")
    if ref is not None and alt is not None:
        j = pd.concat([ref.add_suffix("_r"), alt.add_suffix("_f")], axis=1)
        j["label"] = j["label_r"]

        def dap(b):
            return (average_precision_score(b.label, b.score_f) - average_precision_score(b.label, b.score_r))
        results["delta_AP_fused_minus_rule"] = {"mean": float(dap(j)), "ci": ci(boot(j, dap, n=args.boot))}
    for a, b_, key in [("Fused (all cues)", "Kinematic only", "delta_AP_fused_minus_kin"),
                       ("Fused (all cues)", "Motion only", "delta_AP_fused_minus_motion"),
                       ("Fused (all cues)", "Scene only", "delta_AP_fused_minus_scene"),
                       ("Kinematic only", "Rule: box-TTC (tuned)", "delta_AP_kin_minus_rule"),
                       ("Motion-TTC only", "Rule: flow-divergence TTC (tuned)", "delta_AP_mttc_minus_rule")]:
        if a in at_store and b_ in at_store:
            j = pd.concat([at_store[b_].add_suffix("_r"), at_store[a].add_suffix("_f")], axis=1)
            j["label"] = j["label_r"]
            f = lambda bb: average_precision_score(bb.label, bb.score_f) - average_precision_score(bb.label, bb.score_r)
            results[key] = {"mean": float(f(j)), "ci": ci(boot(j, f, n=args.boot))}
    shap_mask = is_te & (df.label == 1) & (df.t_rel >= -args.horizon) & (df.t_rel < 0)
    return results, models, shap_mask, at_store, tables


def operating_curves(tables, names, args):
    rows = []
    for name in names:
        if name not in tables:
            continue
        ct_val, ct_te = tables[name]
        for far in CURVE_FARS:
            thr = threshold_at_far(ct_val, far)
            m = metrics(alert_stats(ct_te, thr))
            rows.append(dict(method=name, target_far=far, test_far=m["FAR"], recall=m["Recall"], mTTA=m["mTTA"]))
    return pd.DataFrame(rows)


def plot_operating(curves, out):
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    sty = {"Rule: box-TTC (tuned)": ("-", "o", "box-TTC rule"),
           "Rule: flow-divergence TTC (tuned)": (":", "^", "flow-div. TTC rule"),
           "Kinematic only": ("--", "s", "kinematic"),
           "Fused (all cues)": ("-.", "D", "fused")}
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.8))
    for k, (ls, mk, lab) in sty.items():
        d = curves[curves.method == k]
        if d.empty:
            continue
        axes[0].plot(100 * d.target_far, 100 * d.recall, ls, color="k", lw=1, marker=mk, ms=2.5, label=lab)
        axes[1].plot(100 * d.target_far, d.mTTA, ls, color="k", lw=1, marker=mk, ms=2.5, label=lab)
    axes[0].set_ylabel("recall (%)")
    axes[1].set_ylabel("mTTA (s)")
    for a in axes:
        a.set_xlabel("false-alarm rate (%)")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_operating.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- slices (C6)
def slice_table(df, at_store, names, args):
    meta = df[df.label == 1].groupby("clip")[["timing", "weather", "egoinvolve"]].first()
    slices = {"Day": ("timing", "Day"), "Night": ("timing", "Night"), "Normal weather": ("weather", "Normal"),
              "Rainy": ("weather", "Rainy"), "Snowy": ("weather", "Snowy"), "Ego involved": ("egoinvolve", "Yes"),
              "Ego not involved": ("egoinvolve", "No")}
    rows = []
    for name in names:
        if name not in at_store:
            continue
        at = at_store[name]
        neg = at[at.label == 0]
        for sname, (col, val) in slices.items():
            ids = meta.index[meta[col] == val].intersection(at.index)
            sub = pd.concat([at.loc[ids], neg])
            if len(ids) < 10:
                continue
            m = metrics(sub)
            b = boot(sub, metrics, n=300)
            rows.append(dict(method=name, slice=sname, n_crash=len(ids), AP=m["AP"], AP_lo=ci([x["AP"] for x in b])[0],
                             AP_hi=ci([x["AP"] for x in b])[1], recall=m["Recall"],
                             recall_lo=ci([x["Recall"] for x in b])[0], recall_hi=ci([x["Recall"] for x in b])[1],
                             mTTA=m["mTTA"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- event-aligned dynamics (C2)
def dynamics_figure(df, out, n_boot=300):
    feats = [("flow_mag_center", "central flow magnitude"), ("has_det", "frames with any detection"),
             ("max_inv_ttc_path", "in-path box-TTC $1/\\tau$ (s$^{-1}$)")]
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    rng = np.random.default_rng(SEED)
    d = df[(df.t_rel >= -3.0) & (df.t_rel < 0)].copy()
    d["bin"] = (d.t_rel * 10).round() / 10
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.7))
    summary = {}
    for ax, (f, lab) in zip(axes, feats):
        for label, ls, shade, name in [(1, "-", "0.65", "accident"), (0, "--", "0.88", "normal")]:
            m = d[d.label == label].groupby(["clip", "bin"])[f].mean().unstack("bin")
            a = m.values
            mu = np.nanmean(a, 0)
            bs = np.array([np.nanmean(a[rng.integers(0, len(a), len(a))], 0) for _ in range(n_boot)])
            ax.fill_between(m.columns.values, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0),
                            color=shade, lw=0)
            ax.plot(m.columns.values, mu, ls, color="k", lw=1, label=name)
            summary[f"{f}_{name}_late"] = float(np.nanmean(mu[m.columns.values >= -1.0]))
            summary[f"{f}_{name}_early"] = float(np.nanmean(mu[m.columns.values < -2.0]))
        ax.set_xlabel("time to (pseudo-)onset (s)")
        ax.set_title(lab, fontsize=6.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for b in (-2, -1):
            ax.axvline(b, color="0.5", lw=0.4, ls=":")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_dynamics.pdf")
    plt.close(fig)
    return summary


# ----------------------------------------------------------------------------- SHAP
def shap_summary(models, X, mask, out, name="Fused (all cues)"):
    m, cols = models[name]
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
    ax.set_yticks(range(len(top)), list(top.index), fontsize=6)
    ax.set_xlabel("mean |TreeSHAP| (log-odds), test crash frames, last 2 s")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch=h, label=g) for g, h in hatch.items()],
              frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_shap.pdf")
    plt.close(fig)
    return by_cue, by_group


# ----------------------------------------------------------------------------- sanity checks
def sanity_checks(df, X, cues, split, args, s_rule_cols):
    chk = {}
    tr = set(split[split == "train"].index)
    te = set(split[split == "test"].index)
    chk["train_test_clip_overlap"] = len(tr & te)
    # rule score is exactly smoothed max_inv_ttc_path
    s_chk = smooth(df, df["max_inv_ttc_path"].values)
    ref = df.groupby("clip")["max_inv_ttc_path"].transform(lambda x: x.rolling(SMOOTH_K, min_periods=1).mean())
    chk["rule_score_equals_smoothed_max_inv_ttc_path"] = bool(np.allclose(s_chk.values, ref.values))
    # label-shuffle null: permute clip labels (same prevalence), retrain fused model, evaluate on the same test clips
    rng = np.random.default_rng(123)
    lab_clip = df.groupby("clip")["label"].first()
    perm = pd.Series(rng.permutation(lab_clip.values), index=lab_clip.index)
    d2 = df.copy()
    d2["label"] = d2["clip"].map(perm).values
    r, _, _, at, _ = anticipation(d2, X, cues, split, args, only=["Fused (all cues)"])
    m = r["Fused (all cues)"]
    prev = float(d2.loc[d2["clip"].isin(te)].groupby("clip")["label"].first().mean())
    chk["shuffle_null"] = {"AP": m["AP"], "AUC": m["AUC"], "prevalence_test": prev}
    return chk


# ----------------------------------------------------------------------------- LaTeX export
def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def write_latex(S, ctrl, curves, slices, paper):
    paper.mkdir(parents=True, exist_ok=True)
    M = {
        "EvalNCrash": S["n_clips"]["crash"], "EvalNNormal": S["n_clips"]["normal"],
        "EvalNTrain": S["n_clips"]["train"], "EvalNTest": S["n_clips"]["test"],
        "EvalNTestCrash": S["n_clips"]["test_crash"], "EvalNTestNormal": S["n_clips"]["test_normal"],
        "EvalSplit": S["split_source"], "EvalFAR": f"{100 * S['far']:.0f}",
        "EvalHorizon": f"{S['horizon']:.1f}",
    }
    key = {"Nuisance (encoding) only": "Nuis", "Scene cues only": "Scene", "Kinematic cues only": "Kin",
           "Motion cues only": "Motion", "All visual cues": "All", "Visual + nuisance": "VisNuis"}
    for name, k in key.items():
        for tag, src in (("", S["leakage"]), ("M", S.get("leakage_matched", {}))):
            if name in src:
                r = src[name]
                M[f"Leak{tag}{k}AUC"], M[f"Leak{tag}{k}Lo"], M[f"Leak{tag}{k}Hi"] = f2(r["AUC"]), f2(r["lo"]), f2(r["hi"])
    A = S["anticipation"]
    short = {"Rule: box-TTC (tuned)": "RuleTuned", "Rule: box-TTC < 2 s (fixed)": "RuleFixed",
             "Rule: flow-divergence TTC (tuned)": "RuleDiv",
             "Scene only": "Scene", "Kinematic only": "Kin", "Motion only": "Motion", "Motion-TTC only": "MotionTTC",
             "Kinematic + motion": "KinMotion", "Fused (all cues)": "Fused", "Fused + motion-TTC": "FusedMTTC",
             "Fused, logistic regression": "FusedLR"}
    for tag, AA in (("", A), ("M", S.get("anticipation_matched", {}))):
        for name, k in short.items():
            if name in AA:
                r = AA[name]
                M[f"{tag}{k}AP"], M[f"{tag}{k}APLo"], M[f"{tag}{k}APHi"] = f2(r["AP"]), f2(r["AP_ci"][0]), f2(r["AP_ci"][1])
                M[f"{tag}{k}AUC"] = f2(r["AUC"])
                M[f"{tag}{k}Recall"], M[f"{tag}{k}mTTA"] = f"{100 * r['Recall']:.0f}", f2(r["mTTA"])
                M[f"{tag}{k}RecallLo"], M[f"{tag}{k}RecallHi"] = f"{100 * r['Recall_ci'][0]:.0f}", f"{100 * r['Recall_ci'][1]:.0f}"
                M[f"{tag}{k}FAR"] = f"{100 * r['FAR']:.0f}"
    for dk, nm in [("delta_AP_fused_minus_rule", "DeltaAP"), ("delta_AP_fused_minus_kin", "DeltaAPKin"),
                   ("delta_AP_fused_minus_motion", "DeltaAPMotion"), ("delta_AP_fused_minus_scene", "DeltaAPScene"),
                   ("delta_AP_kin_minus_rule", "DeltaAPKinRule"), ("delta_AP_mttc_minus_rule", "DeltaAPMTTC")]:
        if dk in A:
            M[nm], M[nm + "Lo"], M[nm + "Hi"] = f2(A[dk]["mean"]), f2(A[dk]["ci"][0]), f2(A[dk]["ci"][1])
        if dk in S.get("anticipation_matched", {}):
            d = S["anticipation_matched"][dk]
            M["M" + nm], M["M" + nm + "Lo"], M["M" + nm + "Hi"] = f2(d["mean"]), f2(d["ci"][0]), f2(d["ci"][1])
    if "Rule: box-TTC (tuned)" in A:
        M["RuleTau"] = f2(A["Rule: box-TTC (tuned)"]["thr_tau_s"])
    if "Rule: flow-divergence TTC (tuned)" in A:
        M["RuleDivTau"] = f2(A["Rule: flow-divergence TTC (tuned)"]["thr_tau_s"])
    for g in ("kinematic", "motion", "scene", "motion_ttc"):
        v = S["shap_by_group_share"].get(g, 0.0)
        M[f"Shap{''.join(w.capitalize() for w in g.split('_'))}Share"] = f"{100 * v:.0f}"
    for i, nm in enumerate(("One", "Two", "Three")):
        M[f"ShapTop{nm}"] = S["shap_top_cues"][i].replace("_", r"\_")
    for k, v in S.get("nuisance", {}).get("macros", {}).items():
        M[k] = v
    for k, v in S.get("dynamics", {}).items():
        M["Dyn" + "".join(w.capitalize() for w in re.split(r"[_]", k))] = f2(v)
    if S.get("matching"):
        M["MatchPairs"] = S["matching"]["pairs"]
        M["MatchPropAUC"] = f2(S["matching"]["propensity_oof_auc"])
        M["MatchNCrash"], M["MatchNNormal"] = S["matching"]["n_crash"], S["matching"]["n_normal"]
        M["MatchNTest"] = S["matching"]["n_test"]
    if S.get("sanity"):
        sh = S["sanity"]["shuffle_null"]
        M["ShuffleAP"], M["ShuffleAUC"], M["ShufflePrev"] = f2(sh["AP"]), f2(sh["AUC"]), f2(sh["prevalence_test"])
        M["SanityFARFused"] = f"{100 * A['Fused (all cues)']['FAR']:.0f}"
    # operating curve highlights
    if curves is not None and len(curves):
        for nm, k in (("Fused (all cues)", "Fused"), ("Rule: box-TTC (tuned)", "RuleTuned"), ("Kinematic only", "Kin")):
            for far, tag in ((0.05, "FarLo"), (0.30, "FarHi")):
                r = curves[(curves.method == nm) & (np.isclose(curves.target_far, far))]
                if len(r):
                    M[f"Curve{k}Recall{tag}"] = f"{100 * r.recall.iat[0]:.0f}"
                    M[f"Curve{k}mTTA{tag}"] = f2(r.mTTA.iat[0])
    if slices is not None and len(slices):
        for sname, k in (("Day", "Day"), ("Night", "Night"), ("Normal weather", "WNormal"), ("Rainy", "Rain"),
                         ("Snowy", "Snow"), ("Ego involved", "EgoYes"), ("Ego not involved", "EgoNo")):
            r = slices[(slices.method == "Fused (all cues)") & (slices.slice == sname)]
            if len(r):
                M[f"Slice{k}Recall"] = f"{100 * r.recall.iat[0]:.0f}"
                M[f"Slice{k}RecallLo"], M[f"Slice{k}RecallHi"] = f"{100 * r.recall_lo.iat[0]:.0f}", f"{100 * r.recall_hi.iat[0]:.0f}"
                M[f"Slice{k}AP"] = f2(r.AP.iat[0])
                M[f"Slice{k}N"] = int(r.n_crash.iat[0])
    with open(paper / "generated_eval.tex", "w", newline="\n") as f:
        f.write("% AUTO-GENERATED by 04_anticipation_eval.py -- do not edit by hand.\n\\EvalAvailabletrue\n")
        for k, v in M.items():
            f.write(f"\\def\\{k}{{{v}}}\n")
    # main table body
    with open(paper / "generated_tab_main.tex", "w", newline="\n") as f:
        f.write("% AUTO-GENERATED by 04_anticipation_eval.py\n")
        f.write("\\begin{tabular}{@{}llrrlrr@{}}\n\\toprule\n")
        f.write("Method & AP [95\\% CI] & AUC & FAR & Recall [95\\% CI] & mTTA & R@1s\\\\\n\\midrule\n")
        for name in short:
            if name not in A:
                continue
            r = A[name]
            if name == "Scene only":
                f.write("\\midrule\n")
            lab = name.replace("<", "$<$")
            f.write(f"{lab} & {f2(r['AP'])} [{f2(r['AP_ci'][0])}, {f2(r['AP_ci'][1])}] & {f2(r['AUC'])} & "
                    f"{100 * r['FAR']:.0f} & {100 * r['Recall']:.0f} [{100 * r['Recall_ci'][0]:.0f}, {100 * r['Recall_ci'][1]:.0f}] & "
                    f"{f2(r['mTTA'])} & {100 * r['R1s']:.0f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(paper / "generated_tab_control.tex", "w", newline="\n") as f:
        f.write("% AUTO-GENERATED by 04_anticipation_eval.py\n")
        f.write("\\begin{tabular}{@{}lrrrlrr@{}}\n\\toprule\n")
        f.write("Cue & $n$ (acc./norm.) & Acc. & Norm. & Difference [95\\% CI] & $r_{\\mathrm{rb}}$ & $p_{\\mathrm{Holm}}$\\\\\n\\midrule\n")
        for _, r in ctrl.sort_values("p_holm").head(8).iterrows():
            p = "$<$0.001" if r.p_holm < 1e-3 else f"{r.p_holm:.3f}"
            g = lambda x, fmt=".3g": format(x, fmt).replace("-", "$-$")
            f.write(f"\\texttt{{{r.cue.replace('_', chr(92) + '_')}}} & {r.n_crash}/{r.n_normal} & {g(r.med_delta_crash)} & "
                    f"{g(r.med_delta_normal)} & {g(r['diff'])} [{g(r['lo'])}, {g(r['hi'])}] & {g(r.r_rb, '.2f')} & {p}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(paper / "generated_tab_leak.tex", "w", newline="\n") as f:
        f.write("% AUTO-GENERATED by 04_anticipation_eval.py\n")
        # (a matched-subset column is only meaningful if matching yields a usable test set; with perfectly
        #  separable sources it does not, see the "matching" entry of eval_summary.json)
        f.write("\\begin{tabular}{@{}ll@{}}\n\\toprule\n")
        f.write("Cues from the first second only & Test AUC [95\\% CI]\\\\\n\\midrule\n")
        for name in key:
            if name not in S["leakage"]:
                continue
            r = S["leakage"][name]
            f.write(f"{name} & {f2(r['AUC'])} [{f2(r['lo'])}, {f2(r['hi'])}]\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash", required=True)
    ap.add_argument("--normal-dir", required=True)
    ap.add_argument("--split-dir", default="")
    ap.add_argument("--nuisance", nargs="*", default=[], help="nuisance CSVs: crash then normal")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--paper-dir", default="")
    ap.add_argument("--far", type=float, default=0.10)
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--skip-matched", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = load(args)
    df, cues = base_cues(df)
    nuis = load_nuisance(args.nuisance)
    clips = df.groupby("clip")["label"].first()
    split, src, n_unlisted = read_split(args.split_dir, clips)
    keep = df["clip"].isin(split.index)
    df = df[keep].reset_index(drop=True)
    clips = df.groupby("clip")["label"].first()
    if nuis is not None:
        nuis = nuis.loc[nuis.index.intersection(clips.index)]
    print(f"clips: {int(clips.sum())} crash, {int((clips == 0).sum())} normal | split: {src} | not in official lists: {n_unlisted}")
    clips.to_frame("label").assign(split=split.reindex(clips.index)).to_csv(out / "clips_used.csv")
    X = causal_features(df, cues)

    S = {"split_source": src, "far": args.far, "horizon": args.horizon, "boot": args.boot,
         "n_clips": {"crash": int(clips.sum()), "normal": int((clips == 0).sum()),
                     "train": int((split == "train").sum()), "test": int((split == "test").sum()),
                     "test_crash": int(((split == "test") & (clips.reindex(split.index) == 1)).sum()),
                     "test_normal": int(((split == "test") & (clips.reindex(split.index) == 0)).sum()),
                     "not_in_official_lists": n_unlisted},
         "cues": cues}

    # ---- C1: leakage ----
    leak = leakage_test(df, cues, split, nuis, args)
    S["leakage"] = leak
    print("leakage:", json.dumps(leak, indent=1))
    if nuis is not None:
        rep = nuisance_report(nuis, split)
        rep["table"].to_csv(out / "nuisance_single_feature_auc.csv", index=False)
        top = rep["table"].iloc[0]
        S["nuisance"] = {"res_crash": {f"{k[0]}x{k[1]}": v for k, v in rep["res_crash"].items()},
                         "res_normal": {f"{k[0]}x{k[1]}": v for k, v in rep["res_normal"].items()},
                         "codec_crash": rep["codec_crash"], "codec_normal": rep["codec_normal"],
                         "top_features": rep["table"].head(6).to_dict("records"),
                         "macros": {"Frac720Crash": f"{100 * rep['frac_720p_crash']:.0f}",
                                    "Frac720Normal": f"{100 * rep['frac_720p_normal']:.0f}",
                                    "NuisTopFeat": str(top.feature).replace("_", r"\_"),
                                    "NuisTopAUC": f2(top.auc_single)}}
        print(rep["table"].round(3).to_string())
        if not args.skip_matched:
            kept, minfo = matched_subset(nuis, split)
            lab_n = nuis["label"]
            minfo.update({"n_crash": int(lab_n.loc[list(kept)].sum()), "n_normal": int((lab_n.loc[list(kept)] == 0).sum()),
                          "n_test": int(sum(split.get(c) == "test" for c in kept))})
            S["matching"] = minfo
            S["leakage_matched"] = leakage_test(df, cues, split, nuis, args, keep_clips=pd.Index(sorted(kept)))
            print("matched:", json.dumps(minfo), json.dumps(S["leakage_matched"], indent=1))

    # ---- control comparison (C2b) ----
    ctrl = control_comparison(df, cues)
    ctrl.to_csv(out / "control_comparison.csv", index=False)
    print(ctrl.round(4).to_string())

    # ---- anticipation (incl. rule baselines, motion-TTC) ----
    res, models, shap_mask, at_store, tables = anticipation(df, X, cues, split, args)
    S["anticipation"] = res
    by_cue, by_group = shap_summary(models, X, shap_mask, out)
    by_cue.to_csv(out / "shap_by_cue.csv")
    S["shap_by_group_share"] = (by_group / by_group.sum()).to_dict()
    S["shap_top_cues"] = list(by_cue.index[:5])

    # ---- C5a operating curves, C6 slices, C2a dynamics ----
    curves = operating_curves(tables, ["Rule: box-TTC (tuned)", "Rule: flow-divergence TTC (tuned)",
                                       "Kinematic only", "Fused (all cues)"], args)
    curves.to_csv(out / "operating_curves.csv", index=False)
    plot_operating(curves, out)
    slices = slice_table(df, at_store, ["Rule: box-TTC (tuned)", "Kinematic only", "Fused (all cues)"], args)
    slices.to_csv(out / "robustness_slices.csv", index=False)
    S["dynamics"] = dynamics_figure(df, out)

    # ---- matched (source-controlled) anticipation ----
    if "matching" in S and not args.skip_matched:
        mk = df["clip"].isin(kept)
        dfm = df[mk]
        res_m, _, _, at_m, tab_m = anticipation(dfm, X.loc[dfm.index], cues, split, args)
        S["anticipation_matched"] = res_m
        slices_m = slice_table(dfm, at_m, ["Fused (all cues)"], args)
        slices_m.to_csv(out / "robustness_slices_matched.csv", index=False)

    # ---- sanity checks ----
    S["sanity"] = sanity_checks(df, X, cues, split, args, None)
    print("sanity:", json.dumps(S["sanity"], indent=1, default=float))

    (out / "eval_summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith("_ci")} if isinstance(v, dict) else v
                      for k, v in res.items()}, indent=1, default=float))
    if args.paper_dir:
        pd_ = Path(args.paper_dir)
        write_latex(S, ctrl, curves, slices, pd_)
        import shutil
        (pd_ / "figures").mkdir(exist_ok=True)
        for fn in ("fig_shap.pdf", "fig_dynamics.pdf", "fig_operating.pdf"):
            shutil.copy(out / fn, pd_ / "figures" / fn)
        print("LaTeX fragments written to", args.paper_dir)


if __name__ == "__main__":
    main()
