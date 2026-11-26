"""STEP 12: seed spread and clip-permutation test for the within-accident Level-3 results.

(i)  Retrains the within-accident XGBoost models with several seeds (the split, features and protocol are those of
     06_within_clip.py; only the XGBoost seed changes) and reports mean / SD / range of
       - original-protocol frame AUC (near = last 1 s, far = > 2 s before onset),
       - time-matched AUC (frames with t in [2.0, 2.9] s, far frames reweighted to the near-frame time distribution),
         for a model trained on all frames ("original model") and one retrained on the matched window ("retrained").
(ii) Clip-permutation p-value for the time-matched AUC: the near/far role is a property of a whole clip in the matched
     window (no clip is in both), so the null shuffles the roles across those clips, recomputes the time weights and the
     weighted AUC (one-sided, H0: AUC = 0.5).

python 12_seeds_permutation.py --crash results/frame_features_all.csv --split-dir ../../data/CarCrash/splits --out results/eval
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

spec = importlib.util.spec_from_file_location("ev", str(Path(__file__).resolve().parent / "04_anticipation_eval.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

NEAR, FARFRAME, TWIN = (-1.0, 0.0), -2.0, (2.0, 2.9)
GROUPS = {"Fused (all cues)": ["scene", "kinematic", "motion"], "Kinematic only": ["kinematic"],
          "Motion only": ["motion"], "Scene only": ["scene"]}


def time_weights(t, y):
    """Weights for far frames so their clip-time histogram (0.1 s bins) matches the near frames'; near weight 1."""
    b = np.round(t * 10).astype(int)
    w = np.ones(len(t))
    for k in np.unique(b):
        pk, nk = ((b == k) & (y == 1)).sum(), ((b == k) & (y == 0)).sum()
        w[(b == k) & (y == 0)] = pk / nk if nk else 0.0
    w[(y == 1) & ~np.isin(b, b[y == 0])] = 0.0           # near frames at times with no far frame cannot be matched
    return w


def wauc(y, s, t):
    w = time_weights(t, y)
    m = w > 0
    return roc_auc_score(y[m], s[m], sample_weight=w[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--perm", type=int, default=10000)
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
    fit_c, _ = train_test_split(tr_clips, test_size=0.2, random_state=ev.SEED)   # same hold-out as 06
    is_fit, is_te = df["clip"].isin(fit_c), sp == "test"
    inwin = (df.t_sec >= TWIN[0] - 1e-6) & (df.t_sec <= TWIN[1] + 1e-6)
    t = df["t_sec"].values

    # training-time weights for the "retrained" model: matched window, fit clips
    trw = is_fit & use & inwin
    w_tr = pd.Series(0.0, index=df.index)
    w_tr[trw] = time_weights(t[trw], y[trw].values)
    trw = trw & (w_tr > 0)
    te = is_te & use & inwin
    te_all = is_te & use
    yt, tt, cl = y[te].values, t[te], df.loc[te, "clip"].values
    ids = np.unique(cl)
    print(f"test clips in window: {len(ids)}; near frames {yt.sum()}, far frames {(1 - yt).sum()}")

    res = {"seeds": list(range(a.seeds)), "models": {}}
    scores = {}
    rng = np.random.default_rng(ev.SEED)
    boot_idx = [rng.choice(ids, len(ids)) for _ in range(a.boot)]
    by_clip = {k: np.where(cl == k)[0] for k in ids}
    for name, groups in GROUPS.items():
        cols = ev.cols_for(groups, cues, X)
        r = {"orig_protocol": [], "tm_original_model": [], "tm_retrained": []}
        for seed in range(a.seeds):
            m1 = ev.fit_xgb(X.loc[is_fit & use, cols], y[is_fit & use], seed=seed)
            s1 = ev.smooth(df, m1.predict_proba(X[cols])[:, 1])
            r["orig_protocol"].append(float(roc_auc_score(y[te_all], s1[te_all])))
            r["tm_original_model"].append(float(wauc(yt, s1[te].values, tt)))
            m2 = ev.fit_xgb(X.loc[trw, cols], y[trw], w=w_tr[trw].values, seed=seed)
            s2 = ev.smooth(df, m2.predict_proba(X[cols])[:, 1])
            r["tm_retrained"].append(float(wauc(yt, s2[te].values, tt)))
            if name == "Fused (all cues)":
                scores[("orig", seed)], scores[("ret", seed)] = s1[te].values, s2[te].values
        for k, v in list(r.items()):
            r[k] = {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "min": min(v), "max": max(v)}
        res["models"][name] = r
        print(name, {k: f"{v['mean']:.3f}±{v['sd']:.3f} [{v['min']:.3f},{v['max']:.3f}]" for k, v in r.items()}, flush=True)

    # clip-bootstrap CI and clip-permutation p for the fused matched AUC (seed 0, and the seed-averaged score)
    def perm_block(label, s):
        obs = wauc(yt, s, tt)
        lo_hi = np.percentile([wauc(*[np.concatenate([a_[ix] for k in b for ix in [by_clip[k]]]) for a_ in (yt, s, tt)])
                               for b in boot_idx], [2.5, 97.5])
        role = np.array([yt[by_clip[k][0]] for k in ids])
        assert all(len(set(yt[by_clip[k]])) == 1 for k in ids), "a clip has both near and far frames in the window"
        prng = np.random.default_rng(ev.SEED + 1)
        null = np.empty(a.perm)
        for i in range(a.perm):
            pr = prng.permutation(role)
            yp = np.empty_like(yt)
            for k, rr in zip(ids, pr):
                yp[by_clip[k]] = rr
            null[i] = wauc(yp, s, tt) if 0 < yp.sum() < len(yp) else 0.5
        p = float((1 + (null >= obs).sum()) / (1 + a.perm))
        return {"AUC": float(obs), "ci": [float(lo_hi[0]), float(lo_hi[1])], "p_perm_one_sided": p,
                "null_mean": float(null.mean()), "null_sd": float(null.std()), "null_q95": float(np.quantile(null, .95)),
                "n_clips_pos": int((role == 1).sum()), "n_clips_neg": int((role == 0).sum())}
    res["permutation"] = {
        "retrained_seed0": perm_block("ret0", scores[("ret", 0)]),
        "retrained_seedavg": perm_block("retavg", np.mean([scores[("ret", s)] for s in range(a.seeds)], 0)),
        "original_seed0": perm_block("orig0", scores[("orig", 0)]),
    }
    for k, v in res["permutation"].items():
        print(k, v)
    (out / "seeds_permutation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
