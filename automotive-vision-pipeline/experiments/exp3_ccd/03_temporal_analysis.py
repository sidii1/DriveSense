"""STEP 3: statistics + figures. Paired within-video tests (no negatives needed), event-aligned curves,
group comparisons and recurring-pattern clustering.  No claim of crash PREDICTION is made here."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from utils import find_metadata_csv, find_col

RNG = np.random.default_rng(0)
TEST_FEATS = ["n_vehicles", "n_vru", "occupancy", "max_inv_ttc_path", "min_dist_path_m",
              "abs_lane_offset", "brightness", "contrast", "glare_frac", "sharpness", "dark_channel",
              "flow_div_center", "flow_mag_center", "has_det", "has_ttc"]
CURVE_FEATS = ["max_inv_ttc_path", "flow_div_center", "has_det", "has_ttc", "n_vehicles", "abs_lane_offset"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def boot_ci(x, fn=np.median, n=2000):
    x = np.asarray(x)
    bs = [fn(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------- A. paired window tests ----------
def window_tests(win):
    rows = []
    for f in TEST_FEATS:
        if f not in win.columns:
            continue
        w = win.pivot(index="video_id", columns="window", values=f)[["early", "mid", "late"]].dropna()
        if len(w) < 10:
            continue
        d = w["late"] - w["early"]
        try:
            fried = stats.friedmanchisquare(w["early"], w["mid"], w["late"]).pvalue
        except Exception:
            fried = np.nan
        try:
            wil = stats.wilcoxon(w["late"], w["early"]).pvalue
        except Exception:
            wil = np.nan
        lo, hi = boot_ci(d)
        rows.append(dict(feature=f, n_videos=len(w), median_early=w["early"].median(),
                         median_mid=w["mid"].median(), median_late=w["late"].median(),
                         median_diff_late_minus_early=d.median(), ci95_low=lo, ci95_high=hi,
                         prop_increase=(d > 0).mean(), prop_decrease=(d < 0).mean(),
                         friedman_p=fried, wilcoxon_p=wil))
    t = pd.DataFrame(rows)
    ok = t.wilcoxon_p.notna()
    t.loc[ok, "wilcoxon_p_holm"] = holm(t.loc[ok, "wilcoxon_p"])
    return t


# ---------- B. event-aligned curves ----------
def curve(fr, feat, ax, label=None):
    if feat not in fr.columns:
        return
    d = fr[(fr.t_rel >= -3.0) & (fr.t_rel <= 1.0)].copy()
    d["bin"] = (d.t_rel * 10).round() / 10
    m = d.groupby(["video_id", "bin"])[feat].mean().unstack("bin")
    if len(m) < 5:
        return
    arr = m.values
    mean = np.nanmean(arr, axis=0)
    bs = np.array([np.nanmean(arr[RNG.integers(0, len(arr), len(arr))], axis=0) for _ in range(300)])
    ax.plot(m.columns, mean, label=label)
    ax.fill_between(m.columns, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0), alpha=0.2)


def plot_curves(fr, ego_col):
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for ax, f in zip(axs.ravel(), CURVE_FEATS):
        curve(fr, f, ax)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(f)
        ax.set_xlabel("time relative to collision onset (s)")
    fig.suptitle("Event-aligned perception indicators (mean, 95% bootstrap CI across videos)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "event_aligned_curves.png", dpi=200)
    plt.close(fig)
    if ego_col and ego_col in fr.columns and fr[ego_col].nunique() >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for lvl, sub in fr.groupby(ego_col):
            if sub.video_id.nunique() >= 10:
                curve(sub, "max_inv_ttc_path", ax, label=f"{ego_col}={lvl} (n={sub.video_id.nunique()})")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("time relative to collision onset (s)")
        ax.set_ylabel("max in-path 1/TTC (1/s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(C.FIG_DIR / "inv_ttc_by_ego_involvement.png", dpi=200)
        plt.close(fig)


# ---------- C. group comparisons ----------
def group_tests(win, cats):
    late = win[win.window == "late"]
    rows = []
    for c in cats:
        if c not in late.columns:
            continue
        lv = late[c].dropna().unique()
        if len(lv) != 2:
            continue
        a, b = (late[late[c] == lv[0]], late[late[c] == lv[1]])
        if min(len(a), len(b)) < 20:
            continue
        for f in TEST_FEATS:
            if f not in late.columns:
                continue
            x, y = a[f].dropna(), b[f].dropna()
            if min(len(x), len(y)) < 10:
                continue
            p = stats.mannwhitneyu(x, y).pvalue
            rows.append(dict(group_col=c, level_a=lv[0], level_b=lv[1], feature=f, n_a=len(x), n_b=len(y),
                             median_a=x.median(), median_b=y.median(), mannwhitney_p=p))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm(t.mannwhitney_p)
    return t


# ---------- D. clustering (numpy only; no scikit-learn) ----------
def kmeans(Z, k, n_init=10, iters=100):
    best = None
    for s in range(n_init):
        rng = np.random.default_rng(s)
        cent = [Z[rng.integers(len(Z))]]
        for _ in range(1, k):                       # k-means++ init
            d2 = np.min([((Z - c) ** 2).sum(1) for c in cent], axis=0)
            cent.append(Z[rng.choice(len(Z), p=d2 / d2.sum())] if d2.sum() > 0 else Z[rng.integers(len(Z))])
        cent = np.array(cent)
        for _ in range(iters):
            lab = ((Z[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
            new = np.array([Z[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
            if np.allclose(new, cent):
                break
            cent = new
        inertia = ((Z - cent[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def silhouette(Z, lab):
    D = np.sqrt(((Z[:, None, :] - Z[None]) ** 2).sum(2))
    sil = []
    for i in range(len(Z)):
        same = lab == lab[i]
        if same.sum() < 2:
            sil.append(0.0)
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i])
        sil.append((b - a) / max(a, b))
    return float(np.mean(sil))


def cluster(win, pattern_col):
    late = win[win.window == "late"].set_index("video_id")
    feats = [f for f in TEST_FEATS if f in late.columns]
    X = late[feats].copy()
    X = X.fillna(X.median())
    if len(X) < 30:
        return None
    sd = X.std().replace(0, 1)
    Z = ((X - X.mean()) / sd).values
    scored = []
    for k in range(2, 7):
        lab_k = kmeans(Z, k)
        scored.append((silhouette(Z, lab_k), k, lab_k))
    sil, k, lab = max(scored, key=lambda t: t[0])
    prof = X.assign(cluster=lab).groupby("cluster").mean()
    prof["n_videos"] = pd.Series(lab).value_counts().sort_index().values
    prof.to_csv(C.OUT / "cluster_profiles.csv")
    if pattern_col and pattern_col in late.columns:
        pd.crosstab(lab, late[pattern_col].values).to_csv(C.OUT / "cluster_vs_pattern.csv")
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    p2 = (Z - Z.mean(0)) @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(p2[:, 0], p2[:, 1], c=lab, cmap="tab10", s=12)
    ax.set_title(f"Video clusters (k={k}, silhouette={sil:.2f})")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "video_clusters_pca.png", dpi=200)
    plt.close(fig)
    return {"k": int(k), "silhouette": float(sil)}


def main():
    fr = pd.read_csv(C.OUT / "frame_features_all.csv")
    win = pd.read_csv(C.OUT / "window_features.csv")
    cats = {}
    mp = find_metadata_csv()
    if mp is not None:
        meta = pd.read_csv(mp)
        for k, kws in {"ego": ["ego"], "timing": ["timing", "daynight"], "weather": ["weather"],
                       "pattern": ["pattern", "accident", "type"]}.items():
            cats[k] = find_col(meta, kws)
    cat_cols = [c for c in cats.values() if c and c in win.columns]
    if "weather" in win.columns:
        win["adverse_weather"] = np.where(win["weather"] == "Normal", "Normal", "Adverse")
        cat_cols = [c for c in cat_cols if c != "weather"] + ["adverse_weather"]

    summary = {"alignment": fr["alignment"].iloc[0], "n_videos": int(fr.video_id.nunique())}
    t = window_tests(win)
    t.to_csv(C.OUT / "temporal_tests.csv", index=False)
    summary["n_videos_all_three_windows"] = int(win.groupby("video_id").window.nunique().eq(3).sum())
    print(t.round(4).to_string())

    plot_curves(fr, cats.get("ego") if cats.get("ego") in fr.columns else None)
    g = group_tests(win, cat_cols)
    g.to_csv(C.OUT / "group_tests.csv", index=False)
    summary["clusters"] = cluster(win, cats.get("pattern"))
    (C.OUT / "exp3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()