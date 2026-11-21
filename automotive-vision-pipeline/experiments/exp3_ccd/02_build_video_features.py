"""STEP 2: merge frame CSVs, attach CCD metadata, align to collision onset, QA, build video- and window-level tables."""
import json
import numpy as np
import pandas as pd
import config as C
from utils import norm_key, find_col, find_metadata_csv

AGG = {"n_vehicles": "mean", "n_vru": "mean", "occupancy": "mean", "max_inv_ttc_path": "max",
       "max_inv_ttc_all": "max", "min_dist_path_m": "min", "abs_lane_offset": "mean",
       "abs_lane_curv": "mean", "brightness": "mean", "contrast": "mean", "glare_frac": "mean",
       "sharpness": "mean", "dark_channel": "mean", "flow_div_center": "mean",
       "flow_div_center_p90": "mean", "flow_div_full": "mean", "flow_mag_center": "mean",
       "has_det": "mean", "has_ttc": "mean"}


def main():
    files = sorted(C.FRAME_DIR.glob("*.csv"))
    assert files, "No frame CSVs found. Run 01_extract_features.py first."
    fr = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fr["video_id"] = fr["video_id"].astype(str)
    fr["abs_lane_offset"] = fr["lane_offset"].abs()
    fr["abs_lane_curv"] = fr["lane_curv"].abs()
    fr["has_det"] = ((fr["n_vehicles"].fillna(0) + fr["n_vru"].fillna(0)) > 0).astype(float)
    fr["has_ttc"] = (fr["max_inv_ttc_path"].fillna(0) > 0).astype(float)
    fr["key"] = fr["video_id"].map(norm_key)

    # ---- metadata ----
    meta_cols = {}
    mp = find_metadata_csv()
    if mp is not None:
        meta = pd.read_csv(mp)
        vcol = find_col(meta, ["vidname", "video", "name", "id"])
        meta["key"] = meta[vcol].map(norm_key)
        meta = meta.drop_duplicates("key")
        for k, kws in {"event": ["startframe", "start_frame", "accident_frame", "crash_frame", "event"],
                       "ego": ["ego"], "timing": ["timing", "daynight"], "weather": ["weather"],
                       "pattern": ["pattern", "accident", "type"]}.items():
            meta_cols[k] = find_col(meta.drop(columns=["key"]), kws)
        keep = ["key"] + [c for c in meta_cols.values() if c]
        fr = fr.merge(meta[list(dict.fromkeys(keep))], on="key", how="left")
    print("metadata columns used:", meta_cols)

    # ---- alignment to collision onset ----
    fr["t_event"] = np.nan
    alignment = "clip_end_assumption"
    ev = meta_cols.get("event")
    if ev is not None:
        evf = pd.to_numeric(fr[ev], errors="coerce")
        nfr = fr.groupby("video_id")["frame_idx"].transform("max") + 1
        valid = evf.between(0, nfr)
        if valid.mean() > 0.9:
            fr["t_event"] = evf / fr["fps"]
            alignment = "annotation"
        else:
            print(f"WARNING: '{ev}' values fall outside the clip length for {1 - valid.mean():.0%} of rows; not used.")
    if alignment != "annotation":
        end = fr.groupby("video_id")["t_sec"].transform("max") + 1.0 / fr["fps"]
        fr["t_event"] = end
        print("WARNING: no usable annotation -> t=0 assumed at clip end. State this in the paper.")
    fr["t_rel"] = fr["t_sec"] - fr["t_event"]
    fr["alignment"] = alignment
    fr.to_csv(C.OUT / "frame_features_all.csv", index=False)

    # ---- QA ----
    g = fr.groupby("video_id")
    qa = {
        "n_videos": int(g.ngroups), "n_frames": int(len(fr)), "alignment": alignment,
        "frames_per_video_median": float(g.size().median()),
        "lane_detect_rate": float(fr["lane_ok"].mean()),
        "frames_with_any_detection": float(((fr.n_vehicles + fr.n_vru) > 0).mean()),
        "frames_with_in_path_object": float((fr.n_path > 0).mean()),
        "frames_with_ttc_estimate": float((fr.max_inv_ttc_path > 0).mean()),
        "videos_with_zero_detections": int((g[["n_vehicles", "n_vru"]].sum().sum(axis=1) == 0).sum()),
        "nan_rate": {c: float(fr[c].isna().mean()) for c in ["min_dist_path_m", "lane_offset", "brightness"]},
    }
    (C.OUT / "qa_report.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))

    # ---- window-level + video-level tables ----
    cat_cols = [c for c in meta_cols.values() if c and c != ev]
    first = g[cat_cols].first() if cat_cols else pd.DataFrame(index=g.size().index)
    wrows = []
    for w, (lo, hi) in C.WINDOWS.items():
        sub = fr[(fr.t_rel >= lo) & (fr.t_rel < hi)]
        gg = sub.groupby("video_id")
        cnt = gg.size()
        agg = gg.agg({k: v for k, v in AGG.items() if k in sub.columns})
        agg = agg[cnt >= C.MIN_FRAMES_PER_WINDOW].copy()
        agg["window"], agg["n_frames_in_window"] = w, cnt
        wrows.append(agg.reset_index())
    win = pd.concat(wrows, ignore_index=True).merge(first.reset_index(), on="video_id", how="left")
    win.to_csv(C.OUT / "window_features.csv", index=False)

    vid = g.agg({k: v for k, v in AGG.items() if k in fr.columns})
    vid.columns = [f"{c}_{AGG[c]}" for c in vid.columns]
    vid = vid.join(first).reset_index()
    vid.to_csv(C.OUT / "video_features.csv", index=False)
    cover = win.groupby("window")["video_id"].nunique().to_dict()
    print("videos with enough frames per window:", cover)
    print(f"Saved frame_features_all.csv ({len(fr)} rows), window_features.csv, video_features.csv")


if __name__ == "__main__":
    main()
