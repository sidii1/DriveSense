"""STEP 7 (C3): is the late-window loss of detections due to the DETECTOR or to the TRACKER?

For every clip we run, on the same frames and with the same weights/confidence/image size:
  * model.predict  -> raw detections          (no tracker)
  * model.track    -> ByteTrack outputs        (only confirmed tracks are reported)
  * a simple greedy IoU matcher on the raw detections ("raw+IoU"), whose box-growth TTC re-uses the same
    TrackKinematics code, so the two TTC variants differ only in how boxes are linked over time.
One CSV per clip (resumable). Coverage / TTC availability are summarised by 04b_diagnosis_summary.py (here: --summarise).

python 07_detector_diagnosis.py --video-dir <dir> --out <dir> [--shard i/n] [--limit N] [--ids-file ids.txt]
python 07_detector_diagnosis.py --summarise --crash-out <dir> --paper-dir <paper>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class IoUTracker:
    """Greedy IoU association of raw detections; every detection gets an id immediately (no confirmation lag)."""

    def __init__(self, thr=0.3, max_age=3):
        self.thr, self.max_age, self.next_id, self.tracks = thr, max_age, 0, {}

    def reset(self):
        self.next_id, self.tracks = 0, {}

    def update(self, dets):
        used, out = set(), []
        order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
        for i in order:
            d, best, bj = dets[i], 0.0, None
            for tid, t in self.tracks.items():
                if tid in used or t["cls"] != d["cls"]:
                    continue
                s = iou(d["xyxy"], t["xyxy"])
                if s > best:
                    best, bj = s, tid
            if bj is not None and best >= self.thr:
                used.add(bj)
                tid = bj
            else:
                tid = self.next_id
                self.next_id += 1
                used.add(tid)
            self.tracks[tid] = {"xyxy": d["xyxy"], "cls": d["cls"], "age": 0}
            out.append(dict(d, id=tid))
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        return out


def raw_detect(model, frame, device):
    r = model.predict(frame, classes=list(C.CLASS_NAMES), conf=C.CONF, imgsz=C.IMGSZ, verbose=False, device=device)[0]
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        out.append({"xyxy": [float(v) for v in b], "cls": int(c), "conf": float(s), "id": -1})
    return out


def run(a):
    import cv2
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)
    from perception.detector import Detector
    from perception.kinematics import TrackKinematics, frame_summary
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    vids = sorted(Path(a.video_dir).glob("*.mp4"))
    if a.ids_file:
        ids = {l.strip() for l in open(a.ids_file) if l.strip()}
        vids = [v for v in vids if v.stem in ids]
    si, sn = (int(x) for x in a.shard.split("/"))
    vids = vids[si::sn]
    if a.limit:
        vids = vids[: a.limit]
    from ultralytics import YOLO
    raw_model = YOLO(C.YOLO_WEIGHTS)       # separate instance: predict() must not disturb the tracker's predictor state
    det = Detector(device=a.device)
    iout = IoUTracker()
    t0 = time.time()
    for k, p in enumerate(vids):
        out_csv = outdir / f"{p.stem}.csv"
        if out_csv.exists():
            continue
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS_DEFAULT
        det.reset()
        iout.reset()
        kin_i, kin_t, rows, i = None, None, [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            if kin_i is None:
                kin_i, kin_t = TrackKinematics(W, H), TrackKinematics(W, H)
            t = i / fps
            raw = raw_detect(raw_model, frame, a.device)
            trk = det.track(frame)
            d_iou = kin_i.update(t, iout.update([dict(d) for d in raw]))
            si_ = frame_summary(d_iou, W, H)
            rows.append({"video_id": p.stem, "frame_idx": i, "t_sec": t,
                         "n_raw": len(raw), "n_raw_conf50": sum(d["conf"] >= 0.5 for d in raw),
                         "n_track": len(trk), "n_track_id": sum(d["id"] >= 0 for d in trk),
                         "max_inv_ttc_path_iou": si_["max_inv_ttc_path"], "max_inv_ttc_all_iou": si_["max_inv_ttc_all"]})
            i += 1
        cap.release()
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        if (k + 1) % 20 == 0:
            print(f"[{k + 1}/{len(vids)}] {time.time() - t0:.0f}s", flush=True)


def summarise(a):
    """Late-window coverage: raw detector vs ByteTrack, and TTC availability: ByteTrack-based (from main extraction)
    vs raw+IoU. Joins the diagnosis CSVs with the main crash frame features."""
    fr = pd.read_csv(Path(a.crash_out) / "frame_features_all.csv")
    files = sorted(Path(a.diag_dir).glob("*.csv"))
    dg = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dg["video_id"] = dg["video_id"].astype(str)
    fr["video_id"] = fr["video_id"].astype(str).str.zfill(6)
    dg["video_id"] = dg["video_id"].str.zfill(6)
    m = fr.merge(dg, on=["video_id", "frame_idx"], how="inner", suffixes=("", "_dg"))
    m["raw_any"] = (m.n_raw > 0).astype(float)
    m["trk_any"] = (m.n_track > 0).astype(float)
    m["raw_conf50_any"] = (m.n_raw_conf50 > 0).astype(float)
    m["ttc_trk"] = (m.max_inv_ttc_path.fillna(0) > 0).astype(float)
    m["ttc_iou"] = (m.max_inv_ttc_path_iou.fillna(0) > 0).astype(float)
    win = pd.cut(m.t_rel, [-3.0001, -2, -1, 0], right=False, labels=["early", "mid", "late"])
    m["win"] = win
    g = m.dropna(subset=["win"]).groupby("win", observed=True)[["raw_any", "trk_any", "raw_conf50_any", "ttc_trk", "ttc_iou"]].mean()
    n_clips = m.video_id.nunique()
    rng = np.random.default_rng(0)
    clips = m.video_id.unique()
    per = m.dropna(subset=["win"]).groupby(["video_id", "win"], observed=True)[["raw_any", "trk_any", "ttc_trk", "ttc_iou"]].mean()
    drop = {}
    for col in ["raw_any", "trk_any", "ttc_trk", "ttc_iou"]:
        piv = per[col].unstack("win")
        d = (piv["late"] - piv["early"]).dropna().values
        bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(1000)]
        drop[col] = (float(np.mean(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    g.to_csv(Path(a.crash_out) / "eval" / "detector_diagnosis_windows.csv")
    res = {"n_clips": int(n_clips), "windows": g.round(4).to_dict(), "late_minus_early": drop}
    (Path(a.crash_out) / "eval" / "detector_diagnosis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if a.paper_dir:
        M = {"DiagN": n_clips}
        for w, tag in (("early", "Early"), ("late", "Late")):
            M[f"DiagRaw{tag}"] = f"{100 * g.loc[w, 'raw_any']:.0f}"
            M[f"DiagTrk{tag}"] = f"{100 * g.loc[w, 'trk_any']:.0f}"
            M[f"DiagTTCTrk{tag}"] = f"{100 * g.loc[w, 'ttc_trk']:.0f}"
            M[f"DiagTTCIoU{tag}"] = f"{100 * g.loc[w, 'ttc_iou']:.0f}"
        for col, tag in (("raw_any", "Raw"), ("trk_any", "Trk"), ("ttc_trk", "TTCTrk"), ("ttc_iou", "TTCIoU")):
            M[f"DiagDrop{tag}"] = f"{100 * drop[col][0]:.0f}"
            M[f"DiagDrop{tag}Lo"], M[f"DiagDrop{tag}Hi"] = f"{100 * drop[col][1]:.0f}", f"{100 * drop[col][2]:.0f}"
        with open(Path(a.paper_dir) / "generated_diag.tex", "w") as f:
            f.write("% AUTO-GENERATED by 07_detector_diagnosis.py --summarise\n")
            for k, v in M.items():
                f.write(f"\\def\\{k}{{{v}}}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir")
    ap.add_argument("--out")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids-file", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--crash-out", default="results")
    ap.add_argument("--diag-dir", default="results/diag_crash")
    ap.add_argument("--paper-dir", default="")
    a = ap.parse_args()
    summarise(a) if a.summarise else run(a)
