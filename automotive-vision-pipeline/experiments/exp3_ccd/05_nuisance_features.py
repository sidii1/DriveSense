"""STEP 5 (C1a): low-level container / encoding / image-statistics "nuisance" properties of every clip.

These describe HOW a clip was recorded and encoded (resolution, fps, bitrate, codec, letterboxing, block
artefacts, sensor noise, early frame-to-frame change), not what happens on the road. If they separate accident
from normal clips, part of crash-vs-normal separability is source-driven.

python 05_nuisance_features.py <video_dir> <out_csv> [--workers 4]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

N_FIRST = 10      # first second at 10 fps


def blockiness(g):
    """Mean |luma step| across 8-px block boundaries minus across non-boundaries (JPEG/MPEG block artefacts)."""
    d = np.abs(np.diff(g.astype(np.float32), axis=1))
    cols = np.arange(d.shape[1])
    b = d[:, cols % 8 == 7].mean()
    n = d[:, cols % 8 != 7].mean()
    return float(b - n), float(b / (n + 1e-6))


def noise_level(g):
    """Robust sigma of the high-pass residual (MAD), a sensor/compression noise proxy."""
    r = g.astype(np.float32) - cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def letterbox(g, thr=8):
    """Fraction of border rows / columns that are (almost) black."""
    rows, cols = g.mean(axis=1), g.mean(axis=0)
    def run(v):
        k = 0
        for x in v:
            if x < thr:
                k += 1
            else:
                break
        return k
    H, W = g.shape
    return (run(rows) / H, run(rows[::-1]) / H, run(cols) / W, run(cols[::-1]) / W)


def one(path):
    p = Path(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    W, H = int(cap.get(3)), int(cap.get(4))
    fps, n = cap.get(5), int(cap.get(7))
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fcc >> 8 * i) & 255) for i in range(4)) if fcc else ""
    frames = []
    while len(frames) < N_FIRST:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        return None
    gs = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    size = p.stat().st_size
    dur = max(n / fps, 1e-6) if fps else np.nan
    bl, br = [], []
    for g in gs[::3]:
        bl.append(blockiness(g))
    lb = np.mean([letterbox(g) for g in gs[::3]], axis=0)
    diffs = [np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() for a, b in zip(gs[:-1], gs[1:])]
    bgr = np.mean([f.reshape(-1, 3).mean(0) for f in frames[::3]], axis=0)
    return {"video_id": p.stem, "width": W, "height": H, "aspect": W / H, "fps_file": fps, "n_frames": n,
            "codec_fourcc": fcc, "codec": codec, "file_bytes": size, "bytes_per_s": size / dur,
            "bits_per_pixel": size * 8 / max(n * W * H, 1),
            "blockiness_diff": float(np.mean([b[0] for b in bl])), "blockiness_ratio": float(np.mean([b[1] for b in bl])),
            "noise_sigma": float(np.mean([noise_level(g) for g in gs[::3]])),
            "letterbox_top": lb[0], "letterbox_bottom": lb[1], "letterbox_left": lb[2], "letterbox_right": lb[3],
            "framediff_first1s": float(np.mean(diffs)), "framediff_first1s_max": float(np.max(diffs)),
            "mean_b": bgr[0], "mean_g": bgr[1], "mean_r": bgr[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    vids = sorted(p for p in Path(a.video_dir).rglob("*.mp4"))
    with Pool(a.workers) as pool:
        rows = [r for r in pool.imap(one, [str(v) for v in vids], chunksize=8) if r]
    df = pd.DataFrame(rows)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    print(f"{len(df)}/{len(vids)} clips -> {a.out_csv}")
    print(df.drop(columns=["video_id", "codec"]).describe().T[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
