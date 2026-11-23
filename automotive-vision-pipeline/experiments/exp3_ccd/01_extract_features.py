"""STEP 1: run detection, tracking, lane, TTC and scene analysis on each video -> one CSV per video.

python 01_extract_features.py --limit 10 --demo 3     # smoke test
python 01_extract_features.py --demo 5                # full run (resumable)
"""
import argparse
import time
import traceback
import cv2
import numpy as np
import pandas as pd
import config as C
from utils import list_videos
from perception.detector import Detector
from perception.kinematics import TrackKinematics, frame_summary
from perception.flow import FlowEstimator
from perception.lane import detect_lane
from perception.scene import scene_features
from perception.viz import draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--demo", type=int, default=5, help="save annotated video for the first N newly processed videos")
    ap.add_argument("--demo-ids", default="", help="comma-separated video ids to always render")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: process only videos with index %% n == i (run n workers in parallel)")
    a = ap.parse_args()

    vids = list_videos()
    if a.limit:
        vids = vids[: a.limit]
    si, sn = (int(x) for x in a.shard.split("/"))
    vids = vids[si::sn]
    demo_ids = {s.strip() for s in a.demo_ids.split(",") if s.strip()}
    det = Detector(device=a.device)
    flow = FlowEstimator()
    n_demo, t0 = 0, time.time()

    for k, path in enumerate(vids):
        vid = path.stem
        out_csv = C.FRAME_DIR / f"{vid}.csv"
        force = vid in demo_ids
        if out_csv.exists() and not a.no_resume and not force:
            continue
        want_demo = force or n_demo < a.demo
        try:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS_DEFAULT
            det.reset()
            flow.reset()
            kin, writer, rows, i = None, None, [], 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                H, W = frame.shape[:2]
                if kin is None:
                    kin = TrackKinematics(W, H)
                    if want_demo:
                        writer = cv2.VideoWriter(str(C.DEMO_DIR / f"{vid}_annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
                t = i / fps
                dets = kin.update(t, det.track(frame))
                lane = detect_lane(frame)
                row = {"video_id": vid, "frame_idx": i, "t_sec": t, "fps": fps}
                row.update(frame_summary(dets, W, H))
                row.update({k_: v for k_, v in lane.items() if k_ != "geom"})
                row.update(scene_features(frame))
                row.update(flow.update(frame))
                rows.append(row)
                if writer is not None:
                    writer.write(draw(frame, dets, lane, row))
                i += 1
            cap.release()
            if writer is not None:
                writer.release()
                n_demo += 0 if force else 1
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[{k + 1}/{len(vids)}] {vid}: {len(rows)} frames  ({time.time() - t0:.0f}s elapsed)")
        except Exception:
            with open(C.OUT / "errors.log", "a") as f:
                f.write(f"{vid}\n{traceback.format_exc()}\n")
            print(f"[{k + 1}/{len(vids)}] {vid}: FAILED (see errors.log)")


if __name__ == "__main__":
    main()
