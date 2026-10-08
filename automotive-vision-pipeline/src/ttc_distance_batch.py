import cv2
import os
import glob
import json

from src.object_detection import ObjectTracker
from src.ttc_distance import DistanceTTCEstimator

input_dir = 'data/kitti_raw/image_02/data'
output_video_path = 'outputs/ttc_distance_output.mp4'
telemetry_log_path = 'outputs/ttc_telemetry_log.json'

os.makedirs('outputs', exist_ok=True)

frame_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
print(f"Found {len(frame_files)} frames")

if len(frame_files) == 0:
    raise FileNotFoundError("No frames found — check input_dir path")

first_frame = cv2.imread(frame_files[0])
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (w, h))

tracker = ObjectTracker()          # ONE instance — keeps track IDs persistent
ttc_est = DistanceTTCEstimator()   # ONE instance — keeps distance history persistent

telemetry_log = []
fps_assumed = 10  # same timebase as Phase 1 & 2 — keep consistent for later fusion

for i, path in enumerate(frame_files):
    img = cv2.imread(path)
    if img is None:
        print(f"Skipping unreadable frame: {path}")
        continue

    timestamp_sec = i / fps_assumed

    tracked = tracker.track(img, timestamp_sec=timestamp_sec)
    enriched = ttc_est.estimate_distance_and_ttc(tracked, timestamp_sec=timestamp_sec)

    annotated = tracker.draw_tracked_objects(img, tracked)
    annotated = ttc_est.draw_distance_ttc_hud(annotated, enriched)

    out_writer.write(annotated)

    telemetry_log.append({
        "frame": i,
        "timestamp_sec": round(timestamp_sec, 3),
        "objects": enriched
    })

    if i % 20 == 0:
        critical = [o for o in enriched if o["ttc_sec"] <= 2.0]
        print(f"Processed {i}/{len(frame_files)} frames | objects: {len(enriched)} | critical TTC: {len(critical)}")

out_writer.release()

with open(telemetry_log_path, 'w') as f:
    json.dump(telemetry_log, f, indent=2)

print("Done.")
print("Video saved to:", output_video_path)
print("Telemetry log saved to:", telemetry_log_path)