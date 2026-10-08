import cv2
import os
import glob
import json

from src.object_detection import ObjectTracker

input_dir = 'data/kitti_raw/image_02/data'
output_video_path = 'outputs/object_detection_output.mp4'
telemetry_log_path = 'outputs/object_telemetry_log.json'

os.makedirs('outputs', exist_ok=True)

frame_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
print(f"Found {len(frame_files)} frames")

if len(frame_files) == 0:
    raise FileNotFoundError("No frames found — check input_dir path")

first_frame = cv2.imread(frame_files[0])
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (w, h))

tracker = ObjectTracker()   # ONE instance reused — this is what makes track IDs persist
telemetry_log = []

fps_assumed = 10  # matches KITTI capture rate you used in Phase 1

for i, path in enumerate(frame_files):
    img = cv2.imread(path)
    if img is None:
        print(f"Skipping unreadable frame: {path}")
        continue

    timestamp_sec = i / fps_assumed
    tracked_objects = tracker.track(img, timestamp_sec=timestamp_sec)
    annotated = tracker.draw_tracked_objects(img, tracked_objects)

    out_writer.write(annotated)
    telemetry_log.append({
        "frame": i,
        "timestamp_sec": round(timestamp_sec, 3),
        "objects": tracked_objects
    })

    if i % 20 == 0:
        print(f"Processed {i}/{len(frame_files)} frames | objects detected: {len(tracked_objects)}")

out_writer.release()

with open(telemetry_log_path, 'w') as f:
    json.dump(telemetry_log, f, indent=2)

print("Done.")
print("Video saved to:", output_video_path)
print("Telemetry log saved to:", telemetry_log_path)