import cv2
import numpy as np
import os
import glob
import json

from lane_detection import LaneDetector

input_dir = 'data/kitti_raw/image_02/data'
output_video_path = 'outputs/lane_detection_output.mp4'
telemetry_log_path = 'outputs/lane_telemetry_log.json'

frame_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
print(f"Found {len(frame_files)} frames")

if len(frame_files) == 0:
    raise FileNotFoundError("No frames found — check input_dir path")

first_frame = cv2.imread(frame_files[0])
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (w, h))

detector = LaneDetector()  # ONE instance, reused across all frames — this is what makes smoothing work
telemetry_log = []

for i, path in enumerate(frame_files):
    img = cv2.imread(path)
    if img is None:
        print(f"Skipping unreadable frame: {path}")
        continue

    annotated, telemetry = detector.process_frame(img)
    out_writer.write(annotated)

    telemetry_log.append({"frame": i, **telemetry})

    if i % 20 == 0:
        print(f"Processed {i}/{len(frame_files)} frames | lane_detected={telemetry['lane_detected']} | offset={telemetry['lane_offset_m']}")

out_writer.release()

with open(telemetry_log_path, 'w') as f:
    json.dump(telemetry_log, f, indent=2)

print("Done.")
print("Video saved to:", output_video_path)
<<<<<<< HEAD
print("Telemetry log saved to:", telemetry_log_path)
=======
print("Telemetry log saved to:", telemetry_log_path)
>>>>>>> c51a5aa84d0e2108c9798d7d459b82950ba0a76e
