import cv2
import numpy as np
import os
import glob

def process_frame(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = gray.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(0.0*width), height),
        (int(0.50*width), int(0.47*height)),
        (int(0.60*width), int(0.47*height)),
        (int(0.70*width), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        roi_edges, rho=2, theta=np.pi/180,
        threshold=80, minLineLength=60, maxLineGap=30
    )

    mid_x = width / 2
    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if abs(slope) < 0.5:
                continue
            avg_x = (x1 + x2) / 2
            if slope < 0 and avg_x < mid_x:
                left_lines.append((slope, intercept, length))
            elif slope > 0 and avg_x > mid_x:
                right_lines.append((slope, intercept, length))

    def weighted_avg(lines_with_length):
        if not lines_with_length:
            return None
        total_length = sum(l[2] for l in lines_with_length)
        slope = sum(l[0]*l[2] for l in lines_with_length) / total_length
        intercept = sum(l[1]*l[2] for l in lines_with_length) / total_length
        return slope, intercept

    def make_points(avg, y_min, y_max):
        if avg is None:
            return None
        slope, intercept = avg
        if slope == 0:
            return None
        x_min = int((y_min - intercept) / slope)
        x_max = int((y_max - intercept) / slope)
        return [x_min, y_min, x_max, y_max]

    y_min, y_max = 215, height
    left_lane = make_points(weighted_avg(left_lines), y_min, y_max)
    right_lane = make_points(weighted_avg(right_lines), y_min, y_max)

    out = np.copy(img)
    if left_lane:
        cv2.line(out, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 6)
    if right_lane:
        cv2.line(out, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 0, 255), 6)

    return out


input_dir = 'data/kitti_raw/image_02/data'
output_path = 'outputs/lane_detection_output.mp4'

frame_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
print(f"Found {len(frame_files)} frames")

first_frame = cv2.imread(frame_files[0])
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, 10, (w, h))  # 10 fps, matches KITTI capture rate roughly

for i, path in enumerate(frame_files):
    img = cv2.imread(path)
    if img is None:
        print(f"Skipping unreadable frame: {path}")
        continue
    processed = process_frame(img)
    out_writer.write(processed)
    if i % 20 == 0:
        print(f"Processed {i}/{len(frame_files)} frames")

out_writer.release()
print("Done. Saved to", output_path)