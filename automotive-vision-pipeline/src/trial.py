import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- Step 1: Load & grayscale ----
img = cv2.imread('data/kitti_raw/image_02/data/0000000000.png')
if img is None:
    raise FileNotFoundError("Image not found — check your path")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Image shape:", img.shape)

# ---- Step 2: Blur ----
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# ---- Step 3: Canny ----
edges = cv2.Canny(blur, 50, 150)

# ---- Step 4: ROI mask ----
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(0.0*width), height),
        (int(0.50*width), int(0.47*height)),
        (int(0.60*width), int(0.47*height)),
        (int(0.70*width), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked, mask

roi_edges, mask = region_of_interest(edges)

fig, ax = plt.subplots(1, 3, figsize=(16,5))
ax[0].imshow(edges, cmap='gray'); ax[0].set_title('Full Canny')
ax[1].imshow(mask, cmap='gray'); ax[1].set_title('ROI Mask Shape')
ax[2].imshow(roi_edges, cmap='gray'); ax[2].set_title('Masked Edges')
plt.show()

# ---- Step 5: Hough Transform ----
lines = cv2.HoughLinesP(
    roi_edges,
    rho=2,
    theta=np.pi/180,
    threshold=80,
    minLineLength=60,
    maxLineGap=30
)

print("Type of lines:", type(lines))
print("Lines shape:", None if lines is None else lines.shape)

raw_line_img = np.copy(img)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        cv2.line(raw_line_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

plt.figure(figsize=(12,7))
plt.imshow(cv2.cvtColor(raw_line_img, cv2.COLOR_BGR2RGB))
plt.title('Raw Hough Lines')
plt.show()

# ---- Step 6: Filter + weighted average into two lane lines ----

# Define this BEFORE it's called
def average_slope_intercept(lines_with_length):
    if not lines_with_length:
        return None
    total_length = sum(l[2] for l in lines_with_length)
    weighted_slope = sum(l[0]*l[2] for l in lines_with_length) / total_length
    weighted_intercept = sum(l[1]*l[2] for l in lines_with_length) / total_length
    return weighted_slope, weighted_intercept

def make_line_points(avg, y_min, y_max):
    if avg is None:
        return None
    slope, intercept = avg
    if slope == 0:
        return None
    x_min = int((y_min - intercept) / slope)
    x_max = int((y_max - intercept) / slope)
    return [x_min, y_min, x_max, y_max]

def draw_line(img, line, color, thickness=6):
    if line is None:
        print("No lane found on this side — check filters/ROI")
        return
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)

height, width = gray.shape
mid_x = width / 2

left_lines = []
right_lines = []

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

left_avg = average_slope_intercept(left_lines)
right_avg = average_slope_intercept(right_lines)

y_min = 215
y_max = height

left_lane = make_line_points(left_avg, y_min, y_max)
right_lane = make_line_points(right_avg, y_min, y_max)

line_img = np.copy(img)
draw_line(line_img, left_lane, (255, 0, 0))
draw_line(line_img, right_lane, (0, 0, 255))

plt.figure(figsize=(12,7))
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.title('Extrapolated Lanes')
plt.show()