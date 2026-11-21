"""Cheap, interpretable visibility / lighting descriptors."""
import cv2
import numpy as np


def scene_features(frame):
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = cv2.erode(frame.min(axis=2), np.ones((15, 15), np.uint8))
    return {
        "brightness": float(gray.mean() / 255),
        "contrast": float(gray.std() / 255),
        "glare_frac": float((gray >= 245).mean()),           # saturated-pixel fraction
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_channel": float(dark.mean() / 255),            # haze / low-visibility proxy
        "saturation": float(hsv[..., 1].mean() / 255),
    }
