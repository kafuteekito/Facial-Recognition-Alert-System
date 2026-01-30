import numpy as np
from retinaface import RetinaFace


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def detect_faces_retinaface(bgr_img: np.ndarray):
    detections = RetinaFace.detect_faces(bgr_img)
    boxes = []

    if isinstance(detections, dict):
        for _, info in detections.items():
            x1, y1, x2, y2 = info["facial_area"]
            score = float(info.get("score", 0.0))
            boxes.append((x1, y1, x2, y2, score))

    boxes.sort(key=lambda x: x[4], reverse=True)
    return boxes
