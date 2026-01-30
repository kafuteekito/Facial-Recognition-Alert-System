import time
import cv2
from retinaface import RetinaFace

def retinaface_detector(video_source=0, seconds=5):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return 0

    start = time.time()
    frames = 0

    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        _ = RetinaFace.detect_faces(frame)
        frames += 1

    cap.release()
    return frames
