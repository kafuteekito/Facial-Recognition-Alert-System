import time
import cv2

def haar_detector(video_source=0, seconds=5):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return 0

    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    start = time.time()
    frames = 0

    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _ = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        frames += 1

    cap.release()
    return frames
