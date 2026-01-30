# import time
# import cv2
# import dlib

# def dlib_detector(video_source=0, seconds=5):
#     cap = cv2.VideoCapture(video_source)
#     if not cap.isOpened():
#         return 0

#     detector = dlib.get_frontal_face_detector()

#     start = time.time()
#     frames = 0

#     while time.time() - start < seconds:
#         ok, frame = cap.read()
#         if not ok:
#             continue

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _ = detector(gray, 0)
#         frames += 1

#     cap.release()
#     return frames
