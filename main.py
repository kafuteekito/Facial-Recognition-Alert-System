import os
import time
import datetime
from threading import Thread

import cv2
from detection import detect_faces_retinaface, clamp_box
from recognition import load_gallery_json, embed_face_arcface, identify
from alert import play_alarm, send_email_alert

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_JSON = os.path.join(BASE_DIR, "gallery_db.json")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_captures")
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# Settings
CAM_INDEX = 0
FRAME_WIDTH = 960
PROCESS_EVERY_N_FRAMES = 1
KNOWN_WELCOME_SECONDS = 5
UNKNOWN_ALERT_SECONDS = 10

# Load gallery
gallery, threshold = load_gallery_json(DB_JSON)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_i = 0
known_face_timer = {}
unknown_detected = False
unknown_start_time = None
unknown_alert_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if FRAME_WIDTH:
        h0, w0 = frame.shape[:2]
        if w0 > FRAME_WIDTH:
            scale = FRAME_WIDTH / float(w0)
            frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))

    annotated = frame.copy()
    any_known = False
    any_unknown = False
    present_known_names = set()

    frame_i += 1
    if frame_i % PROCESS_EVERY_N_FRAMES == 0:
        try:
            boxes = detect_faces_retinaface(frame)
            h, w = frame.shape[:2]

            for (x1, y1, x2, y2, _score) in boxes:
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                try:
                    emb = embed_face_arcface(face)
                    name, dist = identify(emb, gallery, threshold=threshold)
                except Exception:
                    name, dist = "Unknown", 1.0

                if name == "Unknown":
                    any_unknown = True
                    color = (0, 0, 255)
                else:
                    any_known = True
                    present_known_names.add(name)
                    color = (0, 255, 0)

                label = f"{name} d={dist:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        except Exception as e:
            print("Recognition error:", e)

    # Known face welcome
    current_time = time.time()
    if any_known:
        unknown_detected = False
        unknown_start_time = None
        unknown_alert_sent = False

        for name in present_known_names:
            if name not in known_face_timer:
                known_face_timer[name] = current_time
            else:
                if current_time - known_face_timer[name] >= KNOWN_WELCOME_SECONDS:
                    cv2.putText(annotated, f"Welcome, {name}!", (50, 80),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

        for k in list(known_face_timer.keys()):
            if k not in present_known_names:
                del known_face_timer[k]

    # Unknown alert
    if any_unknown and not any_known:
        cv2.putText(annotated, "Unknown Face!", (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        if not unknown_detected:
            unknown_detected = True
            unknown_start_time = current_time
            unknown_alert_sent = False
        else:
            elapsed = current_time - unknown_start_time
            print(f"UNKNOWN detected • {elapsed:.1f}s / {UNKNOWN_ALERT_SECONDS}s")
            if not unknown_alert_sent and elapsed >= UNKNOWN_ALERT_SECONDS:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(UNKNOWN_DIR, f"Unknown_{timestamp}.png")
                cv2.imwrite(filename, annotated)
                Thread(target=play_alarm, daemon=True).start()
                Thread(target=send_email_alert, args=(filename,), daemon=True).start()
                unknown_alert_sent = True
                print("ALERT sent (UNKNOWN)")

    if not any_unknown:
        unknown_detected = False
        unknown_start_time = None
        unknown_alert_sent = False

    cv2.imshow("Home Security System", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
