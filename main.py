import os, json, time, datetime, cv2
import numpy as np
from retinaface import RetinaFace
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_JSON = os.path.join(BASE_DIR, "gallery_db.json")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_captures")

MODEL_NAME = "VGG-Face"
DEFAULT_THRESHOLD = 0.40
UNKNOWN_ALERT_SECONDS = 10

os.makedirs(UNKNOWN_DIR, exist_ok=True)

def cosine_distance(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    a /= (np.linalg.norm(a) + 1e-12)
    b /= (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))

def detect_faces_retinaface(img):
    det = RetinaFace.detect_faces(img)
    boxes = []
    if isinstance(det, dict):
        for info in det.values():
            x1, y1, x2, y2 = info["facial_area"]
            score = float(info.get("score", 0.0))
            boxes.append((x1, y1, x2, y2, score))
    boxes.sort(key=lambda x: x[4], reverse=True)
    return boxes

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def embed_face_vgg(face):
    rep = DeepFace.represent(
        img_path=face, model_name=MODEL_NAME,
        detector_backend="skip", enforce_detection=False, align=False
    )[0]
    return np.array(rep["embedding"], dtype=np.float32)

def load_gallery_json(path):
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)
    thr = float(db.get("threshold", DEFAULT_THRESHOLD))
    people = db.get("people", {})
    gallery = {n: [np.array(e, dtype=np.float32) for e in embs] for n, embs in people.items()}
    return gallery, thr

def identify(emb, gallery, threshold):
    best_name = "Unknown"
    best_dist = 1e9
    for name, embs in gallery.items():
        for e in embs:
            d = cosine_distance(emb, e)
            if d < best_dist:
                best_dist = d
                best_name = name
    return ("Unknown", best_dist) if best_dist > threshold else (best_name, best_dist)

def main():
    gallery, threshold = load_gallery_json(DB_JSON)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    unknown_detected = False
    unknown_start_time = None
    unknown_saved = False

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        annotated = frame.copy()
        h, w = frame.shape[:2]
        boxes = detect_faces_retinaface(frame)

        any_known = False
        any_unknown = False

        for (x1, y1, x2, y2, _score) in boxes:
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            try:
                emb = embed_face_vgg(face)
                name, dist = identify(emb, gallery, threshold)
            except Exception:
                name, dist = "Unknown", 1.0

            if name == "Unknown":
                any_unknown = True
                color = (0, 0, 255)
            else:
                any_known = True
                color = (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{name} d={dist:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        now = time.time()

        if any_unknown and not any_known:
            if not unknown_detected:
                unknown_detected = True
                unknown_start_time = now
                unknown_saved = False

            elapsed = now - (unknown_start_time or now)
            cv2.putText(annotated, f"UNKNOWN {elapsed:.1f}s / {UNKNOWN_ALERT_SECONDS}s", (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

            if (not unknown_saved) and elapsed >= UNKNOWN_ALERT_SECONDS:
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(UNKNOWN_DIR, f"Unknown_{ts}.png")
                cv2.imwrite(filename, annotated)
                unknown_saved = True
                cv2.putText(annotated, "Saved unknown capture", (10, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        else:
            unknown_detected = False
            unknown_start_time = None
            unknown_saved = False

        cv2.imshow("Unknown Save", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
