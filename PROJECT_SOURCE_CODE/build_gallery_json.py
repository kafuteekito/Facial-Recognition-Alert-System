import os, json
import cv2
import numpy as np
from retinaface import RetinaFace
from deepface import DeepFace

ENROLL_DIR = "enroll"          # input images (one time only)
OUT_JSON = "gallery_db.json"   # output DB
MODEL_NAME = "ArcFace"
THRESHOLD = 0.40

def cosine_distance(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    a /= (np.linalg.norm(a) + 1e-12)
    b /= (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))

def clamp_box(x1,y1,x2,y2,w,h):
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w-1))
    y2 = max(0, min(int(y2), h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1,y1,x2,y2

def detect_faces_retinaface(bgr_img):
    det = RetinaFace.detect_faces(bgr_img)
    boxes = []
    if isinstance(det, dict):
        for _, info in det.items():
            x1,y1,x2,y2 = info["facial_area"]
            score = float(info.get("score", 0.0))
            boxes.append((x1,y1,x2,y2,score))
    boxes.sort(key=lambda x: x[4], reverse=True)
    return boxes

def embed_face_arcface(face_bgr):
    rep = DeepFace.represent(
        img_path=face_bgr,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )[0]
    return np.array(rep["embedding"], dtype=np.float32)

def main():
    people = {}

    for person in sorted(os.listdir(ENROLL_DIR)):
        pdir = os.path.join(ENROLL_DIR, person)
        if not os.path.isdir(pdir):
            continue

        embs = []
        for fn in sorted(os.listdir(pdir)):
            if not fn.lower().endswith((".jpg",".jpeg",".png")):
                continue
            path = os.path.join(pdir, fn)
            img = cv2.imread(path)
            if img is None:
                print(f"[Skip] can't read {path}")
                continue

            boxes = detect_faces_retinaface(img)
            if not boxes:
                print(f"[Skip] no face {path}")
                continue

            x1,y1,x2,y2,_ = boxes[0]
            h,w = img.shape[:2]
            x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2,w,h)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                print(f"[Skip] empty crop {path}")
                continue

            try:
                emb = embed_face_arcface(face)
                embs.append(emb.tolist())  # JSON-friendly
            except Exception as e:
                print(f"[Skip] embed error {path}: {e}")

        if embs:
            people[person] = embs
            print(f"[OK] {person}: {len(embs)} embeddings")
        else:
            print(f"[Warn] {person}: nothing saved")

    db = {
        "model": MODEL_NAME,
        "metric": "cosine",
        "threshold": THRESHOLD,
        "people": people
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)

    print(f"[Done] Saved: {OUT_JSON}")

if __name__ == "__main__":
    main()
