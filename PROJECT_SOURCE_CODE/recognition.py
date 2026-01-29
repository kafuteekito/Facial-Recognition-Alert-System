import os
import json
import numpy as np
from deepface import DeepFace

MODEL_NAME = "ArcFace"
DEFAULT_COSINE_DISTANCE_THRESHOLD = 0.40


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a /= (np.linalg.norm(a) + 1e-12)
    b /= (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))


def embed_face_arcface(face_bgr: np.ndarray) -> np.ndarray:
    rep = DeepFace.represent(
        img_path=face_bgr,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )[0]
    return np.array(rep["embedding"], dtype=np.float32)


def load_gallery_json(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"JSON database not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)

    threshold = float(db.get("threshold", DEFAULT_COSINE_DISTANCE_THRESHOLD))
    people = db.get("people", {})

    gallery = {}
    for name, emb_list in people.items():
        gallery[name] = [np.array(e, dtype=np.float32) for e in emb_list]

    if not gallery:
        raise RuntimeError("JSON database has no people embeddings.")

    return gallery, threshold


def identify(emb: np.ndarray, gallery: dict[str, list[np.ndarray]], threshold: float):
    best_name = "Unknown"
    best_dist = 1e9

    for name, embs in gallery.items():
        for e in embs:
            d = cosine_distance(emb, e)
            if d < best_dist:
                best_dist = d
                best_name = name

    if best_dist > threshold:
        return "Unknown", best_dist
    return best_name, best_dist
