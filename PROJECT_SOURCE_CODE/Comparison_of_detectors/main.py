import time
import json
from pathlib import Path

DB_PATH = Path("database/database.json")
COMPARE_PATH = Path("database/comparison.json")

from detectors.haar import haar_detector
from detectors.dlib import dlib_detector
from detectors.retinaface import retinaface_detector


def save_json(path: Path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved: {path}")


def measure_fps(detector_func, video_source=0, seconds=5):
    start_time = time.time()
    frames_processed = detector_func(video_source, seconds=seconds)
    elapsed = time.time() - start_time
    fps = frames_processed / max(elapsed, 1e-9)
    return round(fps, 2)


def main():
    fps_results = {}

    print("Launching Haar Cascade...")
    fps_results["haar"] = measure_fps(haar_detector, seconds=5)

    print("Launching Dlib...")
    fps_results["dlib"] = measure_fps(dlib_detector, seconds=5)

    print("Launching RetinaFace...")
    fps_results["retinaface"] = measure_fps(retinaface_detector, seconds=5)

    print("\nResults FPS:", fps_results)

    save_json(DB_PATH, {"fps": fps_results})
    save_json(COMPARE_PATH, fps_results)


if __name__ == "__main__":
    main()

