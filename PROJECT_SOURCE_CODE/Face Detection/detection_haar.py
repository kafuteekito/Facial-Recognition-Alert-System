import time
import cv2

# Download Haar Cascade (comes with OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces_haar(bgr_img):
    """
    Returns a list of boxes:
    [(x1, y1, x2, y2, score), ...]
    Haar doesn't have a score, so we set it to 1.0.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    detections = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    boxes = []
    for (x, y, w, h) in detections:
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        boxes.append((x1, y1, x2, y2, 1.0))
    return boxes

def run_haar_webcam(cam_index=0, frame_width=960):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    fps_counter = 0
    fps_t0 = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Resize for speed
        if frame_width:
            h0, w0 = frame.shape[:2]
            if w0 > frame_width:
                scale = frame_width / float(w0)
                frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))

        boxes = detect_faces_haar(frame)

        # Draw
        for (x1, y1, x2, y2, _score) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS
        fps_counter += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_counter / dt
            fps_counter = 0
            fps_t0 = time.time()

        cv2.putText(frame, f"Haar | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Haar Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_haar_webcam()
