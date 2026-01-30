import time
import cv2
import dlib


def detect_faces_dlib(gray_img):
    """
    gray_img: grayscale frame (numpy array)
    return: list of (x1, y1, x2, y2)
    """
    detector = detect_faces_dlib.detector
    rects = detector(gray_img, 0)  # 0 = no upsampling (быстрее)
    boxes = []
    for r in rects:
        boxes.append((r.left(), r.top(), r.right(), r.bottom()))
    return boxes


# создаём детектор один раз (важно для скорости)
detect_faces_dlib.detector = dlib.get_frontal_face_detector()


def run_dlib_webcam(cam_index=0, frame_width=960):
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

        # resize for speed
        if frame_width:
            h0, w0 = frame.shape[:2]
            if w0 > frame_width:
                scale = frame_width / float(w0)
                frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detect_faces_dlib(gray)

        # draw boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS calc
        fps_counter += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_counter / dt
            fps_counter = 0
            fps_t0 = time.time()

        cv2.putText(frame, f"dlib | faces: {len(boxes)} | FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)

        cv2.imshow("dlib Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_dlib_webcam()
