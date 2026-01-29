import os
import time
import datetime
from threading import Thread
import threading
import queue

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from detection import detect_faces_retinaface, clamp_box
#Using VGG based recognition model:
# from recognition_vgg import load_gallery_json, embed_face_vgg, identify
#Using ArcFace based recognition model:
from recognition import load_gallery_json, embed_face_arcface, identify

from alert import play_alarm, send_email_alert

#Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_JSON = os.path.join(BASE_DIR, "gallery_db.json")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_captures")


CAM_INDEX = 0
FRAME_WIDTH = 960
PROCESS_EVERY_N_FRAMES = 1

KNOWN_WELCOME_SECONDS = 5
UNKNOWN_ALERT_SECONDS = 10

class HomeSecurityGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Home Security System")
        self.root.geometry("980x640")
        self.root.minsize(900, 600)

        # Theme
        self.bg = "#0b1220"
        self.panel = "#0f1b2d"
        self.text = "#e6eefc"
        self.muted = "#9fb0cc"
        self.good = "#37d67a"
        self.bad = "#ff4d4d"
        self.warn = "#ffb020"

        self.root.configure(bg=self.bg)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 16, "bold"))
        style.configure("Muted.TLabel", background=self.bg, foreground=self.muted, font=("Segoe UI", 10))

        # Header
        header = tk.Frame(root, bg=self.bg)
        header.pack(fill="x", padx=18, pady=(16, 10))

        ttk.Label(header, text="Home Security System", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Live Camera • RetinaFace + ArcFace • Alerts",
                  style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

        body = tk.Frame(root, bg=self.bg)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.card = tk.Frame(body, bg=self.panel, highlightthickness=1, highlightbackground="#1b2a44")
        self.card.pack(fill="both", expand=True)

        topbar = tk.Frame(self.card, bg=self.panel)
        topbar.pack(fill="x", padx=14, pady=(12, 10))

        self.status_dot = tk.Canvas(topbar, width=12, height=12, bg=self.panel, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 8))
        self.dot_id = self.status_dot.create_oval(2, 2, 10, 10, fill=self.muted, outline="")

        self.status_label = tk.Label(topbar, text="Starting…", bg=self.panel, fg=self.text,
                                     font=("Segoe UI", 12, "bold"))
        self.status_label.pack(side="left")

        self.right_info = tk.Label(topbar, text="", bg=self.panel, fg=self.muted, font=("Segoe UI", 10))
        self.right_info.pack(side="right")

        self.video_label = tk.Label(self.card, bg="#000000")
        self.video_label.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # State
        self.running = True
        self.cap = None

        self.gallery = {}
        self.threshold = None  
        self.last_ui_imgtk = None

        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)

        # Timers
        self.known_face_timer = {}
        self.unknown_detected = False
        self.unknown_start_time = None
        self.unknown_alert_sent = False

        os.makedirs(UNKNOWN_DIR, exist_ok=True)

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        self._ui_tick()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _set_status(self, text: str, state: str):
        if state == "known":
            color = self.good
        elif state == "unknown":
            color = self.bad
        elif state == "warning":
            color = self.warn
        else:
            color = self.muted

        self.status_label.config(text=text)
        self.status_dot.itemconfigure(self.dot_id, fill=color)

    def _worker_loop(self):
        # Load database
        try:
            self.root.after(0, lambda: self._set_status("Loading face DB from JSON…", "neutral"))
            self.gallery, self.threshold = load_gallery_json(DB_JSON)
            self.root.after(
                0,
                lambda thr=self.threshold, n=len(self.gallery):
                self._set_status(f"DB loaded: {n} people • threshold={thr:.2f}", "neutral")
            )
        except Exception as e:
            err = str(e)
            self.gallery = {}
            self.threshold = None
            self.root.after(0, lambda err=err: self._set_status(f"DB error: {err}", "unknown"))

        # Open camera
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            self.root.after(0, lambda: self._set_status("Camera error: cannot open webcam", "unknown"))
            return

        frame_i = 0
        fps_counter = 0
        fps_t0 = time.time()

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue

            if FRAME_WIDTH:
                h0, w0 = frame.shape[:2]
                if w0 > FRAME_WIDTH:
                    scale = FRAME_WIDTH / float(w0)
                    frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))

            current_time = time.time()
            frame_i += 1

            annotated = frame.copy()
            any_known = False
            any_unknown = False
            present_known_names = set()

            # Recognition only if gallery loaded
            if (frame_i % PROCESS_EVERY_N_FRAMES == 0) and self.gallery and (self.threshold is not None):
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
                            name, dist = identify(emb, self.gallery, threshold=self.threshold)
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
                    err = str(e)
                    self.root.after(0, lambda err=err: self._set_status(f"Recognition error: {err}", "unknown"))

            # Known welcome
            if any_known:
                self.unknown_detected = False
                self.unknown_start_time = None
                self.unknown_alert_sent = False

                for name in present_known_names:
                    if name not in self.known_face_timer:
                        self.known_face_timer[name] = current_time
                    else:
                        if current_time - self.known_face_timer[name] >= KNOWN_WELCOME_SECONDS:
                            cv2.putText(annotated, f"Welcome, {name}!", (50, 80),
                                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

                for k in list(self.known_face_timer.keys()):
                    if k not in present_known_names:
                        del self.known_face_timer[k]

                show_name = next(iter(present_known_names)) if present_known_names else "KNOWN"
                self.root.after(0, lambda n=show_name: self._set_status(f"KNOWN: {n}", "known"))

            # Unknown alert logic
            if any_unknown and not any_known:
                cv2.putText(annotated, "Unknown Face!", (10, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

                if not self.unknown_detected:
                    self.unknown_detected = True
                    self.unknown_start_time = current_time
                    self.unknown_alert_sent = False
                else:
                    elapsed = current_time - (self.unknown_start_time or current_time)
                    self.root.after(0, lambda e=elapsed: self._set_status(
                        f"UNKNOWN detected • {e:.1f}s / {UNKNOWN_ALERT_SECONDS}s", "unknown"
                    ))

                    if (not self.unknown_alert_sent) and (elapsed >= UNKNOWN_ALERT_SECONDS):
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = os.path.join(UNKNOWN_DIR, f"Unknown_{timestamp}.png")
                        cv2.imwrite(filename, annotated)

                        Thread(target=play_alarm, daemon=True).start()
                        Thread(target=send_email_alert, args=(filename,), daemon=True).start()

                        self.unknown_alert_sent = True
                        self.root.after(0, lambda: self._set_status("ALERT sent (UNKNOWN)", "warning"))

            if not any_unknown:
                self.unknown_detected = False
                self.unknown_start_time = None
                self.unknown_alert_sent = False

            if (not any_known) and (not any_unknown):
                if not self.gallery:
                    self.root.after(0, lambda: self._set_status("Live Camera (DB not loaded)", "warning"))
                else:
                    self.root.after(0, lambda: self._set_status("Live Camera", "neutral"))

            # FPS info
            fps_counter += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                fps = fps_counter / dt
                fps_counter = 0
                fps_t0 = time.time()
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.root.after(0, lambda f=fps, n=now_str: self.right_info.config(text=f"{n}   •   FPS {f:.1f}"))

            # Push to UI
            try:
                if self.frame_queue.full():
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(annotated)
            except Exception:
                pass

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def _ui_tick(self):
        try:
            frame = self.frame_queue.get_nowait()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            lw = max(1, self.video_label.winfo_width())
            lh = max(1, self.video_label.winfo_height())

            iw, ih = img.size
            scale = min(lw / iw, lh / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            if nw > 0 and nh > 0:
                img = img.resize((nw, nh), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(img)
            self.last_ui_imgtk = imgtk
            self.video_label.config(image=imgtk)

        except queue.Empty:
            pass
        except Exception:
            pass

        self.root.after(15, self._ui_tick)

    def on_close(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    HomeSecurityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
