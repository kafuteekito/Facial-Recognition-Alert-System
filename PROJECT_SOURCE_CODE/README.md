# Facial Recognition Alert System

## Overview
This project implements a **real-time home security system** based on face detection and face recognition technologies.
The application captures live video from a webcam, detects faces, recognizes known individuals, and triggers alerts when an **unknown person** is detected for a specified amount of time.

The system is implemented as a **desktop application in Python** and combines modern deep-learning models with a graphical user interface.

## Key Features
* Live webcam stream
* Face detection using **RetinaFace**
* Face recognition using **ArcFace embeddings** (via DeepFace)
* Cosine-distance matching against a local face database
* Timer-based logic for known and unknown faces
* Alert system for unknown persons:

  * Screenshot capture
  * Alarm sound
  * Email notification
* Simple and extendable architecture

## Project Structure
```
project/
├── main.py                 # Main application (video loop + logic)
├── detection.py            # Face detection (RetinaFace)
├── recognition.py          # Face recognition (ArcFace)
├── alert.py                # Alarm sound and email notifications
├── build_gallery_json.py         # One-time face enrollment script
├── gallery_db.json         # Face embeddings database
├── Alarm.wav               # Alarm sound
├── enroll/                 # Input images for enrollment
└── unknown_captures/       # Saved screenshots of unknown faces
```

## Dependencies
### External Libraries (pip)

```
opencv-python
numpy
retinaface
deepface
playsound
```
Install them with:

```bash
pip install opencv-python numpy retinaface deepface playsound
```

> Note: `deepface` automatically installs additional dependencies such as TensorFlow.

### Standard Python Modules

The project also uses standard Python modules such as:
`os`, `time`, `datetime`, `threading`, `json`, `smtplib`, `email.message`.

## User Guide
### 1. Prerequisites

* Python **3.9-3.11** (recommended: Python 3.10)
* A working **webcam**
* Internet connection (required on first run to download models)

### 2. Required Files
Make sure the following files are present in the project directory:

```
main.py
detection.py
recognition.py
alert.py
gallery_db.json
Alarm.wav
```
The folder `unknown_captures/` will be created automatically if it does not exist.

### 3. Email Alert Configuration
To enable email alerts, set the following environment variables:

* `ALERT_EMAIL` — sender email address
* `ALERT_EMAIL_PASS` — email password or app password
* `ALERT_TO_EMAIL` — recipient email address

Example (Windows PowerShell):

```powershell
setx ALERT_EMAIL "youremail@gmail.com"
setx ALERT_EMAIL_PASS "your_app_password"
setx ALERT_TO_EMAIL "recipient@gmail.com"
```
If these variables are not set, the application will still run, but email alerts will be skipped.

### 4. Running the Application
From the project directory, run:

```bash
python main.py
```
After launch:

* A window opens with the live camera feed
* Faces are detected and labeled
* Known persons are recognized
* Unknown persons trigger alerts after a delay

### 5. Controls
## Adding Your Own Face (Enrollment Guide)

To recognize a new person, the system must first generate face embeddings and store them in the local database (`gallery_db.json`).
This is done using a **one-time enrollment script** based on **RetinaFace** and **ArcFace**.

### 1. Prepare Enrollment Images

Create a folder called `enroll` in the project directory.

Inside it, create **one subfolder per person**, named with the person’s identity:

```
enroll/
├── Aidin/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── Myktybek/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
```

**Recommendations:**

* Use **3–5 images per person**
* Each image should contain **only one face**
* Use different lighting conditions if possible
* Supported format: `.jpg`

### 2. Run the Enrollment Script

Run the enrollment script:

```bash
python enroll_faces.py
```

### 3. Enrollment Process

For each image, the script:

1. Loads the image
2. Detects the face using RetinaFace
3. Crops the most confident face
4. Generates an **ArcFace embedding**
5. Saves embeddings to `gallery_db.json`

Example database structure:

```json
{
  "model": "ArcFace",
  "metric": "cosine",
  "threshold": 0.40,
  "people": {
    "Aidin": [[...], [...]],
    "Myktybek": [[...]]
  }
}
```

---

### 4. Error Handling During Enrollment

The script automatically skips:

* Images without detectable faces
* Corrupted images
* Invalid face crops
At least **one valid embedding** is required to add a person.

### 5. Using the New Face

After `gallery_db.json` is generated:

1. Ensure it is located next to `main.py`
2. Run the main application again:

```bash
python main.py
```

The newly enrolled person will now be recognized in real time.

---

## How the System Works (Pipeline)

1. Capture frame from webcam (OpenCV)
2. Detect faces using RetinaFace
3. Crop detected faces
4. Generate ArcFace embeddings
5. Compare embeddings using cosine distance
6. Decide **Known / Unknown**
7. Trigger alerts if needed
8. Display results in real time
9. 
## Output

When an unknown person triggers an alert, the system:

* Saves a screenshot to:

  ```
  unknown_captures/Unknown_YYYY-MM-DD_HH-MM-SS.png
  ```
* Plays an alarm sound
* Sends an email notification (if configured)

## Notes

* Accuracy can be improved with more enrollment images
* Performance depends on hardware (GPU recommended for best results)

## Conclusion

This project demonstrates a complete **end-to-end AI-based security system**, combining computer vision, deep learning, real-time processing, and user interaction.
It can be extended further for smart-home, IoT, or cloud-based applications.
