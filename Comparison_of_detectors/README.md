# Face Detection Benchmark (FPS Comparison)

## Overview

This document describes a **standalone benchmark** for comparing the performance of different **face detection models** in terms of **Frames Per Second (FPS)**.

The benchmark is designed to evaluate how classical and deep-learning-based face detectors perform under identical conditions when processing live video from a webcam.

The results are used to justify the choice of detection model in the main security system.

---

## Compared Detection Models

The benchmark evaluates the following face detection approaches:

* **Haar Cascade** (OpenCV-based, classical computer vision)
* **Dlib HOG Detector** (CPU-based, machine-learning approach)
* **RetinaFace** (deep-learning-based face detector)

Each detector is executed separately and measured over the same time interval.

---

## Project Structure

```
benchmark_detection.py

detectors/
├── haar.py          # Haar Cascade detector
├── dlib.py          # Dlib HOG-based detector
└── retinaface.py    # RetinaFace detector

database/
├── database.json
└── comparison.json
```

---

## How the Benchmark Works

1. The webcam stream is opened using OpenCV
2. Each detector is run for a fixed duration (default: 5 seconds)
3. The number of processed frames is counted
4. FPS is calculated using:

```
FPS = processed_frames / elapsed_time
```

This method ensures stable and comparable results across detectors.

---

## Running the Benchmark

From the project root directory, run:

```bash
python benchmark_detection.py
```

The script will sequentially launch:

1. Haar Cascade detector
2. Dlib detector
3. RetinaFace detector

FPS results will be printed to the console.

---

## Output Files

After execution, the benchmark generates the following files:

```
database/database.json
database/comparison.json
```

### database.json

Stores FPS results in a structured format:

```json
{
  "fps": {
    "haar": 22.5,
    "dlib": 12.8,
    "retinaface": 4.3
  }
}
```

### comparison.json

Contains a simplified version of the FPS results, suitable for direct comparison or visualization.

---

## Interpretation of Results

* **Haar Cascade**
  Fastest detector with the highest FPS, but lowest detection accuracy.

* **Dlib**
  Provides a balance between speed and accuracy and works efficiently on CPU.

* **RetinaFace**
  Achieves the highest detection accuracy, but significantly lower FPS on CPU due to deep learning computations.

The benchmark highlights the **trade-off between speed and accuracy**, which is a key consideration in real-time computer vision systems.

---

## Purpose of the Benchmark

This benchmark is intended for:

* Performance evaluation
* Model comparison
* Educational demonstrations
* Justifying design choices in the main project

---

## Conclusion

The benchmark provides experimental evidence for selecting an appropriate face detection model. While RetinaFace is slower, its superior accuracy makes it suitable for security-focused applications where reliability is more important than raw speed.

