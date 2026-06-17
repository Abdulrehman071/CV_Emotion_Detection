#  EmotiScan — Real-Time Emotion Detection

A complete computer vision project that detects face feature in real time and classifies
7 emotions using **OpenCV** + **DeepFace** (backed by TensorFlow).



## Features

| Feature | Detail |
|---|---|
| Face Detection | OpenCV Haar Cascade (fast, CPU-only) |
| Emotion Engine | DeepFace (FER+ model via TensorFlow) |
| Emotions Detected | angry · disgust · fear · happy · sad · surprise · neutral |
| Live Annotation | Bounding boxes, emotion bars, confidence scores |
| Two run modes | Standalone OpenCV window **or** Web dashboard |
| Snapshot | Save annotated frames as JPEG |
| Timeline | Rolling 60-reading emotion history |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> First run downloads DeepFace models (~500 MB) automatically.

### 2a. Run the Web Dashboard (recommended)
```bash
cd emotion_detection
python3 app.py
```
Open **http://localhost:5000** in your browser, then click **START DETECTION**.

### 2b. Run the Standalone OpenCV window
```bash
python3 standalone.py
```
- Press **Q** to quit
- Press **S** to save a snapshot

---

## Project Structure

```
emotion_detection/
├── app.py            ← Flask server + MJPEG stream + REST API
├── standalone.py     ← Pure OpenCV window version (no browser needed)
├── requirements.txt
├── templates/
│   └── index.html    ← Web dashboard UI
└── snapshots/        ← Saved screenshots (auto-created)
```

---

## API Endpoints (Web Mode)

| Method | Path | Description |
|---|---|---|
| POST | `/start` | Start webcam capture & analysis |
| POST | `/stop` | Stop capture |
| GET | `/video_feed` | MJPEG stream |
| GET | `/status` | JSON: emotions, FPS, stats, history |
| POST | `/snapshot` | Save + return base64 JPEG |

---

## Performance Tips

- **Increase `analyze_every`** in `app.py` (default=5) to run DeepFace less often → higher FPS
- DeepFace runs in a **background thread** so it never blocks the video stream
- First analysis after startup is slower (model warm-up); subsequent calls are fast

---

## Emotion Color Reference

| Emotion | Color |
|---|---|
| 😠 Angry | Red |
| 🤢 Disgust | Green |
| 😨 Fear | Purple |
| 😊 Happy | Yellow |
| 😢 Sad | Blue |
| 😲 Surprise | Orange |
| 😐 Neutral | Gray |
