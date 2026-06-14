"""
Real-Time Emotion Detection Server  ·  EmotiScan v2
─────────────────────────────────────────────────────
Stack : OpenCV  (face detection + MJPEG stream)
        Custom CNN  (emotion classification, ./emotion_model.py)
        Flask  (REST API + HTML dashboard)

Run:    python3 app.py
Visit:  http://localhost:5000
"""

import cv2
import numpy as np
import time
import threading
import os
import logging
import base64
from collections import deque
from datetime import datetime
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

import emotion_model as EM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)
EM.start_loading()

state = {
    "running":        False,
    "frame":          None,
    "emotions":       {},
    "dominant":       "neutral",
    "confidence":     0.0,
    "fps":            0.0,
    "faces_detected": 0,
    "history":        deque(maxlen=60),
    "session_start":  None,
    "total_frames":   0,
    "emotion_counts": {e: 0 for e in EM.EMOTIONS},
}
lock = threading.Lock()

EMOTION_COLORS = {
    "angry":    (30,  30,  220),
    "disgust":  (30,  180, 30 ),
    "fear":     (180, 30,  200),
    "happy":    (0,   200, 255),
    "sad":      (200, 100, 50 ),
    "surprise": (30,  165, 255),
    "neutral":  (160, 160, 160),
}
EMOTION_EMOJI = {
    "angry":"😠","disgust":"🤢","fear":"😨",
    "happy":"😊","sad":"😢","surprise":"😲","neutral":"😐",
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def draw_overlay(frame, faces, emotions_per_face):
    h, w = frame.shape[:2]
    for i, (x, y, fw, fh) in enumerate(faces):
        emo = emotions_per_face[i] if i < len(emotions_per_face) else {}
        if emo:
            dominant = max(emo, key=emo.get)
            color    = EMOTION_COLORS.get(dominant, (200, 200, 200))
        else:
            dominant = "analyzing" if EM.is_ready() else "loading"
            color    = (120, 120, 120)

        blen = 18
        for (px, py, dx, dy) in [
            (x,    y,    1,  1), (x+fw, y,   -1,  1),
            (x,    y+fh, 1, -1), (x+fw, y+fh,-1, -1),
        ]:
            cv2.line(frame, (px, py), (px+dx*blen, py),  color, 3)
            cv2.line(frame, (px, py), (px, py+dy*blen),  color, 3)
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (*color,), 1)

        label = f"{EMOTION_EMOJI.get(dominant,'')} {dominant.upper()}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        pad  = 6
        py0  = max(y - lh - pad*2, 0)
        ov   = frame.copy()
        cv2.rectangle(ov, (x-1, py0), (x+lw+pad*2, py0+lh+pad*2), color, -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, label, (x+pad, py0+lh+pad-1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        if emo:
            bx = min(x + fw + 10, w - 145)
            for j, (ename, score) in enumerate(sorted(emo.items(), key=lambda k: -k[1])[:5]):
                by     = y + j * 22
                filled = int(110 * score / 100)
                bc     = EMOTION_COLORS.get(ename, (160,160,160))
                cv2.rectangle(frame, (bx, by),         (bx+110, by+15), (40,40,40), -1)
                cv2.rectangle(frame, (bx, by),         (bx+filled, by+15), bc,       -1)
                cv2.putText(frame, f"{ename[:4]} {score:.0f}%",
                            (bx+3, by+11), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (230,230,230), 1, cv2.LINE_AA)

    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h-32), (w, h), (8,8,8), -1)
    cv2.addWeighted(ov2, 0.72, frame, 0.28, 0, frame)
    ts  = datetime.now().strftime("%H:%M:%S")
    eng = "CNN READY" if EM.is_ready() else "CNN LOADING"
    cv2.putText(frame, f"FPS {state['fps']:.1f}",   (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces {len(faces)}",       (100, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)
    cv2.putText(frame, eng, (210, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (100,220,100) if EM.is_ready() else (200,150,50), 1, cv2.LINE_AA)
    cv2.putText(frame, ts, (w-75, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)
    return frame


def capture_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    analyze_every = 5
    frame_count   = 0
    cached_emo    = []
    fps_cnt       = 0
    t_fps         = time.time()
    state["session_start"] = datetime.now().isoformat()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame_count            += 1
        fps_cnt                += 1
        state["total_frames"]  += 1

        now = time.time()
        if now - t_fps >= 1.0:
            state["fps"] = fps_cnt / (now - t_fps)
            fps_cnt  = 0
            t_fps    = now

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        state["faces_detected"] = len(faces)

        if frame_count % analyze_every == 0 and EM.is_ready() and len(faces) > 0:
            cached_emo = []
            for (x, y, fw, fh) in faces:
                emo = EM.predict(frame[y:y+fh, x:x+fw])
                cached_emo.append(emo)
            if cached_emo and cached_emo[0]:
                first = cached_emo[0]
                dom   = max(first, key=first.get)
                conf  = first[dom]
                with lock:
                    state["dominant"]   = dom
                    state["confidence"] = conf
                    state["emotions"]   = first
                    state["emotion_counts"][dom] += 1
                    state["history"].append({"ts": time.time(), "emotion": dom, "scores": dict(first)})

        annotated = draw_overlay(frame.copy(), faces, cached_emo)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with lock:
            state["frame"] = buf.tobytes()

    cap.release()
    with lock:
        state["frame"] = None


capture_thread = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    global capture_thread
    if not state["running"]:
        state["running"]       = True
        state["total_frames"]  = 0
        state["emotion_counts"] = {e: 0 for e in EM.EMOTIONS}
        state["history"].clear()
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop():
    state["running"] = False
    return jsonify({"status": "stopped"})


def _gen_frames():
    while True:
        with lock:
            frame = state["frame"]
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with lock:
        hist = list(state["history"])[-20:]
    return jsonify({
        "running":        state["running"],
        "dominant":       state["dominant"],
        "confidence":     round(state["confidence"], 1),
        "emotions":       {k: round(v, 1) for k, v in state["emotions"].items()},
        "fps":            round(state["fps"], 1),
        "faces":          state["faces_detected"],
        "total_frames":   state["total_frames"],
        "model_ready":    EM.is_ready(),
        "emotion_counts": state["emotion_counts"],
        "history":        hist,
        "session_start":  state["session_start"],
    })


@app.route("/snapshot", methods=["POST"])
def snapshot():
    with lock:
        frame = state["frame"]
    if not frame:
        return jsonify({"error": "No frame available"}), 400
    fname = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    os.makedirs("snapshots", exist_ok=True)
    with open(os.path.join("snapshots", fname), "wb") as f:
        f.write(frame)
    return jsonify({"filename": fname, "data": base64.b64encode(frame).decode()})


@app.route("/model_info")
def model_info():
    return jsonify({
        "ready":    EM.is_ready(),
        "emotions": EM.EMOTIONS,
        "backend":  "Custom CNN (VGG-style, 48x48 input, TensorFlow/Keras)",
    })


if __name__ == "__main__":
    print("\n🎭  EmotiScan v2 — http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
