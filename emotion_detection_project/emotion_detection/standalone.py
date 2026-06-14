"""
Real-Time Emotion Detection — Standalone OpenCV Window
Run this directly: python3 standalone.py
No browser needed — uses cv2.imshow() for display.
"""

import cv2
import numpy as np
import time
import threading
import os
import sys
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ── DeepFace lazy loader ───────────────────────────────────────────────────────
_deepface = None
_ready    = False

def _load_deepface():
    global _deepface, _ready
    try:
        from deepface import DeepFace as DF
        dummy = np.zeros((48,48,3), dtype=np.uint8)
        DF.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
        _deepface = DF
        _ready    = True
        print("[INFO] DeepFace loaded ✓")
    except Exception as e:
        print(f"[WARN] DeepFace unavailable: {e}")

threading.Thread(target=_load_deepface, daemon=True).start()

# ── Config ─────────────────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "angry":    (30,  30,  220),
    "disgust":  (30,  180, 30 ),
    "fear":     (180, 30,  200),
    "happy":    (0,   200, 255),
    "sad":      (200, 100, 50 ),
    "surprise": (30,  165, 255),
    "neutral":  (160, 160, 160),
}
EMOJIS = {
    "angry":"😠","disgust":"🤢","fear":"😨",
    "happy":"😊","sad":"😢","surprise":"😲","neutral":"😐"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Analysis thread helpers ───────────────────────────────────────────────────
result_queue  = {}   # face_index -> emotion dict
analyze_lock  = threading.Lock()
pending_faces = []

def analyze_worker(face_index, face_img):
    if not _ready:
        return
    try:
        res = _deepface.analyze(face_img, actions=["emotion"],
                                enforce_detection=False, silent=True)
        if isinstance(res, list): res = res[0]
        with analyze_lock:
            result_queue[face_index] = res.get("emotion", {})
    except Exception:
        pass

# ── Drawing helpers ────────────────────────────────────────────────────────────
def put_text_shadow(img, text, pos, scale, color, thickness=2):
    """Draw text with a drop shadow for readability."""
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_emotion_panel(frame, emotions, x, y, fw, fh):
    """Sidebar emotion bar chart."""
    if not emotions: return
    panel_x = min(x + fw + 10, frame.shape[1] - 140)
    sorted_e = sorted(emotions.items(), key=lambda k: -k[1])[:7]
    panel_h  = len(sorted_e) * 24 + 10

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x-4, y-4), (panel_x+140, y+panel_h),
                  (20,20,20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (panel_x-4, y-4), (panel_x+140, y+panel_h),
                  (60,60,60), 1)

    for i, (name, score) in enumerate(sorted_e):
        by      = y + 4 + i*24
        color   = EMOTION_COLORS.get(name, (200,200,200))
        bar_len = int(score * 1.1)  # max ~110px for 100%
        cv2.rectangle(frame, (panel_x, by), (panel_x+bar_len, by+14), color, -1)
        cv2.putText(frame, f"{name[:4]:4s} {score:5.1f}%",
                    (panel_x+2, by+11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (240,240,240), 1, cv2.LINE_AA)

def draw_face(frame, x, y, fw, fh, dominant, confidence, emotions):
    color = EMOTION_COLORS.get(dominant, (255,255,255))

    # Animated corner brackets
    blen = 20
    for (px,py,dx,dy) in [(x,y,1,1),(x+fw,y,-1,1),(x,y+fh,1,-1),(x+fw,y+fh,-1,-1)]:
        cv2.line(frame, (px,py),(px+dx*blen,py), color, 3)
        cv2.line(frame, (px,py),(px,py+dy*blen), color, 3)

    # Full box (subtle)
    cv2.rectangle(frame,(x,y),(x+fw,y+fh),(*color[:3],), 1)

    # Label pill
    label = f"{dominant.upper()}  {confidence:.0f}%"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pill_y1 = max(y - th - 16, 0)
    pill_y2 = pill_y1 + th + 12
    overlay = frame.copy()
    cv2.rectangle(overlay,(x, pill_y1),(x+tw+16, pill_y2), color, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    put_text_shadow(frame, label, (x+8, pill_y2-6), 0.65, (255,255,255))

    draw_emotion_panel(frame, emotions, x, y, fw, fh)

def draw_hud(frame, fps, n_faces, deepface_status):
    h, w = frame.shape[:2]
    ts   = time.strftime("%H:%M:%S")

    # Bottom info bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-36), (w, h), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    put_text_shadow(frame, f"FPS: {fps:.1f}", (12, h-10), 0.5, (0,200,255))
    put_text_shadow(frame, f"Faces: {n_faces}", (120, h-10), 0.5, (0,200,255))
    put_text_shadow(frame, ts, (w-100, h-10), 0.5, (180,180,180))
    status_c = (0,200,100) if deepface_status else (255,100,0)
    status_t = "DeepFace: READY" if deepface_status else "DeepFace: LOADING"
    put_text_shadow(frame, status_t, (220, h-10), 0.5, status_c)

    # Top title bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0,0), (w,32), (10,10,10), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    put_text_shadow(frame, "EMOTISCAN — REAL-TIME EMOTION DETECTION  [ Q to quit | S to snapshot ]",
                    (12,22), 0.5, (0,200,255), 1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera — check device index")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n🎭  EmotiScan starting...")
    print("   Press Q to quit | Press S to save snapshot\n")

    analyze_every  = 5
    frame_count    = 0
    cached_emo     = {}    # face_idx -> emotions
    fps = 0.0
    t_fps = time.time()
    fps_cnt = 0

    os.makedirs("snapshots", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying…")
            time.sleep(0.05)
            continue

        frame_count += 1
        fps_cnt     += 1
        now = time.time()
        if now - t_fps >= 1.0:
            fps = fps_cnt / (now - t_fps)
            fps_cnt  = 0
            t_fps    = now

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60)
        )

        if frame_count % analyze_every == 0 and _ready:
            for i, (x,y,fw,fh) in enumerate(faces):
                face_crop = frame[y:y+fh, x:x+fw]
                t = threading.Thread(target=analyze_worker,
                                     args=(i, face_crop.copy()), daemon=True)
                t.start()

        # Draw each face
        for i, (x,y,fw,fh) in enumerate(faces):
            with analyze_lock:
                emo = result_queue.get(i, {})
            if emo:
                dom  = max(emo, key=emo.get)
                conf = emo[dom]
            else:
                dom, conf = ("analyzing", 0.0) if _ready else ("loading…", 0.0)
            draw_face(frame, x, y, fw, fh, dom, conf, emo)

        draw_hud(frame, fps, len(faces), _ready)
        cv2.imshow("EmotiScan — Real-Time Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snap_name = f"snapshots/snap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snap_name, frame)
            print(f"[INFO] Snapshot saved → {snap_name}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ EmotiScan closed.")

if __name__ == "__main__":
    main()
