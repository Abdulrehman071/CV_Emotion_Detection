"""
emotion_model.py
Lightweight VGG-style CNN for 7-class facial emotion recognition.
Input:  48×48 grayscale face crop (normalized 0–1)
Output: softmax probabilities over 7 emotions

Weight loading priority:
  1. Local  ./models/emotion_weights.h5
  2. DeepFace cache  ~/.deepface/weights/facial_expression_model_weights.h5
  3. Random init  (inference still works, just untrained)
"""

import os
import logging
import numpy as np
import threading

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

_model     = None
_model_lock = threading.Lock()
_ready     = False


def _build_model():
    """Construct the CNN architecture."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),

        # Block 1
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        # Block 2
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        # Block 3
        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation="softmax"),
    ], name="EmotiScan_CNN")
    return model


def _try_load_weights(model):
    """Attempt to load pretrained weights from known locations."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "models", "emotion_weights.h5"),
        os.path.expanduser("~/.deepface/weights/facial_expression_model_weights.h5"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            try:
                model.load_weights(path, by_name=False, skip_mismatch=True)
                print(f"[EmotiScan] Loaded weights from {path}")
                return True
            except Exception as e:
                print(f"[EmotiScan] Could not load {path}: {e}")
    print("[EmotiScan] No pretrained weights found — using random init (predictions will be random)")
    return False


def _init_model_thread():
    global _model, _ready
    try:
        m = _build_model()
        _try_load_weights(m)
        # Warm up
        dummy = np.zeros((1, 48, 48, 1), dtype="float32")
        m.predict(dummy, verbose=0)
        with _model_lock:
            _model  = m
            _ready  = True
        print("[EmotiScan] Emotion CNN ready ✓")
    except Exception as e:
        print(f"[EmotiScan] Model init error: {e}")


def start_loading():
    """Call once at startup — loads model in background thread."""
    t = threading.Thread(target=_init_model_thread, daemon=True)
    t.start()


def is_ready() -> bool:
    return _ready


def predict(face_bgr: np.ndarray) -> dict:
    """
    Run emotion inference on a face crop.
    face_bgr: H×W×3 uint8 BGR image (any size — resized internally)
    Returns: dict[emotion_name -> float 0-100]
    """
    with _model_lock:
        if _model is None:
            return {}
        model = _model

    import cv2
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    x    = gray.astype("float32") / 255.0
    x    = x.reshape(1, 48, 48, 1)

    preds = model.predict(x, verbose=0)[0]          # shape (7,)
    return {EMOTIONS[i]: float(preds[i]) * 100.0 for i in range(7)}
