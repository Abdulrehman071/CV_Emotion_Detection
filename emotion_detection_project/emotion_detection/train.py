"""
train.py  —  Train the EmotiScan emotion CNN on FER2013
──────────────────────────────────────────────────────
Dataset : FER2013  (Kaggle or from fer2013.csv)
          CSV format: emotion (0-6), pixels (48×48 space-separated), Usage
Output  : models/emotion_weights.h5

Usage:
    python3 train.py --data fer2013.csv --epochs 60
    python3 train.py --data fer2013.csv --epochs 60 --augment
"""

import os
import argparse
import numpy as np
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_fer2013(csv_path: str):
    """Parse FER2013 CSV → (X_train, y_train, X_val, y_val, X_test, y_test)."""
    import pandas as pd

    print(f"[train] Loading {csv_path} …")
    df = pd.read_csv(csv_path)

    def parse_pixels(s):
        return np.array(s.split(), dtype="float32").reshape(48, 48, 1) / 255.0

    X, y, usage = [], [], []
    for _, row in df.iterrows():
        X.append(parse_pixels(row["pixels"]))
        y.append(int(row["emotion"]))
        usage.append(row.get("Usage", "Training"))

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")
    usage = np.array(usage)

    train_mask = usage == "Training"
    val_mask   = usage == "PublicTest"
    test_mask  = usage == "PrivateTest"

    print(f"[train] Train={train_mask.sum()}  Val={val_mask.sum()}  Test={test_mask.sum()}")
    return (
        X[train_mask], y[train_mask],
        X[val_mask],   y[val_mask],
        X[test_mask],  y[test_mask],
    )


def build_model():
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),

        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation="softmax"),
    ], name="EmotiScan_CNN")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train EmotiScan Emotion CNN")
    parser.add_argument("--data",    required=True, help="Path to fer2013.csv")
    parser.add_argument("--epochs",  type=int, default=60)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--out",     default="models/emotion_weights.h5")
    args = parser.parse_args()

    import tensorflow as tf

    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013(args.data)

    # One-hot encode
    y_train_oh = tf.keras.utils.to_categorical(y_train, 7)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   7)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  7)

    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.out, monitor="val_accuracy", save_best_only=True,
            save_weights_only=True, verbose=1
        ),
    ]

    if args.augment:
        print("[train] Data augmentation enabled")
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(X_train)
        train_gen = datagen.flow(X_train, y_train_oh, batch_size=args.batch)
        steps = len(X_train) // args.batch
        history = model.fit(
            train_gen,
            steps_per_epoch=steps,
            epochs=args.epochs,
            validation_data=(X_val, y_val_oh),
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            X_train, y_train_oh,
            batch_size=args.batch,
            epochs=args.epochs,
            validation_data=(X_val, y_val_oh),
            callbacks=callbacks,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_weights(args.out)
    print(f"\n[train] Weights saved → {args.out}")

    loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"[train] Test accuracy: {acc*100:.2f}%")

    # Confusion matrix
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    from sklearn.metrics import classification_report
    try:
        print("\n" + classification_report(y_test, y_pred, target_names=EMOTIONS))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
