import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow() # so tensorflow is finally silent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import json

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print("Could not set GPU memory growth:", e)
else:
    print("No GPU detected by TensorFlow")

from src.config import (
    IMG_HEIGHT, IMG_WIDTH,
    CLASS_LABELS,
    NUM_CLASSES,
    MODEL_PATH,
    CASCADE_PATH,
    FORCE_RETRAIN,
    FREEZE_BEST_MODEL,
    PROMOTION_METRIC
)

from src.data import load_datasets
from src.model import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model


def main():
    train_data, val_data, test_data = load_datasets()

    model_file = MODEL_PATH
    best_meta_file = MODEL_PATH.with_suffix(".meta.json")

    # =========================
    # DEBUG EXPORTS (optional)
    # =========================
    def save_debug_image_from_dataset(dataset, name: str):
        # 1) Nimm ein zufälliges Batch
        batches = list(dataset.as_numpy_iterator())
        batch_images, batch_labels = random.choice(batches)

        # 2) Nimm ein zufälliges Bild aus diesem Batch
        idx = random.randint(0, batch_images.shape[0] - 1)
        img = batch_images[idx]
        label = batch_labels[idx]

        # 3) Shape fix: (H, W, 1) → (H, W)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[:, :, 0]

        # 4) Wertebereich korrekt für PNG
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)

        out_path = os.path.join(os.getcwd(), f"debug_{name}.png")
        cv2.imwrite(out_path, img)

        print(f"✔ Saved debug image: {out_path}")
        print(f"  shape: {img.shape}, dtype: {img.dtype}, label: {label}")

    save_debug_image_from_dataset(train_data, "train_sample")
    save_debug_image_from_dataset(val_data, "val_sample")

    # =========================
    # CLASS WEIGHTS
    # =========================
    class_counts = np.zeros(NUM_CLASSES)
    for _, labels in train_data:
        class_counts += np.sum(labels, axis=0)

    total_samples = np.sum(class_counts)
    class_weights = total_samples / (NUM_CLASSES * class_counts)
    class_weights = dict(enumerate(class_weights))

    # =========================
    # LOAD BEST META (if any)
    # =========================
    def load_best_score():
        if not best_meta_file.exists():
            return None

        with open(best_meta_file, "r") as f:
            meta = json.load(f)

        return meta.get(PROMOTION_METRIC, None)

    def save_best_meta(metrics: dict):
        with open(best_meta_file, "w") as f:
            json.dump(metrics, f, indent=2)

    # =========================
    # TRAIN OR LOAD
    # =========================
    if model_file.exists() and not FORCE_RETRAIN:
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_file)

    else:
        print("Training new model...")
        model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))

        history = train_model(
            model,
            train_data,
            val_data,
            class_weights
        )

        # =========================
        # SAVE TRAINING CURVES
        # =========================
        train_loss = history.history["loss"]
        val_loss   = history.history["val_loss"]
        train_acc  = history.history["accuracy"]
        val_acc    = history.history["val_accuracy"]

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label="Train Loss", linestyle="--")
        plt.plot(epochs, val_loss, label="Val Loss", linestyle="--")
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Val Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Curves: Loss and Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_path = "training_curves.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"✔ Training curves saved to: {output_path}")

        # =========================
        # EVALUATE + PROMOTION
        # =========================
        metrics = evaluate_model(model, test_data)

        if FREEZE_BEST_MODEL:
            best_score = load_best_score()
            new_score = metrics[PROMOTION_METRIC]

            if best_score is None or new_score > best_score:
                print(
                    f"New best model! {PROMOTION_METRIC}: {new_score:.4f} "
                    f"(old: {best_score})"
                )

                model.save(model_file)
                save_best_meta(metrics)

            else:
                print(
                    f"Model rejected. {PROMOTION_METRIC}: {new_score:.4f} "
                    f"(best: {best_score:.4f})"
                )

        else:
            model.save(model_file)

    # =========================
    # FINAL EVALUATION (best model)
    # =========================
    print("\nEvaluating frozen best model...\n")
    model = tf.keras.models.load_model(model_file)
    evaluate_model(model, test_data)


if __name__ == "__main__":
    main()
