import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow() # so tensorflow is finally silent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from src.config import (
    IMG_HEIGHT, IMG_WIDTH,
    CLASS_LABELS,
    NUM_CLASSES,
    MODEL_PATH
)

from src.data import load_datasets
from src.model import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model


def main():
    train_data, val_data, test_data = load_datasets()

    model_file = MODEL_PATH

    # Calculate class weights
    class_counts = np.zeros(NUM_CLASSES)
    for _, labels in train_data:
        class_counts += np.sum(labels, axis=0)
        total_samples = np.sum(class_counts)
        class_weights = total_samples / (NUM_CLASSES * class_counts)
        class_weights = dict(enumerate(class_weights))

    # Plot class weights
    plt.bar(range(NUM_CLASSES), class_weights.values(), tick_label=CLASS_LABELS)
    plt.xlabel('Emotion')
    plt.ylabel('Class Weight')
    plt.title('Class Weights')
    plt.show()

    if not os.path.exists(model_file):
        print("No saved model found. Training a new model...")

        model = build_model(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)
        )

        # class_weights must already exist in your code
        history = train_model(
            model,
            train_data,
            val_data,
            class_weights
        )

        model.save(model_file)

    else:
        print("Loading saved model...")
        model = tf.keras.models.load_model(model_file)

    evaluate_model(model, test_data)


if __name__ == "__main__":
    main()
