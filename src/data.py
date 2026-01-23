import tensorflow as tf
from src.config import (
    TRAIN_DIR, TEST_DIR,
    IMG_HEIGHT, IMG_WIDTH,
    BATCH_SIZE
)


def load_datasets():
    train_data = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels='inferred',
        label_mode='categorical', 
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        seed=42
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        subset='training',
        seed=42
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        subset='validation',
        seed=42
    )

    return train_data, val_data, test_data
