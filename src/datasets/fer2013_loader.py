import tensorflow as tf

from src.datasets.base import BaseDatasetLoader
from src.data import ensure_fer2013_available
from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
)


class FER2013Loader(BaseDatasetLoader):
    """
    FER2013 dataset loader.
    - ensures dataset availability via data.py
    - builds tf.data pipelines
    """

    def load(self):
        # ----------------------------------
        # 1) Ensure data is available
        # ----------------------------------
        ensure_fer2013_available()

        # ----------------------------------
        # 2) Build datasets
        # ----------------------------------
        train_data = tf.keras.utils.image_dataset_from_directory(
            directory=str(TRAIN_DIR),
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            seed=42,
        )

        val_data = tf.keras.utils.image_dataset_from_directory(
            directory=str(TEST_DIR),
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            validation_split=0.2,
            subset="training",
            seed=42,
        )

        test_data = tf.keras.utils.image_dataset_from_directory(
            directory=str(TEST_DIR),
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            validation_split=0.2,
            subset="validation",
            seed=42,
        )

        # ----------------------------------
        # 3) Performance optimizations
        # ----------------------------------
        autotune = tf.data.AUTOTUNE

        train_data = train_data.cache().prefetch(autotune)
        val_data   = val_data.cache().prefetch(autotune)
        test_data  = test_data.cache().prefetch(autotune)

        return train_data, val_data, test_data
