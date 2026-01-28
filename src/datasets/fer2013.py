import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from .base import BaseDataset


class FER2013Dataset(BaseDataset):
    """
    FER2013 dataset loader.

    Responsibilities:
    - load images from disk
    - preprocess (resize, normalize)
    - encode labels
    - create train/validation splits
    - compute class weights
    """

    def __init__(self, config):
        super().__init__(config)

        self.image_size = config.dataset.image_size
        self.batch_size = config.dataset.batch_size
        self.num_classes = config.dataset.num_classes
        self.seed = config.dataset.seed

        self.train_dir = Path(config.dataset.train_dir)
        self.test_dir = Path(config.dataset.test_dir)

        self.class_names = config.CLASS_LABELS

        # runtime state
        self._train_ds = None
        self._val_ds = None
        self._class_weights = None

    # -------------------------
    # Load
    # -------------------------
    def load(self):
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train dir not found: {self.train_dir}")

        print(f"Loading FER2013 from {self.train_dir}")

        raw_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed,
        )

        self._raw_ds = raw_ds

    # -------------------------
    # Prepare
    # -------------------------
    def prepare(self):
        def preprocess(images, labels):
            images = tf.cast(images, tf.float32) / 255.0
            return images, labels

        self._raw_ds = self._raw_ds.map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # -------------------------
    # Split
    # -------------------------
    def split(self):
        train_ratio = self.config.dataset.train_split
        val_ratio = self.config.dataset.val_split

        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_split + val_split must be < 1.0")

        ds_size = tf.data.experimental.cardinality(self._raw_ds).numpy()

        train_size = int(train_ratio * ds_size)
        val_size = int(val_ratio * ds_size)

        self._train_ds = (
            self._raw_ds
            .take(train_size)
            .cache()
            .shuffle(1000, seed=self.seed)
            .prefetch(tf.data.AUTOTUNE)
        )

        self._val_ds = (
            self._raw_ds
            .skip(train_size)
            .take(val_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        self._compute_class_weights()

    # -------------------------
    # Class weights
    # -------------------------
    def _compute_class_weights(self):
        labels = []

        for _, y in self._train_ds:
            labels.extend(tf.argmax(y, axis=1).numpy())

        labels = np.array(labels)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.num_classes),
            y=labels,
        )

        self._class_weights = dict(enumerate(weights))

    # -------------------------
    # Accessors
    # -------------------------
    def get_train(self):
        return self._train_ds, None

    def get_val(self):
        return self._val_ds, None

    def get_class_weights(self):
        return self._class_weights
