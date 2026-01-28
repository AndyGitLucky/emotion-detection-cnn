import csv
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.datasets.base import BaseDataset
from src.data_download import ensure_fer2013_available


class FER2013Dataset(BaseDataset):
    """
    FER2013 dataset based on icml_face_data.csv.
    Uses shared FER2013 images (same base as FER+).
    """

    def __init__(self, config):
        super().__init__(config)

        # internal state (IDENTICAL to FER+)
        self._train_items = []
        self._val_items = []

        self._train_ds = None
        self._val_ds = None
        self._class_weights = None

    # -------------------------------------------------
    # Load: CSV parsing
    # -------------------------------------------------
    def load(self):
        ensure_fer2013_available(self.config)

        root = self.config.dataset.root
        csv_path = root / "fer2013_competition" / "icml_face_data.csv"
        image_dir = root / "fer2013_competition" / "images"

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        class_labels = self.config.CLASS_LABELS

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            # normalize headers
            field_map = {k.strip().lower(): k for k in reader.fieldnames}

            usage_key = field_map.get("usage")
            emotion_key = field_map.get("emotion")

            if usage_key is None or emotion_key is None:
                raise RuntimeError(
                    f"Unexpected FER2013 CSV header: {reader.fieldnames}"
                )

            for idx, row in enumerate(reader):
                usage = row[usage_key].strip().lower()
                emotion = int(row[emotion_key])

                if emotion < 0 or emotion >= len(class_labels):
                    continue

                img_path = image_dir / f"fer{idx:07d}.png"
                if not img_path.exists():
                    continue

                item = (str(img_path), emotion)

                if usage == "training":
                    self._train_items.append(item)
                elif usage == "publictest":
                    self._val_items.append(item)

        if not self._train_items:
            raise RuntimeError("FER2013 dataset: zero training samples")

    # -------------------------------------------------
    # Prepare: tf.data pipeline
    # -------------------------------------------------
    def prepare(self):
        image_size = self.config.dataset.image_size
        batch_size = self.config.dataset.batch_size
        num_classes = self.config.dataset.num_classes
        seed = self.config.dataset.seed

        def build_dataset(items, shuffle=False):
            paths, labels = zip(*items)

            labels = tf.keras.utils.to_categorical(
                labels, num_classes=num_classes
            )

            ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))

            def _load_image(path, label):
                img = tf.io.read_file(path)
                img = tf.image.decode_png(img, channels=1)
                img = tf.image.resize(img, image_size)
                img = tf.cast(img, tf.float32) / 255.0
                return img, label

            ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

            if shuffle:
                ds = ds.shuffle(1000, seed=seed)

            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        self._train_ds = build_dataset(self._train_items, shuffle=True)
        self._val_ds = build_dataset(self._val_items)

        self._compute_class_weights()

    # -------------------------------------------------
    def split(self):
        pass

    # -------------------------------------------------
    def _compute_class_weights(self):
        labels = [label for _, label in self._train_items]

        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.config.dataset.num_classes),
            y=np.array(labels),
        )

        self._class_weights = dict(enumerate(weights))

    # -------------------------------------------------
    def get_train(self):
        return self._train_ds, None

    def get_val(self):
        return self._val_ds, None

    def get_class_weights(self):
        return self._class_weights
