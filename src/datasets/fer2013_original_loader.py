import csv
from pathlib import Path

import tensorflow as tf

from src.datasets.base import BaseDatasetLoader
from src.data import ensure_fer2013_available, FER2013_IMG_DIR
import src.config as config


class FER2013OriginalLoader(BaseDatasetLoader):
    """
    Loader for ORIGINAL FER2013 labels from icml_face_data.csv.

    Uses the SAME image base as FERPlusLoader:
    - archive/fer2013_competition/images/ferXXXXXXX.png

    The only difference to FERPlusLoader:
    - labels come from the original FER2013 emotion column
    """

    def load(self):
        # --------------------------------------------------
        # 1) Ensure original FER2013 images exist
        # --------------------------------------------------
        ensure_fer2013_available()

        csv_path = Path("archive/fer2013_competition/icml_face_data.csv")
        image_dir = FER2013_IMG_DIR

        train_items, val_items, test_items = [], [], []

        # --------------------------------------------------
        # 2) Parse CSV (index-based, header-aware)
        # --------------------------------------------------
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)

            for row_index, row in enumerate(reader):
                # image path (index-based, identical to FER+)
                img_path = image_dir / f"fer{row_index:07d}.png"
                if not img_path.exists():
                    continue

                # original FER2013 label (0..6)
                try:
                    label_idx = int(row["emotion"])
                except (KeyError, ValueError):
                    continue

                if not (0 <= label_idx < config.NUM_CLASSES):
                    continue

                usage = row["Usage"].strip().lower()

                item = (str(img_path), label_idx)

                if usage == "training":
                    train_items.append(item)
                elif usage == "publictest":
                    val_items.append(item)
                elif usage == "privatetest":
                    test_items.append(item)

        if not train_items:
            raise RuntimeError("FER2013OriginalLoader: zero training samples")

        # --------------------------------------------------
        # 3) tf.data builder (IDENTICAL to FERPlusLoader)
        # --------------------------------------------------
        def build_dataset(items, shuffle=False):
            paths, labels = zip(*items)

            labels = tf.keras.utils.to_categorical(
                labels, num_classes=config.NUM_CLASSES
            )

            ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))

            def _load_image(path, label):
                img = tf.io.read_file(path)
                img = tf.image.decode_png(img, channels=1)
                img = tf.image.resize(
                    img, (config.IMG_HEIGHT, config.IMG_WIDTH)
                )
                img = tf.cast(img, tf.float32) / 255.0
                return img, label

            ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

            if shuffle:
                ds = ds.shuffle(1000, seed=42)

            return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return (
            build_dataset(train_items, shuffle=True),
            build_dataset(val_items),
            build_dataset(test_items),
        )
