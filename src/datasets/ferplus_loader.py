import csv
from pathlib import Path

import tensorflow as tf

from src.datasets.base import BaseDatasetLoader
from src.data import ensure_ferplus_available, FER2013_IMG_DIR
import src.config as config


class FERPlusLoader(BaseDatasetLoader):
    """
    FER+ loader – positional CSV, no headers, no assumptions.
    """

    def load(self):
        ensure_ferplus_available()

        csv_path = Path(config.FERPLUS_LABEL_CSV)
        image_dir = FER2013_IMG_DIR

        train_items, val_items, test_items = [], [], []

        stats = {
            "rows_total": 0,
            "no_image_name": 0,
            "image_missing": 0,
            "contempt": 0,
            "nf": 0,
            "zero_votes": 0,
            "low_agreement": 0,
            "kept": 0,
        }

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row_index, row in enumerate(reader):

                stats["rows_total"] += 1

                if len(row) < 11:
                    continue

                usage = row[0].strip().lower()
                img_path = image_dir / f"fer{row_index:07d}.png"
                if not img_path.exists():
                    stats["image_missing"] += 1
                    continue


                if int(row[9]) > 0:
                    stats["contempt"] += 1
                    continue

                if int(row[10]) > 0:
                    stats["nf"] += 1
                    continue

                votes = {
                    "neutral": int(row[2]),
                    "happy": int(row[3]),
                    "surprised": int(row[4]),
                    "sad": int(row[5]),
                    "angry": int(row[6]),
                    "disgusted": int(row[7]),
                    "fearful": int(row[8]),
                }

                total = sum(votes.values())
                if total == 0:
                    stats["zero_votes"] += 1
                    continue

                best = max(votes, key=votes.get)
                if votes[best] / total < config.FERPLUS_MIN_AGREEMENT:
                    stats["low_agreement"] += 1
                    continue

                stats["kept"] += 1

                label_idx = config.CLASS_LABELS.index(best)
                item = (str(img_path), label_idx)

                if usage == "training":
                    train_items.append(item)
                elif usage == "publictest":
                    val_items.append(item)
                elif usage == "privatetest":
                    test_items.append(item)

            print("FER+ DEBUG STATS")
            for k, v in stats.items():
                print(f"{k:>15}: {v}")

        if not train_items:
            raise RuntimeError("FER+ loader: zero training samples")

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
