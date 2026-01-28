import csv
from pathlib import Path
import tensorflow as tf

from src.datasets.base import BaseDataset
from src.data_download import ensure_ferplus_available


class FERPlusDataset(BaseDataset):
    """
    FER+ Dataset (hard labels, majority vote).
    Uses the same FER2013 image base.
    """

    def __init__(self, config):
        super().__init__(config)

        # internal state (IDENTICAL to FER2013)
        self._train_items = []
        self._val_items = []
        self._test_items = []

        self._train_ds = None
        self._val_ds = None

    # -------------------------------------------------
    # Load: CSV parsing + filtering
    # -------------------------------------------------
    def load(self):
        ensure_ferplus_available(self.config)

        root = self.config.dataset.root
        csv_path = Path(self.config.dataset.ferplus.label_csv)
        image_dir = root / "fer2013_competition" / "images"

        class_labels = self.config.CLASS_LABELS
        min_agreement = self.config.dataset.ferplus.min_agreement
        drop_contempt = self.config.dataset.ferplus.drop_contempt

        stats = {
            "rows_total": 0,
            "image_missing": 0,
            "contempt": 0,
            "nf": 0,
            "zero_votes": 0,
            "low_agreement": 0,
            "kept": 0,
        }

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            field_map = {k.strip().lower(): k for k in reader.fieldnames}

            usage_key = field_map.get("usage")

            vote_keys = {
                "neutral": field_map.get("neutral"),
                "happy": field_map.get("happy"),
                "surprised": field_map.get("surprise"),
                "sad": field_map.get("sad"),
                "angry": field_map.get("anger"),
                "disgusted": field_map.get("disgust"),
                "fearful": field_map.get("fear"),
            }

            contempt_key = field_map.get("contempt")
            nf_key = field_map.get("unknown")

            if usage_key is None:
                raise RuntimeError(
                    f"Unexpected FER+ CSV header: {reader.fieldnames}"
                )

            for idx, row in enumerate(reader):
                stats["rows_total"] += 1

                usage = row[usage_key].strip().lower()
                img_path = image_dir / f"fer{idx:07d}.png"

                if not img_path.exists():
                    stats["image_missing"] += 1
                    continue

                if drop_contempt and contempt_key and int(row[contempt_key]) > 0:
                    stats["contempt"] += 1
                    continue

                if nf_key and int(row[nf_key]) > 0:
                    stats["nf"] += 1
                    continue

                votes = {
                    name: int(row[key])
                    for name, key in vote_keys.items()
                    if key is not None
                }

                total = sum(votes.values())
                if total == 0:
                    stats["zero_votes"] += 1
                    continue

                best = max(votes, key=votes.get)
                if votes[best] / total < min_agreement:
                    stats["low_agreement"] += 1
                    continue

                label_idx = class_labels.index(best)
                item = (str(img_path), label_idx)

                if usage == "training":
                    self._train_items.append(item)
                elif usage == "publictest":
                    self._val_items.append(item)
                elif usage == "privatetest":
                    self._test_items.append(item)

                stats["kept"] += 1

        print("\nFER+ DEBUG STATS")
        for k, v in stats.items():
            print(f"{k:>15}: {v}")

        if not self._train_items:
            raise RuntimeError("FER+ dataset: zero training samples")

    # -------------------------------------------------
    def prepare(self):
        image_size = self.config.dataset.image_size
        batch_size = self.config.dataset.batch_size
        seed = self.config.dataset.seed
        num_classes = self.config.dataset.num_classes

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

    # -------------------------------------------------
    def split(self):
        pass

    # -------------------------------------------------
    def get_train(self):
        return self._train_ds, None

    def get_val(self):
        return self._val_ds, None
