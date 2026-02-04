from pathlib import Path
import csv
from collections import Counter

from .base import BaseDataset
from src.data_download import ensure_fer2013_available


class FER2013Dataset(BaseDataset):
    """
    FER2013 dataset using pre-built images and CSV metadata
    (emotion label + train/val/test split).
    """

    def __init__(self, config):
        super().__init__(config)

        root = Path(config.DATASET_ROOT)

        # CSV provides ONLY labels + usage split
        self.csv_path = root / "fer2013_competition" / "icml_face_data.csv"

        # Images are pre-built by data_download
        self.image_dir = root / "fer2013_competition" / "images"

        self._class_weights = None

    def load(self):
        ensure_fer2013_available(self.config)

        train_labels = []

        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]

            for idx, row in enumerate(reader):
                usage = row["Usage"].strip().lower()
                label = int(row["emotion"])

                img_path = self.image_dir / f"fer{idx:07d}.png"
                if not img_path.exists():
                    continue

                item = {
                    "path": str(img_path),
                    "label": label,
                    "weight": None,
                }

                if usage == "training":
                    self._train_items.append(item)
                    train_labels.append(label)
                elif usage == "publictest":
                    self._val_items.append(item)
                elif usage == "privatetest":
                    self._test_items.append(item)

        # ---- class weights (training split only) ----
        self._build_class_weights(train_labels)

        print(
            f"[FER2013Dataset] "
            f"train={len(self._train_items)} "
            f"val={len(self._val_items)} "
            f"test={len(self._test_items)}"
        )

    def _build_class_weights(self, labels):
        """
        Inverse-frequency class weights for training split.
        """
        counter = Counter(labels)
        num_classes = self.config.dataset.num_classes
        total = sum(counter.values())

        self._class_weights = {
            cls: total / (num_classes * counter.get(cls, 1))
            for cls in range(num_classes)
        }

    def get_class_weights(self):
        return self._class_weights
