from pathlib import Path
import csv
import numpy as np

from .base import BaseDataset
from src.data_download import ensure_ferplus_available

FERPLUS_COLUMN_MAP = {
    "angry": "anger",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "surprised": "surprise",
}


class FERPlusDataset(BaseDataset):
    """
    FER+ dataset with soft labels and agreement-based sample weights.
    """

    def __init__(self, config):
        super().__init__(config)

        root = Path(config.DATASET_ROOT)

        self.image_dir = root / "fer2013_competition" / "images"
        self.csv_path  = root / "ferplus" / "fer2013new.csv"

        # Expected order of FER+ label columns
        self.class_labels = config.CLASS_LABELS

    def load(self):
        ensure_ferplus_available(self.config)

        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError("CSV has no header")
            
            reader.fieldnames = [h.strip() for h in reader.fieldnames]

            for row in reader:
                usage = row["Usage"].strip().lower()

                img = self.image_dir / row["Image name"].strip()
                if not img.exists():
                    continue

                # ---- soft label vector in MODEL order----
                votes = np.array(
                    [float(row[FERPLUS_COLUMN_MAP[c]]) for c in self.class_labels],
                    dtype=np.float32,
                )

                total = votes.sum()
                if total <= 0:
                    continue

                soft = votes / total

                # ---- sample_weight ----
                neutral_idx = self.class_labels.index("neutral")

                agreement = soft.max()

                # 1. Neutral-Dämpfung
                neutral_penalty = 1.0 - 0.7 * soft[neutral_idx]

                # 2. Dominanz-Dämpfung (Label-Seite)
                label_dominance_penalty = 1.0 - 0.5 * soft.max()

                # 3. Entropie-Förderung (Label-Seite)
                entropy = -np.sum(soft * np.log(soft + 1e-8))
                max_entropy = np.log(len(soft))
                entropy_bonus = entropy / max_entropy   # ∈ [0,1]

                sample_weight = agreement \
                            * neutral_penalty \
                            * label_dominance_penalty \
                            * (0.5 + 0.5 * entropy_bonus)


                item = {
                    "path": str(img),
                    "label": soft,
                    "weight": float(sample_weight),
                }



                if usage == "training":
                    self._train_items.append(item)
                elif usage == "publictest":
                    self._val_items.append(item)
                elif usage == "privatetest":
                    self._test_items.append(item)
        

        item = np.random.choice(self._train_items)
        
        print(
            f"label={item['label']} | "
            f"sum={item['label'].sum():.4f} | "
            f"weight={item['weight']:.4f}"
        )

    def get_class_weights(self):
        # FER+ uses sample_weight, NOT class_weight
        return None
    
    def debug_show_samples(self, split="train", n=5):
        import matplotlib.pyplot as plt

        if split == "train":
            ds = self.get_train()
        elif split == "val":
            ds = self.get_val()
        elif split == "test":
            ds = self.get_test()
        else:
            raise ValueError(split)

        print(f"\n=== DEBUG {split.upper()} SAMPLES ===")

        shown = 0
        for batch in ds:
            images = batch[0]
            labels = batch[1]

            for i in range(images.shape[0]):
                img = images[i].numpy().squeeze()
                label = labels[i].numpy()

                if label.ndim == 1:
                    cls = int(np.argmax(label))
                    dist = label
                else:
                    cls = int(label)
                    dist = None

                print(f"Label argmax: {cls} ({self.class_labels[cls]})")
                if dist is not None:
                    print("Distribution:", np.round(dist, 3))

                plt.imshow(img, cmap="gray")
                plt.title(self.class_labels[cls])
                plt.axis("off")
                plt.show()

                shown += 1
                if shown >= n:
                    return

