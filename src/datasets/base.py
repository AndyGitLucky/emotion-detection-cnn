import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=alles, 1=INFO, 2=WARNING, 3=ERROR

from abc import ABC, abstractmethod
import tensorflow as tf

class BaseDataset(ABC):
    """
    Dataset-Vertrag:
      item = {
        "path": str,
        "label": int | list[float],
        "weight": Optional[float]
      }
    Semantik liegt in den Subklassen.
    """

    def __init__(self, config):
        self.config = config

        # Items
        self._train_items = []
        self._val_items = []
        self._test_items = []

        # tf.data
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    # ------------------------------
    @abstractmethod
    def load(self):
        """IO/Parsing. Füllt _train_items/_val_items/_test_items."""
        ...

    # ------------------------------
    def prepare(self):
        """Baut tf.data Datasets aus Items (ohne Domänenwissen)."""
        self._train_ds = self._build_tf_dataset(self._train_items, shuffle=True)
        self._val_ds   = self._build_tf_dataset(self._val_items)
        self._test_ds  = self._build_tf_dataset(self._test_items)

        self._debug_print()

    # ------------------------------
    def _build_tf_dataset(self, items, shuffle=False):
        image_size = self.config.dataset.image_size
        batch_size = self.config.dataset.batch_size
        seed = self.config.dataset.seed
        num_classes = self.config.dataset.num_classes

        if not items:
            return None

        paths   = [it["path"]  for it in items]
        labels  = [it["label"] for it in items]
        weights = [it.get("weight") for it in items]

        # Labels: hard (int) ODER soft (vector) — Struktur entscheidet
        if isinstance(labels[0], int):
            y = tf.keras.utils.to_categorical(labels, num_classes)
        else:
            y = tf.convert_to_tensor(labels, dtype=tf.float32)

        x = tf.convert_to_tensor(paths, dtype=tf.string)

        if weights[0] is None:
            ds = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            w = tf.convert_to_tensor(weights, dtype=tf.float32)
            ds = tf.data.Dataset.from_tensor_slices((x, y, w))

        def _load_image(*args):
            # args: (path, label) oder (path, label, weight)
            path = args[0]
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.resize(img, image_size)
            img = tf.cast(img, tf.float32) / 255.0
            return (img,) + args[1:]

        ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        # Training-only augmentation:
        # When `shuffle=True` (i.e. training split), apply image augmentation
        # AFTER loading/decoding the image, but BEFORE shuffling and batching.
        #
        # The lambda preserves the original dataset structure:
        # - args[0] is the image tensor
        # - args[1:] are label (and optional sample_weight)
        # This way, augmentation modifies ONLY the image, never labels or weights.
        if shuffle:
            ds = ds.map(
                lambda *args: (self.augment(args[0]),) + args[1:],
                num_parallel_calls=tf.data.AUTOTUNE
            )
            ds = ds.shuffle(1000, seed=seed)

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


    # ------------------------------
    def augment(self, img):
            """
            Training-only image augmentation.
            Override in subclasses if needed.
            Default: light, safe augmentation for FER-style data.
            """
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            return img

    # ------------------------------
    def get_class_weights(self):
        """
        Optional class weights for imbalanced datasets.
        Default: no class weighting.
        """
        return None

    # ------------------------------
    def _debug_print(self):
        print(
            f"[{self.__class__.__name__}] "
            f"train={len(self._train_items)} "
            f"val={len(self._val_items)} "
            f"test={len(self._test_items)}"
        )

    # ------------------------------
    def get_train(self): return self._train_ds
    def get_val(self):   return self._val_ds
    def get_test(self):  return self._test_ds
