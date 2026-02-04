import json
import tensorflow as tf
import matplotlib.pyplot as plt

from src.datasets.factory import DatasetFactory
from src.model import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.evaluator_soft import evaluate_model_soft


class TrainingPipeline:
    """
    Orchestrates the full training lifecycle.

    Responsibilities:
    - dataset orchestration (not implementation)
    - model lifecycle
    - training, evaluation, promotion
    """

    def __init__(self, config):
        self.config = config

        # --- Dataset (factory-based) ---
        self.dataset = DatasetFactory.create(
            config.dataset.name,
            config
        )

        # --- Runtime state ---
        self.model = None
        self.history = None
        self.metrics = None

        self.model_file = self.config.MODEL_PATH
        self.best_meta_file = self.config.MODEL_PATH.with_suffix(".meta.json")

    # -------------------------
    # Data
    # -------------------------
    def prepare_data(self):
        print("Preparing dataset...")
        self.dataset.load()
        self.dataset.prepare()

        self.train_ds = self.dataset.get_train()
        self.val_ds = self.dataset.get_val()
        self.dataset.debug_show_samples("train", n=5)
        self.dataset.debug_show_samples("val", n=5)
        self.dataset.debug_show_samples("test", n=5)


    # -------------------------
    # Model
    # -------------------------
    def build_model(self):
        print("Building model...")
        self.model = build_model(self.config)
        #print(self.model.summary())

    # -------------------------
    # Training
    # -------------------------
    def train(self):
        print("Training model...")

        self.history = train_model(
            model=self.model,
            train_data=self.train_ds,
            val_data=self.val_ds,
            config=self.config,
            class_weights=self.dataset.get_class_weights()
        )


    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self):
        print("Evaluating model...")
        self.metrics = evaluate_model_soft(
            model=self.model,
            val_data=self.dataset.get_val(),
            config=self.config
        )


    # -------------------------
    # Best-model helpers
    # -------------------------
    def load_best_score(self):
        if not self.best_meta_file.exists():
            return None

        with open(self.best_meta_file, "r") as f:
            meta = json.load(f)

        return meta.get(self.config.PROMOTION_METRIC)

    def save_best_meta(self, metrics: dict):
        with open(self.best_meta_file, "w") as f:
            json.dump(metrics, f, indent=2)

    # -------------------------
    # Promotion logic
    # -------------------------
    def promote_if_best(self):
        if not self.config.FREEZE_BEST_MODEL:
            print("Saving model (freeze disabled)...")
            self.model.save(self.model_file)
            return

        best_score = self.load_best_score()
        new_score = self.metrics[self.config.PROMOTION_METRIC]

        if best_score is None or new_score < best_score:
            print(
                f"New best model! {self.config.PROMOTION_METRIC}: {new_score:.4f} "
                f"(old: {best_score})"
            )

            self.model.save(self.model_file)
            self.save_best_meta(self.metrics)

        else:
            print(
                f"Model rejected. {self.config.PROMOTION_METRIC}: {new_score:.4f} "
                f"(best: {best_score:.4f})"
            )

    # -------------------------
    # Plotting
    # -------------------------
    def save_training_curves(self):
        import matplotlib.pyplot as plt

        history = self.history.history

        plt.figure(figsize=(10, 4))

        # -----------------
        # Loss
        # -----------------
        plt.subplot(1, 2, 1)
        plt.plot(history["loss"], label="train_loss")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="val_loss")
        plt.title("Loss")
        plt.legend()

        # -----------------
        # Accuracy (optional)
        # -----------------
        if "accuracy" in history:
            plt.subplot(1, 2, 2)
            plt.plot(history["accuracy"], label="train_acc")
            if "val_accuracy" in history:
                plt.plot(history["val_accuracy"], label="val_acc")
            plt.title("Accuracy")
            plt.legend()

        plt.tight_layout()
        plt.savefig("training_curves.png")
        plt.close()

        print("✔ Training curves saved to: training_curves.png")

    # -------------------------
    # Orchestration
    # -------------------------
    def run(self):
        self.prepare_data()

        if self.model_file.exists() and not self.config.FORCE_RETRAIN:
            print("Loading existing frozen best model...")
            self.model = tf.keras.models.load_model(self.model_file)

        else:
            self.build_model()
            self.train()
            self.save_training_curves()
            self.evaluate()
            self.promote_if_best()

        print("\nEvaluating frozen best model...\n")
        self.model = tf.keras.models.load_model(self.model_file)
        evaluate_model(
            model=self.model,
            val_data=self.val_ds,
            config=self.config
)

