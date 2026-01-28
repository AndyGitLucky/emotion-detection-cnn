import json
import tensorflow as tf
import matplotlib.pyplot as plt

from src.datasets.factory import DatasetFactory
from src.model import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model


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
        self.dataset.split()

        self.X_train, self.y_train = self.dataset.get_train()
        self.X_val, self.y_val = self.dataset.get_val()

    # -------------------------
    # Model
    # -------------------------
    def build_model(self):
        print("Building model...")
        self.model = build_model(self.config)

    # -------------------------
    # Training
    # -------------------------
    def train(self):
        print("Training model...")

        self.history = train_model(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            config=self.config,
            class_weights=self.dataset.get_class_weights()
        )

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self):
        print("Evaluating model...")
        self.metrics = evaluate_model(
            model=self.model,
            val_data=self.dataset.get_val()[0],
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

        if best_score is None or new_score > best_score:
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
        if self.history is None:
            return

        print("Saving training curves...")

        train_loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        train_acc = self.history.history["accuracy"]
        val_acc = self.history.history["val_accuracy"]

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label="Train Loss", linestyle="--")
        plt.plot(epochs, val_loss, label="Val Loss", linestyle="--")
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Val Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Curves: Loss and Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_path = "training_curves.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"✔ Training curves saved to: {output_path}")

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
            val_data=self.dataset.get_val()[0],
            config=self.config
)

