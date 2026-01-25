import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TrainingPipeline:
    def __init__(
        self,
        dataset_loader,
        model_builder,
        trainer_fn,
        evaluator_fn,
        config
    ):
        # --- Injected dependencies ---
        self.dataset_loader = dataset_loader
        self.model_builder = model_builder
        self.trainer_fn = trainer_fn
        self.evaluator_fn = evaluator_fn
        self.config = config

        # --- Runtime state ---
        self.model = None
        self.history = None
        self.metrics = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.model_file = self.config.MODEL_PATH
        self.best_meta_file = self.config.MODEL_PATH.with_suffix(".meta.json")

    # -------------------------
    # Data
    # -------------------------
    def load_data(self):
        print("Loading datasets...")
        self.train_data, self.val_data, self.test_data = self.dataset_loader()

    def compute_class_weights(self):
        print("Computing class weights...")
        class_counts = np.zeros(self.config.NUM_CLASSES)

        for _, labels in self.train_data:
            class_counts += np.sum(labels, axis=0)

        total_samples = np.sum(class_counts)
        class_weights = total_samples / (self.config.NUM_CLASSES * class_counts)

        return dict(enumerate(class_weights))

    # -------------------------
    # Model
    # -------------------------
    def build_model(self):
        print("Building model...")
        self.model = self.model_builder(
            input_shape=(
                self.config.IMG_HEIGHT,
                self.config.IMG_WIDTH,
                1
            )
        )

    # -------------------------
    # Training
    # -------------------------
    def train(self):
        print("Training model...")
        class_weights = self.compute_class_weights()

        self.history = self.trainer_fn(
            self.model,
            self.train_data,
            self.val_data,
            class_weights
        )

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self):
        print("Evaluating model...")
        self.metrics = self.evaluator_fn(
            self.model,
            self.test_data
        )

    # -------------------------
    # Best-model freeze helpers
    # -------------------------
    def load_best_score(self):
        if not self.best_meta_file.exists():
            return None

        with open(self.best_meta_file, "r") as f:
            meta = json.load(f)

        return meta.get(self.config.PROMOTION_METRIC, None)

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
        self.load_data()

        if self.model_file.exists() and not self.config.FORCE_RETRAIN:
            print("Loading existing frozen best model...")
            self.model = tf.keras.models.load_model(self.model_file)

        else:
            self.build_model()
            self.train()
            self.save_training_curves()
            self.evaluate()
            self.promote_if_best()

        # Always evaluate frozen best model at the end
        print("\nEvaluating frozen best model...\n")
        self.model = tf.keras.models.load_model(self.model_file)
        self.evaluator_fn(self.model, self.test_data)
