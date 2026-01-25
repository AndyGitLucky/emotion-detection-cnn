from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from src.runtime import configure_tensorflow_runtime
configure_tensorflow_runtime()

from src.data import load_datasets
from src.model import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model
import src.config as config

from src.training_pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline(
        dataset_loader=load_datasets,
        model_builder=build_model,
        trainer_fn=train_model,
        evaluator_fn=evaluate_model,
        config=config
    )

    pipeline.run()


if __name__ == "__main__":
    main()
