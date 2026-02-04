import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=alles, 1=INFO, 2=WARNING, 3=ERROR

from src.training_pipeline import TrainingPipeline
import src.config as config

def configure_runtime():
    from src.runtime import configure_tensorflow_runtime
    configure_tensorflow_runtime()


def main():
    configure_runtime()
    pipeline = TrainingPipeline(config=config)
    pipeline.run()


if __name__ == "__main__":
    main()
