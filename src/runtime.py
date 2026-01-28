from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf


def configure_tensorflow_runtime():
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print("Could not set GPU memory growth:", e)
    else:
        print("No GPU detected by TensorFlow")
