import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import KLDivergence, CategoricalCrossentropy


def train_model(
    *,
    model,
    train_data,
    val_data,
    config,
    class_weights=None,
):
    """
    Compile and train the model using tf.data datasets.

    Responsibilities:
    - optimizer + learning rate
    - loss + metrics
    - callbacks
    - fit loop
    """

    # -------------------------
    # Compile (training concern)
    # -------------------------
    if config.DATASET == "ferplus":
        loss = CategoricalCrossentropy()
        metrics = []                  # accuracy optional
        monitor = "val_loss"
        mode = "min"
        class_weights = None
    else:
        loss = CategoricalCrossentropy()
        metrics = ["accuracy"]
        monitor = "val_accuracy"
        mode = "max"


    model.compile(
        optimizer=Adam(learning_rate=config.training.learning_rate),
        loss=loss,
        metrics=metrics,
        weighted_metrics=[],         # silence sample_weight warnings
    )

    # -------------------------
    # Callbacks
    # -------------------------
    callbacks = []
    '''
    if val_data is not None and config.training.early_stop_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=config.training.early_stop_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )'''
    x, y, w = next(iter(train_data))
    # print("y argmax:", np.unique(np.argmax(y.numpy(), axis=1), return_counts=True))

    # -------------------------
    # Training
    # -------------------------
    start_time = time.time()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.training.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f} seconds")

    return history
