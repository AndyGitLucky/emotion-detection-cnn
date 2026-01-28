import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    class_weights=None,
):
    """
    Compile and train the model.

    Responsibilities:
    - optimizer + learning rate
    - loss + metrics
    - callbacks
    - fit loop
    """

    # -------------------------
    # Compile (training concern)
    # -------------------------
    model.compile(
        optimizer=Adam(learning_rate=config.training.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # -------------------------
    # Callbacks
    # -------------------------
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=config.training.early_stop_patience,
        restore_best_weights=True,
        verbose=1,
    )

    # -------------------------
    # Training
    # -------------------------
    start_time = time.time()

    history = model.fit(
        X_train,
        validation_data=(X_val, y_val),
        epochs=config.training.epochs,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f} seconds")

    return history
