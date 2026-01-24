import time
from keras.callbacks import EarlyStopping
from src.config import EPOCHS, EARLY_STOP_PATIENCE


def train_model(model, train_data, val_data, class_weights):
    es = EarlyStopping(
        monitor='val_accuracy',
        mode='auto',
        verbose=1,
        patience=EARLY_STOP_PATIENCE
    )

    tik = time.time()
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=es
    )
    tok = time.time()

    training_time = tok - tik
    print("Training time: {:.2f} seconds".format(training_time))

    return history
