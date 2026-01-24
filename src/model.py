import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dropout,
    Dense, BatchNormalization, Input
)

from src.config import NUM_CLASSES, LEARNING_RATE


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
    
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),


        Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
