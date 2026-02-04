from keras.models import Sequential
from keras.layers import (
    Input,
    Conv2D, Activation,
    MaxPooling2D,
    Flatten, GlobalAveragePooling2D,
    Dense,
    Dropout, SpatialDropout2D,  # Für Conv-Layer niemals normales Dropout. 
    BatchNormalization,
)


def build_model(config):
    img_height, img_width = config.dataset.image_size
    num_classes = config.dataset.num_classes

    model = Sequential([
        Input(shape=(img_height, img_width, 1)),

        Conv2D(32, (3, 3), padding="same"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        #SpatialDropout2D(0.1),

        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        #SpatialDropout2D(0.1),

        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.1),

        #GlobalAveragePooling2D(),#  Flatten(),
        #Dense(256, activation="relu"),
        #Dropout(0.5),

        Conv2D(256, 3, padding="same", use_bias=False),
        BatchNormalization(),
        Activation('relu'),

        GlobalAveragePooling2D(),
        Dense(num_classes, activation="softmax"),
    ])

    return model
