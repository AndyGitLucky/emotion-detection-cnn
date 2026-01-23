import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dropout, 
    Dense, RandomContrast, RandomZoom, 
    RandomBrightness, RandomFlip, Rescaling,
    BatchNormalization, RandomTranslation
)

from src.config import NUM_CLASSES, LEARNING_RATE



# Function to create and compile the model
def build_model(input_shape):
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers
    # batch normalization, dann die aktivierungsfunktion ausprobieren!
    model = Sequential([
        RandomZoom(height_factor=(-0.2, -0.05), width_factor=(-0.2, -0.05), 
                   input_shape=input_shape), 
        RandomTranslation(0.2, 0.2, fill_mode= 'nearest'),
        RandomBrightness(factor=0.2, value_range=[0,255]),
        RandomContrast(0.3),
        RandomFlip(mode="horizontal"),
        Rescaling(scale=1./255),
        Conv2D(16, (3, 3), padding='valid', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(padding='same'),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(padding='same'),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(padding='same'),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                  weighted_metrics=['accuracy'])
    
    return model