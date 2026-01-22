"""
Created: 29.02.2024

@author: Andreas Eichmann
"""

import os
#import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from silence_tensorflow import silence_tensorflow
silence_tensorflow() # so tensorflow is finally silent
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dropout, 
    Dense, RandomContrast, RandomZoom, 
    RandomBrightness, RandomFlip, Rescaling,
    BatchNormalization, RandomTranslation)
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import random
import time
import cv2 

# Define constants
TRAIN_DIR = "H:/alfa-training/Deep Learning/_Abschlussprojekt/archive/train"
TEST_DIR = "H:/alfa-training/Deep Learning/_Abschlussprojekt/archive/test"
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
NUM_CLASSES = 7
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 20
EPOCHS = 50

# Define class labels
CLASS_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Function to plot a random picture from each class
def show_random_images(directory, classes) -> None:
    plt.figure(figsize=(10, 10))
    for i, class_name in enumerate(classes):
        plt.subplot(1, len(classes), i+1)
        images_dir = os.path.join(directory, class_name)
        image_files = os.listdir(images_dir)
        rand_im = np.random.randint(0, len(image_files))
        fname = image_files[rand_im]
        image = Image.open(os.path.join(images_dir, fname)).convert("L")
        arr = np.asarray(image)
        plt.title(class_name)
        plt.axis('off')
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.show()

# Function to visualize emotion distribution
def visualize_emotion_distribution(directory):
    frequencies = [len(os.listdir(os.path.join(directory, emotion))) for emotion in CLASS_LABELS]
    plt.bar(CLASS_LABELS, frequencies, align='center')
    plt.xlabel('Emotion')
    plt.ylabel('Frequency')
    plt.show()

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

# Function to show augmentation at work
def show_augmentation(train_data):
    model = Sequential([
        RandomZoom(height_factor=(-0.2, -0.05), width_factor=(-0.2, -0.05), 
                   input_shape=[IMG_WIDTH, IMG_HEIGHT, 1]), 
        RandomTranslation(0.2, 0.2, fill_mode= 'nearest'),
        RandomBrightness(factor=0.2, value_range=[0,255]),
        RandomContrast(0.3), 
    ])

    # Load 5 random images from the training dataset
    random_images = []
    for images, _ in train_data.take(1):  # Take 1 batch
        random_images.extend(images[:5])

    # Pass each random image through the model and plot
    plt.figure(figsize=(15, 5))
    for i, random_image in enumerate(random_images):
        # Apply preprocessing layers
        processed_image = random_image.numpy().squeeze()  # Initialize with the original image
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        processed_image = model(processed_image).numpy().squeeze()  # Pass through the model

        plt.subplot(2, 5, i + 1)
        plt.imshow(random_image.numpy().squeeze(), cmap='gray')  # Plot the original image
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(processed_image.squeeze(), cmap='gray')  # Plot the processed image
        plt.title('Processed Image')
        plt.axis('off')

    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    disp.plot()
    plt.show()

# Lights! Camera! ...
def action(model):
    ## source:"https://realpython.com/face-detection-in-python-using-a-webcam/"
    #The 'r' before the string because of the backslashes,
    #otherwise it would be interpreted as an escape character.
    cascPath = r"H:\alfa-training\Deep Learning\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0) # 1 works, 0 (default) does not

    while True:
        # Capture frame-by-frame
        """The 'ret' code in the next line tells us if we have run out of frames, 
        which will happen if we are reading from a file. 
        This doesnt matter when reading from the webcam, 
        since we can record forever, so we will ignore it. 
        """
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,   # not relevant for one person  
            minSize=(48, 48), # fitting size for the model
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # Crop the image using the rectangle coordinates
            y_offset = h# + 100
            x_offset = w# + 20
            cropped_image = frame[y:y+y_offset, x:x+x_offset]

            # Convert the cropped image to grayscale
            cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            # Resize the image to match the input size of the model
            resized_image = cv2.resize(cropped_gray, (IMG_HEIGHT, IMG_WIDTH))

            # Normalize the image
            normalized_image = resized_image / 255.0 # that is bad! -> dont normalize the image

            # Expand dimensions to create a batch dimension
            input_image = np.expand_dims(resized_image, axis=0) # resize
            input_image = np.expand_dims(input_image, axis=-1)  # Add channel dimension for grayscale

            # Make prediction using the loaded model
            prediction = model.predict(input_image)

            # Get the predicted emotion label
            predicted_emotion = CLASS_LABELS[np.argmax(prediction)]
            
            # Get the probability of the predicted emotion
            probability = np.max(prediction)

            # Format the probability as percentage
            probability_percentage = "{:.2f}%".format(probability * 100)

            # Put a label on the rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_emotion} ({probability_percentage})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # # Print an example picture along with its predicted label
            # print(f"Predicted emotion: {predicted_emotion}")
            # print(f"Prediction: {prediction}")
            # plt.imshow(resized_image, cmap='gray')
            # plt.show()
            
        # Display the resulting frame
        cv2.imshow('Video', frame)

        # quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

# Model training, evaluation and test on video
def main():

    # Visualize emotion distribution in training and testing data
    visualize_emotion_distribution(TRAIN_DIR)
    # visualize_emotion_distribution(TEST_DIR)

    # Load Training and Validation Images
    """ 
    we have less train data and more test data (0.25), so we split the test data files into test and validate 
    to validate the training with the 'training' split and Test the model at the very end with this  
    small - never seen before - 'validation' split 
    """
    train_data = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels='inferred',
        label_mode='categorical', 
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        seed=42
    )
    # for training eval
    val_data = tf.keras.utils.image_dataset_from_directory(
    directory=TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=0.2,  # 20% of the data will be used for validation
    subset='training',     # Specifies that this dataset is for training (-validation)
    seed=42
    )
    # for final testing
    test_data = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,  # 20% of the data will be used for validation
        subset='validation',   # Specifies that this dataset is for validation (I call it Test data for the final test)
        seed=42
    )

    show_random_images(TRAIN_DIR, CLASS_LABELS)

    # Calculate class weights
    class_counts = np.zeros(NUM_CLASSES)
    for _, labels in train_data:
        class_counts += np.sum(labels, axis=0)
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (NUM_CLASSES * class_counts)
    class_weights = dict(enumerate(class_weights))

    # Plot class weights
    plt.bar(range(NUM_CLASSES), class_weights.values(), tick_label=CLASS_LABELS)
    plt.xlabel('Emotion')
    plt.ylabel('Class Weight')
    plt.title('Class Weights')
    plt.show()

    # Show samples from augmentation layer (not the actual layer)
    show_augmentation(train_data)

    # If not already existing, make a new model and save it
    model_file = '_Abschlussprojekt/emotion_detection_model_V6'
    if not os.path.exists(model_file):
        
        # Build and compile the model
        model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))  # grayscale images have 1 channel
        es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience=EARLY_STOP_PATIENCE)
        
        # Train the model
        tik = time.time()
        history = model.fit(
            train_data,
            epochs=EPOCHS,
            validation_data=val_data,
            class_weight=class_weights,
            callbacks=es
        )
        # Training timer
        tok = time.time()
        training_time = tok - tik
        print("Training time: {:.2f} seconds".format(training_time))
        
        # Save the trained model
        model.save(model_file)
        
        # Evaluate the model on test data
        test_loss, test_accuracy = model.evaluate(test_data)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)
        
        # Plot training and validation accuracy
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    # Predict labels for test data
    try:
        loaded_model = tf.keras.models.load_model(model_file)
        print(f"'{model_file}' erfolgreich geladen.")
        
        loss, accuracy = loaded_model.evaluate(test_data)

        # Print the loss and accuracy
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        y_true = []
        y_pred_probs = []
        for images, labels in test_data:
            y_true.extend(np.argmax(labels, axis=1))
            y_pred_probs.extend(np.argmax(loaded_model.predict(images), axis=1))
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred_probs)
        # Classification report
        report = classification_report(y_true, y_pred_probs, target_names = CLASS_LABELS)
        print(report)

        # Wähle einen Batch von Bildern aus dem Testdatensatz aus
        for images, labels in test_data.take(1):
            # Mache Vorhersagen für den Batch von Bildern
            predictions = loaded_model.predict(images)

            # Wähle zufällig Bilder aus dem Batch aus
            random_indices = random.sample(range(len(images)), min(9, len(images)))

            # Plotte die ausgewählten Bilder
            plt.figure(figsize=(8, 8))
            for i, index in enumerate(random_indices):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[index].numpy().astype("uint8"))
                plt.title(
                    f"Predicted: {CLASS_LABELS[np.argmax(predictions[index])]}," 
                    f"Actual: {CLASS_LABELS[np.argmax(labels[index])]}", 
                    fontsize=8
                )
                plt.axis("off")
            plt.show()

        action(loaded_model)

    except Exception as e:
        print("Fehler beim Laden des Modells:", e)

if __name__ == "__main__":
    main()


""" model = Sequential([
    RandomContrast(0.5, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)), 
    #RandomBrightness(factor=0.2, value_range=[0,255]), 
    #RandomZoom(height_factor=(-0.2, -0.05), width_factor=(-0.2, -0.05)), 
    #RandomFlip(mode="horizontal"), 
    #Rescaling(scale=1./255),
    # 1st
    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # 2nd
    #Conv2D(64, (3, 3), padding='same'),
    #BatchNormalization(),
    #Activation('relu'),
    #MaxPooling2D(pool_size=(2, 2)),
    # 3rd
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # End conv
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])
model.summary() 

# Define the initial learning rate and decay parameters
initial_learning_rate = 0.01
decay_steps = 500
decay_rate = 0.5

# Define the learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)
# Create the optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# Or Create the optimizer with a fixed LR
#optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', weighted_metrics=['accuracy'])
"""