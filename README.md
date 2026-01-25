# Emotion Detection with Custom CNN
Real-Time Facial Emotion Recognition – Training Pipeline + Windows Demo

Author: Andreas Eichmann  
Date:   25.01.2026

---

## Overview

This project implements a full end-to-end facial emotion recognition system using a custom Convolutional Neural Network (CNN).

It consists of:
- a GPU-accelerated training pipeline running in WSL
- a real-time inference demo running on Windows using a webcam
- a synchronized model + configuration hand-off between both environments

The goal of this project is not only model accuracy, but to demonstrate clean ML software architecture:
- object-oriented design
- dependency injection
- separation of training and inference
- single source of truth for configuration
- production-adjacent deployment patterns

---

## Features

- Custom CNN for facial emotion classification (7 classes)
- End-to-end training pipeline:
  - dataset loading
  - preprocessing
  - class weighting
  - training & validation
  - evaluation (confusion matrix + classification report)

- Real-time webcam demo on Windows:
  - Haar cascade face detection
  - live emotion prediction overlay

- Deliberate WSL / Windows split:
  - WSL was chosen for training due to reliable NVIDIA GPU support
  - Windows was required for real-time inference due to webcam access limitations in WSL
  - Model and configuration are synchronized across environments at runtime

- Clean architecture:
  - OOP + Dependency Injection
  - WSL → Windows file-based contract (model + config sync)
  - no cross-OS Python imports

- GPU-accelerated training using TensorFlow on WSL2

---

## Environment & Tech Stack

Training environment (WSL):

- Windows 11 with WSL2
- NVIDIA GPU (RTX-class)
- CUDA 11.8 runtime (WSL)
- cuDNN 8.8
- TensorFlow 2.13.1 (GPU build)
- Python 3.10 (venv: tf-gpu)

Inference environment (Windows):

- Windows 10/11
- Python 3.11
- TensorFlow (CPU build)
- OpenCV
- Webcam

---

## Model & Dataset

- Architecture: Custom CNN (grayscale 48×48 input)
- Classes:
  - angry
  - disgusted
  - fearful
  - happy
  - neutral
  - sad
  - surprised

- Dataset: FER2013-style facial emotion dataset

### Current baseline performance

- Accuracy: ~55–58 %
- Macro F1: ~0.54–0.56
- Weighted F1: ~0.55–0.58

This is a baseline model with clear room for improvement.  
The focus of this project is primarily engineering quality and architecture.

---

## Repository Structure

emotion_detection_model_V6/
├── src/                 # Training pipeline (WSL)
│   ├── main.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── runtime.py
│   ├── training_pipeline.py
│   └── config.py        # Single source of truth (WSL)
│
├── realtime/            # Realtime inference (Windows)
│   ├── realtime_detector_windows.py
│   └── run_windows_demo.py
│
├── shared/              # Synced config (runtime artifact)
│   └── __init__.py
│
├── assets/              # Haar cascade, etc.
│   └── haarcascade_frontalface_default.xml
│
├── checkpoints/         # Saved models (gitignored)
└── README.md

---

## How to Run

### A) Training (WSL) 

Requirements:
- WSL2
- NVIDIA GPU with CUDA support
- TensorFlow GPU environment

Commands:

source ~/tf-gpu/bin/activate  
python -m src.main

This will:
- configure the TensorFlow runtime
- load and preprocess the dataset
- build the CNN
- train the model
- evaluate on the validation set
- save the trained model to checkpoints/
- export the training configuration

---

### B) Real-Time Demo (Windows)

Requirements:
- Windows
- Python 3.11+
- OpenCV
- TensorFlow (CPU is sufficient)
- Webcam

Command:

py realtime/run_windows_demo.py

This will:
- sync the trained model and config from WSL
- load the model locally
- start the webcam
- detect faces
- classify emotions in real time
- overlay predictions on the video stream

Press q to quit.

---

## Architecture & Design Decisions

### 1) Clean separation of concerns

- Training logic lives exclusively in src/
- Inference logic lives exclusively in realtime/
- No cross-OS Python imports

---

### 2) Single source of truth for configuration

- src/config.py is the authoritative config
- It is synced from WSL to Windows at runtime
- Windows imports the synced config locally

This guarantees that inference always uses exactly the same parameters as training.

---

### 3) File-based contract between WSL and Windows

Instead of attempting cross-OS imports:
- the trained model is copied from WSL to Windows
- the config file is copied from WSL to Windows

This mirrors how models are deployed in real systems.

---

### 4) Object-oriented design + Dependency Injection

Both major components are implemented as injectable services:
- TrainingPipeline
- EmotionDetector

External dependencies (model loader, preprocessing, sync logic) are injected, making the system:
- testable
- extensible
- replaceable (e.g. FER+ dataset, new model, different camera input)

---

## Future Work

- Train on FER+ dataset for improved label quality
- Data augmentation (pose, illumination, occlusion)
- Deeper CNN or transfer learning (e.g. MobileNetV2)
- Temporal smoothing for real-time predictions

---

## Author

Andreas Eichmann  
Applied Data Scientist / ML Engineer (aspiring)

This project was built as a portfolio-grade demonstration of:
- applied deep learning
- ML system design
- clean Python architecture
- production-adjacent engineering practices
