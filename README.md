# Emotion Detection Model (FER2013 / FER+)

This repository contains a **clean, modular emotion recognition pipeline** built with TensorFlow.
It supports **FER2013** and **FER+ (hard labels)** using a **shared image base**, automatic dataset setup,
and a reproducible training and evaluation pipeline.

This is an **engineering-focused project**, not a research publication.

---

> **Note:** This repository is a work in progress and subject to change.


## Features

- Support for **FER2013** and **FER+**
- **Single shared image base** (no duplicated images)
- Automatic dataset download and preparation
- Modular architecture (datasets, model, training, evaluation)
- GPU support (CUDA / cuDNN)
- Reproducible training pipeline
- Clear separation of concerns



## Project Structure

```text
src/
├── config.py                 # Central configuration
├── main.py                   # Entry point
├── runtime.py                # TensorFlow runtime setup
├── model.py                  # Model architecture
├── trainer.py                # Training logic
├── evaluator.py              # Evaluation & metrics
├── training_pipeline.py      # Pipeline orchestration
├── data_download.py          # Dataset download & preparation
└── datasets/
    ├── base.py               # BaseDataset abstraction
    ├── factory.py            # Dataset factory
    ├── fer2013.py            # FER2013 dataset (CSV-based)
    └── ferplus.py            # FER+ dataset (hard labels)
```



## Datasets

### FER2013

- Original FER2013 dataset
- Images are generated once from `icml_face_data.csv`
- Dataset splits are taken from the CSV (`Training`, `PublicTest`)
- Uses the **original FER2013 image set**

### FER+

- Uses the **same FER2013 images**
- Labels are taken from `fer2013new.csv`
- Hard labels generated via **majority vote**
- Optional filtering:
  - minimum agreement threshold
  - contempt removal
  - not-face removal
- Uses official FER+ splits (`Training`, `PublicTest`, `PrivateTest`)

### Data Handling

- Datasets are **not committed** to the repository
- All required data is prepared automatically on first run
- Fail-fast behavior if required files are missing
- Dataset download and preparation logic lives in `data_download.py`



## Configuration

All configuration is centralized in **`src/config.py`**, including:

- Dataset selection (`fer2013` or `ferplus`)
- Image size and batch size
- Training hyperparameters
- FER+ filtering options
- Runtime settings


## Running the Project

### Requirements

- Python 3.10
- TensorFlow 2.13
- CUDA 11.8 + cuDNN 8.8 (optional, for GPU)

### Run

```bash
python -m src.main
```

On first run:
- FER2013 data is downloaded and images are generated
- FER+ labels are downloaded automatically if selected



## Evaluation

The pipeline reports:

- Accuracy
- Macro F1 score
- Weighted F1 score
- Classification report
- Confusion matrix

Due to class imbalance, **weighted F1** is the recommended metric for comparison.



## Design Notes

- Datasets are implemented as first-class objects
- Training, evaluation, and runtime configuration are decoupled
- The pipeline is dataset-agnostic once data is prepared
- The project prioritizes clarity and reproducibility over maximum performance



## Scope

- This repository is intended for **engineering, experimentation, and demonstration**
- It is **not a research project**
- No claims of state-of-the-art performance are made



## License

The code is released under the repository license.
Datasets are subject to their respective original licenses.
