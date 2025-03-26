# YOLOv9 Training Pipeline

This repository contains a comprehensive training pipeline for YOLOv9 object detection model, specifically designed for damage classification in images. The pipeline includes dataset versioning, model training, hyperparameter optimization, and model evaluation capabilities.

## Project Overview

This project implements an end-to-end training pipeline for YOLOv9, a state-of-the-art object detection model, with the following key features:

- Automated dataset versioning and management using ClearML
- End-to-end training pipeline with configurable components
- Hyperparameter optimization using Optuna
- Model evaluation and performance metrics
- Support for damage classification with 4 classes:
  - No-damage
  - Minor-damage
  - Major-damage
  - Destroyed

## Project Structure

```
.
├── datasets/               # Dataset directory
├── predictions_final/     # Model predictions output
├── runs/                  # Training runs and logs
├── data.yaml             # Dataset configuration
├── yolov9_architecture.yaml  # Model architecture configuration
├── yolov9_pipeline.py    # Main pipeline implementation
├── yolov9_training.py    # Training script
├── evaluate_model.py     # Model evaluation script
├── best.pt              # Best trained model weights
└── yolo11n.pt          # Pre-trained model weights
```

## Prerequisites

- Python 3.x
- PyTorch
- Ultralytics YOLO
- ClearML
- Optuna (for hyperparameter optimization)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Dataset Preparation

Place your dataset in the following structure:
```
datasets/
└── dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### 2. Configuration

1. Update `data.yaml` with your dataset paths and class names
2. Configure model architecture in `yolov9_architecture.yaml` if needed
3. Set up ClearML credentials for experiment tracking

### 3. Running the Pipeline

To run the complete training pipeline:

```bash
python yolov9_pipeline.py
```

The pipeline will:
1. Version your dataset
2. Perform base model training
3. Conduct hyperparameter optimization

### 4. Model Evaluation

To evaluate the trained model:

```bash
python evaluate_model.py
```

## Pipeline Components

### Dataset Versioning
- Uses ClearML for dataset versioning and management
- Ensures reproducibility of experiments
- Handles dataset storage and versioning

### Training Pipeline
- Implements YOLOv9 training with configurable parameters
- Supports transfer learning from pre-trained weights
- Includes validation and checkpointing

### Hyperparameter Optimization
- Uses Optuna for hyperparameter optimization
- Optimizes key parameters like learning rate, batch size, and model architecture
- Implements early stopping and model selection

### Model Evaluation
- Comprehensive evaluation metrics
- Performance analysis on test set
- Visualization of results

## Results

The trained model and evaluation results are stored in:
- `best.pt`: Best trained model weights
- `predictions_final/`: Model predictions and evaluation results
- `runs/`: Training logs and metrics


## Acknowledgments

- Ultralytics for the YOLOv9 implementation
- ClearML for experiment tracking and pipeline management
- Optuna for hyperparameter optimization 