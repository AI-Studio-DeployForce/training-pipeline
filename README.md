# Building Damage Assessment Model Evaluator

This package evaluates a YOLO segmentation model for building damage assessment, calculating precision, recall, and F1 scores across various confidence thresholds.

## Project Structure

The codebase is organized into several modules:

- `config.py` - Configuration settings for paths, model, and visualization
- `utils.py` - Utility functions for image processing and mask handling
- `model.py` - Model loading and inference functionality
- `metrics.py` - Metrics calculation and aggregation
- `visualization.py` - Plotting and results visualization
- `evaluator.py` - Core evaluation logic
- `evaluate_model.py` - Main script to run the evaluation

## Usage

1. Adjust settings in `config.py` to match your environment:
   - Set `IMAGES_DIR` and `LABELS_DIR` to your test image and label directories
   - Set `MODEL_PATH` to your YOLO model path

2. Run the evaluation:
   ```bash
   python evaluate_model.py
   ```

3. The script will:
   - Evaluate the model on all test images
   - Test different confidence thresholds
   - Calculate metrics for each class
   - Find the optimal confidence threshold
   - Generate performance curve plots

## Output

- Console output with metrics for each threshold
- `performance_curves.png` showing precision/recall curves for each class

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Ultralytics (YOLO)
- scikit-learn
- tqdm 