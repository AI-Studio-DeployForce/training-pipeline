import os

# Folder paths for input data
IMAGES_DIR = "datasets/original_data_yolo/post/test/images"
LABELS_DIR = "datasets/original_data_yolo/post/test/labels"

# Model configuration
MODEL_PATH = "training-pipeline/best.pt"
POST_DISASTER = True
# Processing parameters
TILE_SIZE = 256
NUM_TILES = 4
CONFIDENCE_THRESHOLDS = [round(i/20, 2) for i in range(21)]  # 0 to 1 in steps of 0.05
NUM_WORKERS = min(3, os.cpu_count())  # Limit to 4 workers max
IOU_THRESHOLD = 0.6

# Color mapping (BGR) for each class
CLASS_COLORS = {
    0: (0, 255, 0),    # Green (No damage)
    1: (0, 255, 255),  # Yellow (Minor damage)
    2: (0, 165, 255),  # Orange (Major damage)
    3: (0, 0, 255)     # Red (Destroyed)
}

# Class names
CLASS_NAMES = {
    0: "No damage",
    1: "Minor damage",
    2: "Major damage",
    3: "Destroyed"
}

# Plot configurations
PLOT_STYLES = {
    0: {'color': 'green', 'linestyle': '-', 'marker': 'o', 'label': f'Class 0 ({CLASS_NAMES[0]})'},
    1: {'color': 'blue', 'linestyle': '--', 'marker': 's', 'label': f'Class 1 ({CLASS_NAMES[1]})'},
    2: {'color': 'orange', 'linestyle': '-.', 'marker': '^', 'label': f'Class 2 ({CLASS_NAMES[2]})'},
    3: {'color': 'red', 'linestyle': ':', 'marker': 'D', 'label': f'Class 3 ({CLASS_NAMES[3]})'},
    'avg': {'color': 'black', 'linestyle': '-', 'marker': 'x', 'label': 'Average (weighted)'}
}

# Output settings
OUTPUT_PLOT_FILENAME = 'performance_curves_256.png'
OUTPUT_PLOT_DPI = 300 