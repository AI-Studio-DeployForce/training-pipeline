import os
import cv2
import numpy as np
from config import CLASS_COLORS

def load_image(image_path):
    """
    Load an image from disk
    
    Args:
        image_path: Path to the image file
        
    Returns:
        The loaded image or None if loading failed
    """
    image = cv2.imread(image_path)
    return image

def load_labels(label_path, img_width, img_height):
    """
    Load YOLO format labels from a text file
    
    Args:
        label_path: Path to the label file
        img_width: Width of the corresponding image
        img_height: Height of the corresponding image
        
    Returns:
        List of (class, polygon) tuples
    """
    if not os.path.exists(label_path):
        return []
        
    labels = []
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))

        points = []
        for i in range(0, len(coords), 2):
            x_norm = coords[i]
            y_norm = coords[i + 1]
            x_pix = int(x_norm * img_width)
            y_pix = int(y_norm * img_height)
            points.append((x_pix, y_pix))

        labels.append((cls, np.array(points, dtype=np.int32)))
    
    return labels

def create_gt_mask(labels, width, height):
    """
    Create a ground truth mask from labels
    
    Args:
        labels: List of (class, polygon) tuples
        width: Width of the mask
        height: Height of the mask
        
    Returns:
        Ground truth mask as a numpy array
    """
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for cls, points in labels:
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.fillPoly(mask, [points], color)
    
    return mask

def bgr_mask_to_labels(mask, color_arrays):
    """
    Convert a BGR mask to a label mask
    
    Args:
        mask: BGR mask as a numpy array
        color_arrays: Dictionary mapping class indices to color arrays
        
    Returns:
        Label mask as a numpy array
    """
    H, W, _ = mask.shape
    label_img = np.full((H, W), -1, dtype=np.int32)

    # Create a 3D array for comparison
    mask_3d = mask.reshape(-1, 3)
    
    for class_idx, color in color_arrays.items():
        # Vectorized comparison
        matches = np.all(mask_3d == color, axis=1)
        label_img.flat[matches] = class_idx

    return label_img

# Precompute color arrays for faster processing
COLOR_ARRAYS = {k: np.array(v, dtype=np.uint8) for k, v in CLASS_COLORS.items()} 