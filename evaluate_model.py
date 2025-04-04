import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# ------------------------------------------------------------------------
# CONFIG: Flag to control image saving behavior
# ------------------------------------------------------------------------
skip_save = True  # When True, only saves visualization images every 50 iterations

# ------------------------------------------------------------------------
# CONFIG: Folder paths for big 1024x1024 images and their .txt labels
# ------------------------------------------------------------------------
images_dir = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/datasets/original_data_yolo/pre/test/images"  # Folder with 1024x1024 .png images
labels_dir = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/datasets/original_data_yolo/pre/test/labels"  # Folder with corresponding .txt labels

# Temporary folder for 256x256 tiles
temp_dir = "temp_tiles"
os.makedirs(temp_dir, exist_ok=True)

# Folder where we will save final masks
predictions_dir = "predictions_final_256_new_pre"
os.makedirs(predictions_dir, exist_ok=True)

# ------------------------------------------------------------------------
# CONFIG: Load segmentation model (must be YOLO-seg)
# ------------------------------------------------------------------------
model = YOLO("best_localization.pt")  # e.g. YOLOv9-seg model

# ------------------------------------------------------------------------
# COLOR MAPPING (BGR) for each class
# ------------------------------------------------------------------------
fixed_colors = {
    0: (0, 255, 0),    # Green (No damage)
    1: (0, 255, 255),  # Yellow (Minor damage)
    2: (0, 165, 255),  # Orange (Major damage)
    3: (0, 0, 255)     # Red (Destroyed)
}

# Precompute color arrays for vectorized operations
color_arrays = {k: np.array(v, dtype=np.uint8) for k, v in fixed_colors.items()}

def bgr_mask_to_labels_vectorized(mask, color_arrays):
    """Vectorized version of color matching"""
    H, W, _ = mask.shape
    label_img = np.full((H, W), -1, dtype=np.int32)
    
    # Create a 3D array for comparison
    mask_3d = mask.reshape(-1, 3)
    
    for class_idx, color in color_arrays.items():
        # Vectorized comparison
        matches = np.all(mask_3d == color, axis=1)
        label_img.flat[matches] = class_idx
        
    return label_img

def process_tile(args):
    """Process a single tile in memory"""
    tile, confidence_threshold = args
    results = model.predict(tile, conf=confidence_threshold, verbose=False)
    return results[0].masks, results[0].boxes.cls if results[0].masks is not None else None

def process_image_with_confidence(image_path, label_path, confidence_threshold):
    # Load image
    big_image = cv2.imread(image_path)
    if big_image is None:
        return None, None

    H, W, _ = big_image.shape
    if (H != 1024) or (W != 1024):
        return None, None

    # Build ground truth mask
    gt_mask = np.zeros((H, W, 3), dtype=np.uint8)
    if not os.path.exists(label_path):
        return None, None

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
            x_pix = int(x_norm * W)
            y_pix = int(y_norm * H)
            points.append((x_pix, y_pix))

        points = np.array(points, dtype=np.int32)
        color = fixed_colors.get(cls, (255, 255, 255))
        cv2.fillPoly(gt_mask, [points], color)

    # Prepare prediction mask
    pred_mask = np.zeros((H, W, 3), dtype=np.uint8)

    # Process tiles sequentially
    tile_size = 256
    num_tiles = 4

    for row in range(num_tiles):
        for col in range(num_tiles):
            y1 = row * tile_size
            y2 = (row + 1) * tile_size
            x1 = col * tile_size
            x2 = (col + 1) * tile_size
            
            # Process single tile
            tile = big_image[y1:y2, x1:x2]
            results = model.predict(tile, conf=confidence_threshold, iou=0.6, verbose=False)
            
            if results[0].masks is not None:
                for seg_polygon, cls_idx in zip(results[0].masks.xy, results[0].boxes.cls):
                    offset_polygon = seg_polygon + [x1, y1]
                    offset_polygon = offset_polygon.astype(np.int32)
                    color = fixed_colors.get(int(cls_idx), (255, 255, 255))
                    cv2.fillPoly(pred_mask, [offset_polygon], color)
            
            # Clear memory
            del tile
            del results

    # Clear memory
    del big_image

    return gt_mask, pred_mask

def calculate_metrics(gt_mask, pred_mask):
    gt_labels = bgr_mask_to_labels_vectorized(gt_mask, color_arrays)
    pred_labels = bgr_mask_to_labels_vectorized(pred_mask, color_arrays)

    # Only exclude invalid class 4, but keep background (-1)
    valid = (gt_labels != 4) & (pred_labels != 4)
    
    y_true = gt_labels[valid]
    y_pred = pred_labels[valid]

    # Initialize results dictionary
    metrics = {
        'per_class': {},
        'weighted': {'precision': 0, 'recall': 0, 'f1': 0},
        'macro': {'precision': 0, 'recall': 0, 'f1': 0}
    }

    if len(y_true) > 0:
        # Get all unique classes including background (-1)
        all_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
        
        # Store per-class metrics
        for i, cls in enumerate(all_classes):
            metrics['per_class'][cls] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'count': int(np.sum(y_true == cls))
            }
        
        # Calculate averages for the curves
        metrics['weighted']['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['weighted']['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['weighted']['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Calculate macro average
        metrics['macro']['precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['macro']['recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['macro']['f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    # Return all metrics
    return metrics

def process_single_image(args):
    """Process a single image with a specific confidence threshold"""
    image_file, confidence = args
    base_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(images_dir, image_file)
    label_path = os.path.join(labels_dir, base_name + ".txt")

    gt_mask, pred_mask = process_image_with_confidence(image_path, label_path, confidence)
    if gt_mask is None or pred_mask is None:
        return None

    metrics = calculate_metrics(gt_mask, pred_mask)
    
    # Clear memory
    del gt_mask
    del pred_mask
    
    return metrics

def main():
    all_image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    
    # Use fewer workers to reduce memory usage
    num_workers = min(4, mp.cpu_count())  # Limit to 4 workers max
    
    # Test different confidence thresholds
    confidence_thresholds = np.linspace(0, 1, 21)
    precisions = []
    recalls = []
    f1_scores = []

    # Main progress bar for confidence thresholds
    for conf in tqdm(confidence_thresholds, desc="Testing confidence thresholds"):
        print(f"\nTesting confidence threshold: {conf:.2f}")
        
        # Process images in parallel with fewer workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a list of futures for progress tracking
            futures = [executor.submit(process_single_image, (image_file, conf)) 
                      for image_file in all_image_files]
            
            # Process results with progress bar
            all_metrics = []
            for future in tqdm(futures, desc="Processing images", leave=False):
                result = future.result()
                if result is not None:
                    all_metrics.append(result)
        
        # Skip if no valid results
        if not all_metrics:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            continue
            
        # Aggregate results for all classes across all images
        aggregated_metrics = {
            'per_class': {},
            'weighted': {'precision': 0, 'recall': 0, 'f1': 0},
            'macro': {'precision': 0, 'recall': 0, 'f1': 0}
        }
        
        # Combine all class occurrences
        all_classes = set()
        for metric in all_metrics:
            all_classes.update(metric['per_class'].keys())
        
        # Initialize aggregated per-class metrics
        for cls in all_classes:
            aggregated_metrics['per_class'][cls] = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'count': 0
            }
        
        # Sum up metrics and counts
        for metric in all_metrics:
            for cls, values in metric['per_class'].items():
                for key in ['precision', 'recall', 'f1']:
                    # Weight by pixel count
                    aggregated_metrics['per_class'][cls][key] += values[key] * values['count']
                aggregated_metrics['per_class'][cls]['count'] += values['count']
        
        # Calculate weighted averages for per-class metrics
        for cls, values in aggregated_metrics['per_class'].items():
            if values['count'] > 0:
                for key in ['precision', 'recall', 'f1']:
                    values[key] /= values['count']
        
        # Calculate dataset-wide averages
        total_weighted_precision = 0
        total_weighted_recall = 0
        total_weighted_f1 = 0
        total_pixels = 0
        
        # Calculate weighted averages
        for cls, values in aggregated_metrics['per_class'].items():
            total_weighted_precision += values['precision'] * values['count']
            total_weighted_recall += values['recall'] * values['count']
            total_weighted_f1 += values['f1'] * values['count']
            total_pixels += values['count']
        
        if total_pixels > 0:
            aggregated_metrics['weighted']['precision'] = total_weighted_precision / total_pixels
            aggregated_metrics['weighted']['recall'] = total_weighted_recall / total_pixels
            aggregated_metrics['weighted']['f1'] = total_weighted_f1 / total_pixels
        
        # Calculate macro averages (unweighted)
        macro_precision = np.mean([values['precision'] for values in aggregated_metrics['per_class'].values()])
        macro_recall = np.mean([values['recall'] for values in aggregated_metrics['per_class'].values()])
        macro_f1 = np.mean([values['f1'] for values in aggregated_metrics['per_class'].values()])
        
        aggregated_metrics['macro']['precision'] = macro_precision
        aggregated_metrics['macro']['recall'] = macro_recall
        aggregated_metrics['macro']['f1'] = macro_f1
        
        # Now print the final aggregated results
        print("\nPer-class results:")
        for cls in sorted(aggregated_metrics['per_class'].keys()):
            values = aggregated_metrics['per_class'][cls]
            class_name = "Background" if cls == -1 else f"Class {cls}"
            print(f"{class_name}: Precision={values['precision']:.4f}, Recall={values['recall']:.4f}, F1={values['f1']:.4f}, Pixels={values['count']}")
        
        print(f"\nWeighted Average: Precision={aggregated_metrics['weighted']['precision']:.4f}, "
              f"Recall={aggregated_metrics['weighted']['recall']:.4f}, "
              f"F1={aggregated_metrics['weighted']['f1']:.4f}")
        
        print(f"Macro Average: Precision={aggregated_metrics['macro']['precision']:.4f}, "
              f"Recall={aggregated_metrics['macro']['recall']:.4f}, "
              f"F1={aggregated_metrics['macro']['f1']:.4f}")
        
        # Store weighted averages for plotting
        precisions.append(aggregated_metrics['weighted']['precision'])
        recalls.append(aggregated_metrics['weighted']['recall'])
        f1_scores.append(aggregated_metrics['weighted']['f1'])
        
        # Clear memory after each confidence threshold
        del all_metrics
        del futures

    # Find optimal confidence threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_conf = confidence_thresholds[optimal_idx]
    print(f"\nOptimal confidence threshold: {optimal_conf:.2f}")
    print(f"At this threshold - Precision: {precisions[optimal_idx]:.4f}, Recall: {recalls[optimal_idx]:.4f}, F1: {f1_scores[optimal_idx]:.4f}")

    # Plot the curves
    plt.figure(figsize=(15, 5))

    # Confidence vs Precision/Recall
    plt.subplot(1, 3, 1)
    plt.plot(confidence_thresholds, precisions, 'b-', label='Precision')
    plt.plot(confidence_thresholds, recalls, 'r-', label='Recall')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Confidence vs Precision/Recall')
    plt.legend()
    plt.grid(True)

    # Precision vs Recall
    plt.subplot(1, 3, 2)
    plt.plot(recalls, precisions, 'g-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.grid(True)

    # Confidence vs F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(confidence_thresholds, f1_scores, 'm-')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('Confidence vs F1 Score')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('performance_curves.png')
    plt.close()

    # Clean up
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
