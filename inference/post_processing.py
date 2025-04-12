import os
import cv2
import numpy as np
import shutil
from glob import glob
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Define paths
pre_disaster_folder = "inference/predictions_final_localizations"
post_disaster_folder = "inference/predictions_256_damage_assesment"
output_folder = "inference/combined_predictions"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Class colors as defined
CLASS_COLORS = {
    0: (0, 255, 0),    # Green (No damage)
    1: (0, 255, 255),  # Yellow (Minor damage)
    2: (0, 165, 255),  # Orange (Major damage)
    3: (0, 0, 255)     # Red (Destroyed)
}

# Class names for reporting
CLASS_NAMES = {
    0: "No Damage",
    1: "Minor Damage",
    2: "Major Damage",
    3: "Destroyed"
}

# Get all prediction files from pre-disaster folder
pre_disaster_files = glob(os.path.join(pre_disaster_folder, "*pre_disaster_pred_mask.png"))

# Number of classes including background
num_classes = len(CLASS_COLORS) + 1

# Initialize confusion matrix for incremental metrics calculation
conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

# Process each pre-disaster file
total_processed = 0
for pre_file in pre_disaster_files:
    # Get the base image name
    base_name = os.path.basename(pre_file)
    
    # Construct corresponding post-disaster filename
    post_file_name = base_name.replace("pre_disaster", "post_disaster")
    post_file = os.path.join(post_disaster_folder, post_file_name)
    
    # Check if post-disaster file exists
    if not os.path.exists(post_file):
        print(f"Warning: No matching post-disaster file for {base_name}")
        continue
    
    # Load pre and post disaster images
    pre_img = cv2.imread(pre_file)
    post_img = cv2.imread(post_file)
    
    # Create a mask for buildings in pre-disaster (green pixels)
    pre_building_mask = np.all(pre_img == CLASS_COLORS[0], axis=2)
    
    # Create a mask for background in post-disaster (black pixels)
    post_background_mask = np.all(post_img == [0, 0, 0], axis=2)
    
    # Identify pixels that are buildings in pre-disaster but background in post-disaster
    destroyed_mask = pre_building_mask & post_background_mask
    
    # Create the combined image (start with post-disaster image)
    combined_img = post_img.copy()
    
    # Mark destroyed buildings as class 3 (red)
    combined_img[destroyed_mask] = CLASS_COLORS[3]
    
    # Save the combined image
    combined_output_path = os.path.join(output_folder, post_file_name)
    cv2.imwrite(combined_output_path, combined_img)
    
    # Find and load the corresponding ground truth for metrics calculation
    gt_file_name = post_file_name.replace("pred_mask", "gt_mask")
    gt_file = os.path.join(post_disaster_folder, gt_file_name)
    
    if os.path.exists(gt_file):
        # Copy the ground truth to the output folder
        gt_output_path = os.path.join(output_folder, gt_file_name)
        shutil.copy(gt_file, gt_output_path)
        
        # Load ground truth for metrics calculation
        gt_img = cv2.imread(gt_file)
        
        # Convert predictions and ground truth to class indices
        pred_classes = np.zeros(combined_img.shape[:2], dtype=np.uint8)
        gt_classes = np.zeros(gt_img.shape[:2], dtype=np.uint8)
        
        # Background (black) stays as 0 index
        # Map RGB colors to class indices for predictions
        for class_idx, color in CLASS_COLORS.items():
            pred_classes[np.all(combined_img == color, axis=2)] = class_idx + 1  # +1 to reserve 0 for background
        
        # Map RGB colors to class indices for ground truth
        for class_idx, color in CLASS_COLORS.items():
            gt_classes[np.all(gt_img == color, axis=2)] = class_idx + 1  # +1 to reserve 0 for background
        
        # Update confusion matrix incrementally
        # This is memory efficient as we don't store all pixel predictions
        for i in range(num_classes):
            for j in range(num_classes):
                conf_matrix[i, j] += np.sum((gt_classes == i) & (pred_classes == j))
        
        total_processed += 1
        if total_processed % 10 == 0:
            print(f"Processed {total_processed} image pairs...")
    else:
        print(f"Warning: No ground truth file found for {post_file_name}")

print(f"Processing complete! Combined predictions saved to {output_folder}")

# Calculate pixel-level metrics from confusion matrix
if total_processed > 0:
    # Class names including background
    class_names = ['Background'] + [CLASS_NAMES[i] for i in range(len(CLASS_COLORS))]
    
    # Initialize arrays for metrics - skip background (index 0)
    damage_classes = num_classes - 1  # Number of actual damage classes
    precision = np.zeros(damage_classes)
    recall = np.zeros(damage_classes)
    f1_score = np.zeros(damage_classes)
    support = np.zeros(damage_classes)
    
    # Calculate metrics for each class (skip background class at index 0)
    for i in range(1, num_classes):  # Start from 1 to skip background
        # True positives are on the diagonal
        tp = conf_matrix[i, i]
        
        # Sum of row = all actual class instances
        actual_sum = np.sum(conf_matrix[i, :])
        
        # Sum of column = all predicted class instances
        pred_sum = np.sum(conf_matrix[:, i])
        
        # Calculate precision, recall, and F1 score
        precision[i-1] = tp / pred_sum if pred_sum > 0 else 0  # i-1 because we're skipping background
        recall[i-1] = tp / actual_sum if actual_sum > 0 else 0
        f1_score[i-1] = 2 * precision[i-1] * recall[i-1] / (precision[i-1] + recall[i-1]) if (precision[i-1] + recall[i-1]) > 0 else 0
        support[i-1] = actual_sum
    
    # Create DataFrame for results - only showing damage classes (no background)
    metrics_df = pd.DataFrame({
        'Class': [CLASS_NAMES[i] for i in range(len(CLASS_COLORS))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Support (pixels)': support
    })
    
    # Calculate overall metrics
    total_support = np.sum(support)
    if total_support > 0:
        weighted_precision = np.sum(precision * support) / total_support
        weighted_recall = np.sum(recall * support) / total_support
        weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0
    
    # Add overall metrics to the DataFrame
    metrics_df.loc[len(metrics_df)] = [
        'Overall', 
        weighted_precision, 
        weighted_recall, 
        weighted_f1, 
        total_support
    ]
    
    # Save metrics to CSV
    metrics_path = os.path.join(output_folder, 'pixel_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # Print metrics
    print("\nPixel-level Metrics per Class (excluding background):")
    print(metrics_df)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save confusion matrix - still include background for completeness
    cm_df = pd.DataFrame(
        conf_matrix, 
        index=['True ' + c for c in class_names], 
        columns=['Pred ' + c for c in class_names]
    )
    cm_path = os.path.join(output_folder, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
else:
    print("No ground truth files found for metric calculation.")