import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------------------------------------------------------------------
# CONFIGURATION: Define all constants here
# ------------------------------------------------------------------------
# Paths
IMAGES_DIR = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/inference/datasets/original_data_yolo/post/test/images"  # Directory containing the 1024x1024 images
LABELS_DIR = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/inference/datasets/original_data_yolo/post/test/labels"  # Directory containing the YOLO labels
MODEL_PATH = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/best_256_new.pt"  # Path to the YOLO segmentation model
OUTPUT_DIR = "./inference/predictions_256_damage_assesment"  # Directory to save prediction masks
VIS_DIR = "./inference/visualization_256_damage_assesment"  # Directory to save visualizations
TEMP_DIR = "./inference/temp_tiles"  # Temporary directory for tiles

# Process settings
SAVE_INTERVAL = 50  # Save visualization every N images
CONF_THRESHOLD = 0.1  # Confidence threshold for predictions
SKIP_SAVE = False  # Skip saving all masks, save only at intervals
TILE_SIZE = 256  # Size of tiles for inference

# Color mapping (BGR) for each class
COLOR_MAP = {
    0: (0, 255, 0),    # Green (No damage)
    1: (0, 255, 255),  # Yellow (Minor damage)
    2: (0, 165, 255),  # Orange (Major damage)
    3: (0, 0, 255)     # Red (Destroyed)
}

# Inverse mapping: BGR -> class index
COLOR_MAP_INVERSE = {
    (0, 255, 0): 0,
    (0, 255, 255): 1,
    (0, 165, 255): 2,
    (0, 0, 255): 3,
    (255, 255, 255): 4  # "Unknown" class
}

class DamageSegmentation:
    """Class to handle building damage segmentation workflow."""
    
    def __init__(self):
        """Initialize with configuration parameters."""
        self.model = YOLO(MODEL_PATH)
        
        # Create required directories
        for directory in [TEMP_DIR, OUTPUT_DIR, VIS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Stats tracking
        self.metrics = {
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def process_all_images(self):
        """Process all images in the dataset."""
        image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])
        
        for idx, image_file in enumerate(image_files):
            try:
                base_name = Path(image_file).stem
                print(f"Processing image {idx+1}/{len(image_files)}: {base_name}")
                
                # Process individual image
                self.process_single_image(image_file, idx)
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
        
        # Cleanup
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print(f"\nDeleted temporary folder: {TEMP_DIR}")
        
        # Report overall metrics
        self.report_metrics()
    
    def process_single_image(self, image_file, idx):
        """Process a single image, create tiles, run inference, and evaluate."""
        base_name = Path(image_file).stem
        big_image_path = os.path.join(IMAGES_DIR, image_file)
        annotation_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        
        # Load and validate image
        big_image = cv2.imread(big_image_path)
        if big_image is None:
            print(f"Warning: Could not read image file: {big_image_path}")
            return
        
        H, W, _ = big_image.shape
        if (H != 1024) or (W != 1024):
            print(f"Warning: {base_name} is {W}x{H}, expected 1024x1024")
        
        # Create ground truth mask
        gt_mask = self.create_ground_truth_mask(annotation_path, H, W)
        if gt_mask is None:
            return
        
        # Generate prediction mask from tiles
        pred_mask = self.generate_prediction_mask(big_image, base_name, H, W)
        
        # Compute metrics
        precision, recall, f1 = self.compute_metrics(gt_mask, pred_mask)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['f1'].append(f1)
        
        print(f"  -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save visualizations and masks
        should_save = not SKIP_SAVE or idx % SAVE_INTERVAL == 0
        if idx % SAVE_INTERVAL == 0:
            self.save_visualization(big_image, gt_mask, pred_mask, base_name, idx)
        
        if should_save:
            self.save_masks(gt_mask, pred_mask, base_name)
    
    def create_ground_truth_mask(self, annotation_path, height, width):
        """Create ground truth mask from annotation file."""
        gt_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not os.path.exists(annotation_path):
            print(f"Warning: No annotation file found at {annotation_path}. Skipping.")
            return None
        
        with open(annotation_path, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                x_norm, y_norm = coords[i], coords[i + 1]
                x_pix, y_pix = int(x_norm * width), int(y_norm * height)
                points.append((x_pix, y_pix))
            
            # Fill polygon with appropriate color
            points = np.array(points, dtype=np.int32)
            color = COLOR_MAP.get(cls, (255, 255, 255))
            cv2.fillPoly(gt_mask, [points], color)
        
        return gt_mask
    
    def generate_prediction_mask(self, big_image, base_name, height, width):
        """Split image into tiles, perform inference, and create prediction mask."""
        pred_mask = np.zeros((height, width, 3), dtype=np.uint8)
        num_tiles = height // TILE_SIZE  # Assuming square images
        
        for row in range(num_tiles):
            for col in range(num_tiles):
                y1, y2 = row * TILE_SIZE, (row + 1) * TILE_SIZE
                x1, x2 = col * TILE_SIZE, (col + 1) * TILE_SIZE
                
                # Extract and save tile
                tile = big_image[y1:y2, x1:x2]
                tile_path = os.path.join(TEMP_DIR, f"{base_name}_tile_{row}_{col}.png")
                cv2.imwrite(tile_path, tile)
                
                # Run model inference
                results = self.model.predict(tile_path, conf=CONF_THRESHOLD)
                
                # Add predictions to the mask
                if results[0].masks is not None:
                    for seg_polygon, cls_idx in zip(results[0].masks.xy, results[0].boxes.cls):
                        # Adjust polygon coordinates to the original image
                        offset_polygon = seg_polygon + [x1, y1]
                        offset_polygon = offset_polygon.astype(np.int32)
                        
                        # Fill polygon with class color
                        color = COLOR_MAP.get(int(cls_idx), (255, 255, 255))
                        cv2.fillPoly(pred_mask, [offset_polygon], color)
        
        return pred_mask
    
    def compute_metrics(self, gt_mask, pred_mask):
        """Compute precision, recall, and F1 score between ground truth and prediction."""
        # Convert RGB masks to class labels using vectorized operations
        gt_labels = self.bgr_mask_to_labels(gt_mask)
        pred_labels = self.bgr_mask_to_labels(pred_mask)
        
        # Filter out invalid pixels
        valid = (
            (gt_labels != -1) & (pred_labels != -1) &
            (gt_labels != 4) & (pred_labels != 4)
        )
        
        y_true = gt_labels[valid]
        y_pred = pred_labels[valid]
        
        if len(y_true) > 0:
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            return precision, recall, f1
        else:
            return 0, 0, 0
    
    def bgr_mask_to_labels(self, mask):
        """
        Convert BGR mask to label array using vectorized operations.
        
        Args:
            mask: HxWx3 BGR image
            
        Returns:
            HxW array with class indices (-1 for unrecognized colors)
        """
        H, W, _ = mask.shape
        label_img = np.full((H, W), -1, dtype=np.int32)
        
        # Vectorized approach
        for color, class_idx in COLOR_MAP_INVERSE.items():
            # Create a boolean mask for each color
            color_mask = np.all(mask == color, axis=2)
            label_img[color_mask] = class_idx
            
        return label_img
    
    def save_visualization(self, original, gt_mask, pred_mask, base_name, idx):
        """Save visualization of original image, ground truth, and prediction."""
        plt.figure(figsize=(18, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        # Prediction mask
        plt.subplot(1, 3, 3)
        plt.title("Prediction Mask")
        plt.imshow(cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(VIS_DIR, f"visualization_{base_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved visualization to {save_path}")
    
    def save_masks(self, gt_mask, pred_mask, base_name):
        """Save ground truth and prediction masks to output directory."""
        gt_path = os.path.join(OUTPUT_DIR, f"{base_name}_gt_mask.png")
        pred_path = os.path.join(OUTPUT_DIR, f"{base_name}_pred_mask.png")
        
        cv2.imwrite(gt_path, gt_mask)
        cv2.imwrite(pred_path, pred_mask)
    
    def report_metrics(self):
        """Report average metrics across all processed images."""
        if not self.metrics['precision']:
            print("\nNo images were successfully processed. No metrics to report.")
            return
            
        avg_precision = np.mean(self.metrics['precision'])
        avg_recall = np.mean(self.metrics['recall'])
        avg_f1 = np.mean(self.metrics['f1'])
        
        print("\n=== Overall Dataset Metrics (Macro-Average) ===")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall:    {avg_recall:.4f}")
        print(f"F1 Score:  {avg_f1:.4f}")

def main():
    """Main entry point."""
    # Expand relative paths to absolute paths if needed
    global IMAGES_DIR, LABELS_DIR
    if not os.path.isabs(IMAGES_DIR):
        IMAGES_DIR = os.path.abspath(IMAGES_DIR)
    if not os.path.isabs(LABELS_DIR):
        LABELS_DIR = os.path.abspath(LABELS_DIR)
    
    # Create and run the segmentation pipeline
    segmentation = DamageSegmentation()
    segmentation.process_all_images()

if __name__ == "__main__":
    main()
