import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.ndimage import label

# ------------------------------------------------------------------------
# CONFIGURATION: Define all constants here
# ------------------------------------------------------------------------
# Paths
IMAGES_DIR = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/inference/datasets/original_data_yolo/post/test/images"  # Directory containing the 1024x1024 images
LABELS_DIR = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/inference/datasets/original_data_yolo/post/test/labels"  # Directory containing the YOLO labels
MODEL_PATH = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/best_256_new.pt"  # Path to the YOLO segmentation model
OUTPUT_DIR = "./inference/predictions_256_damage_assesment_postprocessing_final"  # Directory to save prediction masks
VIS_DIR = "./inference/visualization_256_damage_assesment_postprocessing_final"  # Directory to save visualizations
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
    
    def color_to_class_id(self, color):
        """
        Convert a color in the mask to a class ID
        
        Args:
            color: RGB color tuple
            
        Returns:
            Class ID (int)
        """
        # Map from color space to class IDs
        class_colors = {
            COLOR_MAP[0]: 0,    
            COLOR_MAP[1]: 1,    
            COLOR_MAP[2]: 2,    
            COLOR_MAP[3]: 3
        }
        
        return class_colors.get(tuple(color), 0)  # Default to background (0) if color not found

    def apply_morphological_operations(self, pred_mask, visualize=False, output_dir=None):
        """
        Apply morphological operations to denoise, fill holes and separate connected masks
        for each class separately, handling overlaps between classes.
        With visualization of each step.
        
        Args:
            pred_mask: The combined prediction mask with all classes
            visualize: Whether to visualize each step
            output_dir: Directory to save visualizations (if None, will just display)
            
        Returns:
            Processed prediction mask
        """
        
        # Create a copy of the mask to avoid modifying the original
        processed_mask = np.zeros_like(pred_mask)
        
        # Create a "class priority" map to handle overlaps
        priority_map = np.zeros(pred_mask.shape[:2], dtype=np.uint8)
        
        # Get unique colors from the mask (excluding background which is 0)
        unique_colors = np.unique(pred_mask.reshape(-1, pred_mask.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors if np.any(color != 0)]
        
        # Optional: Define class priorities
        class_priorities = {
            0: 1,  # Background (lowest priority)
            1: 2,
            2: 3,
            3: 4   # Highest priority
        }
        
        # Create a list of (color, priority) tuples and sort by priority
        color_priorities = []
        for color in unique_colors:
            # Get class ID for this color
            class_id = self.color_to_class_id(color)  
            priority = class_priorities.get(class_id, 1)
            color_priorities.append((color, priority))
        
        # Sort by priority (low to high)
        color_priorities.sort(key=lambda x: x[1])
        
        # Prepare visualization directory
        if visualize and output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process each class separately, from lowest to highest priority
        for i, (color, priority) in enumerate(color_priorities):
            class_id = self.color_to_class_id(color)
            
            # Extract binary mask for this class
            binary_mask = np.all(pred_mask == color, axis=2).astype(np.uint8) * 255
            
            # For visualization
            if visualize:
                stages = {"1_original": binary_mask.copy()}
            
            # 1. Remove small noise (erosion followed by dilation = opening)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            if visualize:
                stages["2_opened"] = opened.copy()
            
            # 2. Fill holes (dilation followed by erosion = closing)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            if visualize:
                stages["3_closed"] = closed.copy()
            
            # 3. Separate connected instances 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            separated = cv2.erode(closed, kernel, iterations=1)
            
            if visualize:
                stages["4_separated"] = separated.copy()
            
            # 4. Remove very small objects
            contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = np.zeros_like(separated)
            min_area = 50  
            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    cv2.drawContours(filtered, [contour], 0, 255, -1)
            
            if visualize:
                stages["5_filtered"] = filtered.copy()
            
            # 5. Update the result only where this class has higher priority than existing content
            class_result = np.zeros_like(pred_mask)
            for y in range(filtered.shape[0]):
                for x in range(filtered.shape[1]):
                    if filtered[y, x] > 0 and priority > priority_map[y, x]:
                        processed_mask[y, x] = color
                        priority_map[y, x] = priority
                        class_result[y, x] = color
            
            if visualize:
                # Visualize the masks for this class
                fig, axs = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(f'Class {class_id} (Priority {priority})', fontsize=16)
                
                # Display original and all processing steps
                axs[0, 0].imshow(stages["1_original"], cmap='gray')
                axs[0, 0].set_title('Original Binary Mask')
                
                axs[0, 1].imshow(stages["2_opened"], cmap='gray')
                axs[0, 1].set_title('After Opening (Noise Removal)')
                
                axs[0, 2].imshow(stages["3_closed"], cmap='gray')
                axs[0, 2].set_title('After Closing (Fill Holes)')
                
                axs[1, 0].imshow(stages["4_separated"], cmap='gray')
                axs[1, 0].set_title('After Erosion (Separation)')
                
                axs[1, 1].imshow(stages["5_filtered"], cmap='gray')
                axs[1, 1].set_title('After Small Object Removal')
                
                # Display this class's contribution to final result
                axs[1, 2].imshow(class_result)
                axs[1, 2].set_title('Added to Final Result')
                
                # Adjust layout
                plt.tight_layout()
                
                # Save or display
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'class_{class_id}_process.png'))
                    plt.close()
                else:
                    plt.show()
        
        # Visualize final combined result with all classes
        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(processed_mask)
            plt.title('Final Processed Mask (All Classes)')
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'final_processed_mask.png'))
                plt.close()
            else:
                plt.show()
                
            # Visualize original vs processed for comparison
            plt.figure(figsize=(18, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(pred_mask)
            plt.title('Original Mask')
            
            plt.subplot(1, 2, 2)
            plt.imshow(processed_mask)
            plt.title('Processed Mask')
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'before_after_comparison.png'))
                plt.close()
            else:
                plt.show()
        
        return processed_mask
    
    def majority_voting_building_damage_mask(
        self,
        rgb_mask: np.ndarray,
        kernel_size: int = 3,
        dilate_iter: int = 1,
        return_rgb: bool = True
        ) -> np.ndarray:
            """
            Post‑process a YOLO‑style RGB mask so each building instance
            has exactly one damage class (connected‑component majority vote).

            Parameters
            ----------
            rgb_mask : np.ndarray
                H×W×3 array with background = black (0,0,0) and one unique
                RGB colour per damage class.
            kernel_size : int, optional
                Size of the square structuring element used to dilate the
                building mask before finding connected components.
            dilate_iter : int, optional
                Number of dilation iterations (helps bridge 1‑pixel gaps).
            return_rgb : bool, optional
                If True, return an RGB mask with the same colours as the
                input.  If False, return an integer label map where
                0 = background and 1…K = damage classes.

            Returns
            -------
            np.ndarray
                Post‑processed mask in the requested format.
            """
            # ---------- helpers -----------------------------------------------------
            def _rgb_to_label(img):
                colours = np.unique(img.reshape(-1, 3), axis=0)
                colour2id = {tuple(c): i + 1          # start at 1, reserve 0 for bg
                            for i, c in enumerate(colours) if not np.all(c == 0)}
                label_map = np.zeros(img.shape[:2], dtype=np.int32)
                for col, idx in colour2id.items():
                    label_map[np.all(img == col, axis=-1)] = idx
                return label_map, colour2id

            def _label_to_rgb(lbl, colour2id):
                id2colour = {v: k for k, v in colour2id.items()}
                out = np.zeros((*lbl.shape, 3), dtype=np.uint8)
                for idx, col in id2colour.items():
                    out[lbl == idx] = col
                return out
            # -----------------------------------------------------------------------

            # 1)  RGB → integer label map
            label_map, colour_map = _rgb_to_label(rgb_mask)

            # 2)  Dilate the binary building mask to merge touching fragments
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            building_bin = (label_map > 0).astype(np.uint8)
            building_dil = cv2.dilate(building_bin, kernel, iterations=dilate_iter)

            # 3)  Connected components on the dilated mask
            blobs, n_blobs = label(building_dil, structure=np.ones((3, 3)))

            # 4)  Majority vote inside each component
            refined = label_map.copy()
            for b in range(1, n_blobs + 1):
                mask = blobs == b
                if not np.any(mask):
                    continue
                vals, counts = np.unique(refined[mask], return_counts=True)
                majority_cls = vals[counts.argmax()]
                refined[mask] = majority_cls

            # 5)  Return in the requested format
            if return_rgb:
                return _label_to_rgb(refined, colour_map)
            return refined

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
        
        pred_mask = self.apply_morphological_operations(pred_mask)
        pred_mask = self.majority_voting_building_damage_mask(pred_mask,3,3)
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
