import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from scipy.ndimage import label
import yaml
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

@dataclass
class Config:
    """Configuration class to manage all settings."""
    # Paths
    images_dir: str
    labels_dir: str
    model_path: str
    output_dir: str
    vis_dir: str
    temp_dir: str
    
    # Process settings
    save_interval: int
    conf_threshold: float
    skip_save: bool
    tile_size: int
    ground_truth: bool  # Flag to determine if ground truth should be saved
    
    # Color mapping (BGR) for each class
    color_map: Dict[int, Tuple[int, int, int]]
    color_map_inverse: Dict[Tuple[int, int, int], int]
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Convert color_map_inverse keys from string to tuple
        color_map_inverse = {}
        for k, v in config_data.get('color_map_inverse', {}).items():
            # Convert string like "(0, 255, 0)" to tuple (0, 255, 0)
            color_tuple = eval(k) if isinstance(k, str) else k
            color_map_inverse[color_tuple] = v
        
        config_data['color_map_inverse'] = color_map_inverse
        
        return cls(**config_data)
    
    @classmethod
    def default_config(cls) -> 'Config':
        """Create default configuration."""
        return cls(
            images_dir="/home/diego/Documents/master/S4/AI_studio/training-pipeline/inference/datasets/original_data_yolo/post/test/images",
            labels_dir="/home/diego/Documents/master/S4/AI_studio/training-pipeline/inference/datasets/original_data_yolo/post/test/labels",
            model_path="/home/diego/Documents/master/S4/AI_studio/training-pipeline/best_256_new.pt",
            output_dir="./inference/predictions_256_damage_assesment_postprocessing_final",
            vis_dir="./inference/visualization_256_damage_assesment_postprocessing_final",
            temp_dir="./inference/temp_tiles",
            
            save_interval=50,
            conf_threshold=0.1,
            skip_save=False,
            tile_size=256,
            ground_truth=False,  # Default to not saving ground truth
            
            color_map={
                0: (0, 255, 0),    # Green (No damage)
                1: (0, 255, 255),  # Yellow (Minor damage)
                2: (0, 165, 255),  # Orange (Major damage)
                3: (0, 0, 255)     # Red (Destroyed)
            },
            color_map_inverse={
                (0, 255, 0): 0,
                (0, 255, 255): 1,
                (0, 165, 255): 2,
                (0, 0, 255): 3,
                (255, 255, 255): 4  # "Unknown" class
            }
        )

    def create_directories(self):
        """Create all necessary directories."""
        for directory in [self.temp_dir, self.output_dir, self.vis_dir]:
            os.makedirs(directory, exist_ok=True)
        
    def save_to_yaml(self, output_path: str):
        """Save configuration to YAML file."""
        # Convert tuples to strings for YAML serialization
        config_dict = self.__dict__.copy()
        
        # Convert color_map_inverse keys from tuples to strings
        color_map_inverse_str = {}
        for k, v in config_dict['color_map_inverse'].items():
            color_map_inverse_str[str(k)] = v
        config_dict['color_map_inverse'] = color_map_inverse_str
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# ------------------------------------------------------------------------
# Data Handling
# ------------------------------------------------------------------------

class DataHandler:
    """Class to handle data loading and processing."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def list_image_files(self) -> List[str]:
        """List all image files in the dataset."""
        return sorted([f for f in os.listdir(self.config.images_dir) if f.endswith(".png")])
    
    def load_image(self, image_file: str) -> Optional[np.ndarray]:
        """Load image from file."""
        image_path = os.path.join(self.config.images_dir, image_file)
        image = cv2.imread(image_path)
        return image
    
    def create_ground_truth_mask(self, base_name: str, height: int, width: int) -> Optional[np.ndarray]:
        """Create ground truth mask from annotation file."""
        annotation_path = os.path.join(self.config.labels_dir, f"{base_name}.txt")
        gt_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not os.path.exists(annotation_path):
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
            color = self.config.color_map.get(cls, (255, 255, 255))
            cv2.fillPoly(gt_mask, [points], color)
        
        return gt_mask
    
    def save_masks(self, pred_mask: np.ndarray, base_name: str, gt_mask: Optional[np.ndarray] = None):
        """Save prediction mask and optionally ground truth mask to output directory."""
        # Always save prediction mask
        pred_path = os.path.join(self.config.output_dir, f"{base_name}_pred_mask.png")
        cv2.imwrite(pred_path, pred_mask)
        
        # Save ground truth only if it's provided and the config flag is set
        if self.config.ground_truth and gt_mask is not None:
            gt_path = os.path.join(self.config.output_dir, f"{base_name}_gt_mask.png")
            cv2.imwrite(gt_path, gt_mask)
    
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.config.temp_dir, ignore_errors=True)


# ------------------------------------------------------------------------
# Model Inference
# ------------------------------------------------------------------------

class YoloInference:
    """Class to handle YOLO model inference."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.model = YOLO(config.model_path)
    
    def process_tile(self, tile: np.ndarray, tile_path: str, x_offset: int, y_offset: int) -> np.ndarray:
        """Process a single tile and return mask with predictions."""
        mask = np.zeros_like(tile)
        
        # Save tile
        cv2.imwrite(tile_path, tile)
        
        # Run model inference
        results = self.model.predict(tile_path, conf=self.config.conf_threshold,verbose=False)
        
        # Add predictions to the mask
        if results[0].masks is not None:
            for seg_polygon, cls_idx in zip(results[0].masks.xy, results[0].boxes.cls):
                # Convert polygon coordinates to integer
                polygon = seg_polygon.astype(np.int32)
                
                # Fill polygon with class color
                color = self.config.color_map.get(int(cls_idx), (255, 255, 255))
                cv2.fillPoly(mask, [polygon], color)
        
        return mask
    
    def generate_prediction_mask(self, image: np.ndarray, base_name: str) -> np.ndarray:
        """Generate prediction mask by dividing image into tiles and processing each."""
        height, width = image.shape[:2]
        pred_mask = np.zeros((height, width, 3), dtype=np.uint8)
        tile_size = self.config.tile_size
        
        num_tiles_h = height // tile_size
        num_tiles_w = width // tile_size
        
        for row in range(num_tiles_h):
            for col in range(num_tiles_w):
                y1, y2 = row * tile_size, (row + 1) * tile_size
                x1, x2 = col * tile_size, (col + 1) * tile_size
                
                # Extract tile
                tile = image[y1:y2, x1:x2]
                tile_path = os.path.join(self.config.temp_dir, f"{base_name}_tile_{row}_{col}.png")
                
                # Process tile and get mask
                tile_mask = self.process_tile(tile, tile_path, x1, y1)
                
                # Copy tile predictions to the main mask
                pred_mask[y1:y2, x1:x2] = tile_mask
        
        return pred_mask


# ------------------------------------------------------------------------
# Postprocessing
# ------------------------------------------------------------------------

class Postprocessor:
    """Class for post-processing segmentation masks."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def color_to_class_id(self, color: Tuple[int, int, int]) -> int:
        """Convert a color in the mask to a class ID."""
        return self.config.color_map_inverse.get(tuple(color), 0)
    
    def apply_morphological_operations(self, pred_mask: np.ndarray, visualize: bool = False, 
                                      output_dir: Optional[str] = None) -> np.ndarray:
        """Apply morphological operations to improve mask quality."""
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
                self._visualize_processing_steps(stages, class_id, priority, class_result, output_dir)
        
        # Visualize final combined result if needed
        if visualize:
            self._visualize_final_result(pred_mask, processed_mask, output_dir)
        
        return processed_mask
    
    def _visualize_processing_steps(self, stages: Dict[str, np.ndarray], class_id: int, 
                                   priority: int, class_result: np.ndarray, output_dir: Optional[str] = None):
        """Visualize the processing steps for a class."""
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
    
    def _visualize_final_result(self, original_mask: np.ndarray, processed_mask: np.ndarray, 
                               output_dir: Optional[str] = None):
        """Visualize the final processed mask compared to the original."""
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
        plt.imshow(original_mask)
        plt.title('Original Mask')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed_mask)
        plt.title('Processed Mask')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'before_after_comparison.png'))
            plt.close()
        else:
            plt.show()
    
    def majority_voting_building_damage_mask(
        self,
        rgb_mask: np.ndarray,
        kernel_size: int = 3,
        dilate_iter: int = 1,
        return_rgb: bool = True
    ) -> np.ndarray:
        """
        Post‑process a mask so each building has exactly one damage class (majority vote).
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


# ------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------

class Visualizer:
    """Class to handle visualization of results."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def save_visualization(self, original: np.ndarray, pred_mask: np.ndarray, base_name: str, 
                          gt_mask: Optional[np.ndarray] = None):
        """Save visualization of original image, prediction, and optionally ground truth."""
        # Determine the layout based on whether we have ground truth
        if self.config.ground_truth and gt_mask is not None:
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
        else:
            plt.figure(figsize=(12, 6))
            
            # Original image
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            # Prediction mask
            plt.subplot(1, 2, 2)
            plt.title("Prediction Mask")
            plt.imshow(cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB))
            plt.axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(self.config.vis_dir, f"visualization_{base_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


# ------------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------------

class DamageSegmentationPipeline:
    """Main class to orchestrate the damage segmentation pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline with configuration."""
        if config_path and os.path.exists(config_path):
            self.config = Config.load_from_yaml(config_path)
        else:
            self.config = Config.default_config()
            
        # Create necessary directories
        self.config.create_directories()
        
        # Initialize components
        self.data_handler = DataHandler(self.config)
        self.model = YoloInference(self.config)
        self.postprocessor = Postprocessor(self.config)
        self.visualizer = Visualizer(self.config)
    
    def run(self):
        """Run the complete pipeline."""
        # Get all image files
        image_files = self.data_handler.list_image_files()
        
        total_start_time = time.time()
        print(f"Starting processing of {len(image_files)} images")
        
        for idx, image_file in enumerate(image_files):
            try:
                base_name = Path(image_file).stem
                print(f"\nProcessing image {idx+1}/{len(image_files)}: {base_name}")
                
                # Process individual image
                self.process_single_image(image_file, idx)
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
        
        # Cleanup temp files
        self.data_handler.cleanup()
        
        total_time = time.time() - total_start_time
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/len(image_files):.2f} seconds")
    
    def process_single_image(self, image_file: str, idx: int):
        """Process a single image through the pipeline."""
        base_name = Path(image_file).stem
        
        # Overall timing for this image
        image_start_time = time.time()
        
        # Load image
        load_start = time.time()
        image = self.data_handler.load_image(image_file)
        load_time = time.time() - load_start
        print(f"  Load image: {load_time:.2f} seconds")
        
        if image is None:
            print("  Failed to load image")
            return
        
        height, width = image.shape[:2]
        
        # Create ground truth mask if needed
        gt_mask = None
        if self.config.ground_truth:
            gt_start = time.time()
            gt_mask = self.data_handler.create_ground_truth_mask(base_name, height, width)
            gt_time = time.time() - gt_start
            print(f"  Create ground truth: {gt_time:.2f} seconds")
        
        # Generate prediction mask from tiles
        inference_start = time.time()
        raw_pred_mask = self.model.generate_prediction_mask(image, base_name)
        inference_time = time.time() - inference_start
        print(f"  Model inference: {inference_time:.2f} seconds")
        
        # Apply post-processing
        postproc_start = time.time()
        pred_mask = self.postprocessor.apply_morphological_operations(raw_pred_mask)
        postproc_time = time.time() - postproc_start
        print(f"  Post-processing: {postproc_time:.2f} seconds")
        
        # Alternative post-processing method
        # pred_mask = self.postprocessor.majority_voting_building_damage_mask(raw_pred_mask, 3, 3)
        
        # Save visualizations and masks
        should_save = not self.config.skip_save or idx % self.config.save_interval == 0
        
        if idx % self.config.save_interval == 0:
            vis_start = time.time()
            self.visualizer.save_visualization(image, pred_mask, base_name, gt_mask)
            vis_time = time.time() - vis_start
            print(f"  Visualization: {vis_time:.2f} seconds")
        
        if should_save:
            save_start = time.time()
            self.data_handler.save_masks(pred_mask, base_name, gt_mask)
            save_time = time.time() - save_start
            print(f"  Save masks: {save_time:.2f} seconds")
        
        # Total time for this image
        image_time = time.time() - image_start_time
        print(f"  Total time for image: {image_time:.2f} seconds")


# ------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run building damage segmentation pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--ground-truth', action='store_true', help='Enable ground truth processing')
    args = parser.parse_args()
    
    # Create and run the segmentation pipeline
    pipeline = DamageSegmentationPipeline(args.config)
    
    # Override ground truth setting if specified via command line
    if args.ground_truth:
        pipeline.config.ground_truth = True
        
    pipeline.run()

if __name__ == "__main__":
    main()
