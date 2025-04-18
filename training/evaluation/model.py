import os
import numpy as np
import cv2
from ultralytics import YOLO
from config import MODEL_PATH, TILE_SIZE, NUM_TILES, CLASS_COLORS, IOU_THRESHOLD
import matplotlib.pyplot as plt
class SegmentationModel:
    """
    Class to handle a YOLO segmentation model for building damage assessment
    """
    
    def __init__(self, model_path=MODEL_PATH):
        """
        Initialize the model
        
        Args:
            model_path: Path to the model weights file
        """
        self.model = YOLO(model_path)
    
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
            CLASS_COLORS[0]: 0,    
            CLASS_COLORS[1]: 1,    
            CLASS_COLORS[2]: 2,    
            CLASS_COLORS[3]: 3
        }
        
        return class_colors.get(tuple(color), 0)  # Default to background (0) if color not found

    def post_process_prediction(self, pred_mask, visualize=False, output_dir=None):
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

    def predict_tiles(self, image, confidence_threshold):
        """
        Process a large image by dividing it into tiles
        
        Args:
            image: Input image (1024x1024 expected)
            confidence_threshold: Confidence threshold for predictions
            
        Returns:
            Prediction mask as a numpy array
        """
        H, W, _ = image.shape
        pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
        
        for row in range(NUM_TILES):
            for col in range(NUM_TILES):
                y1 = row * TILE_SIZE
                y2 = (row + 1) * TILE_SIZE
                x1 = col * TILE_SIZE
                x2 = (col + 1) * TILE_SIZE
                
                # Process single tile
                tile = image[y1:y2, x1:x2]
                results = self.model.predict(
                    tile, 
                    conf=confidence_threshold,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )
                
                if results[0].masks is not None:
                    for seg_polygon, cls_idx in zip(results[0].masks.xy, results[0].boxes.cls):
                        offset_polygon = seg_polygon + [x1, y1]
                        offset_polygon = offset_polygon.astype(np.int32)
                        color = CLASS_COLORS.get(int(cls_idx), (255, 255, 255))
                        cv2.fillPoly(pred_mask, [offset_polygon], color)
        pred_mask = self.post_process_prediction(pred_mask)
        return pred_mask
        
    def predict_single_tile(self, tile, confidence_threshold):
        """
        Process a single tile
        
        Args:
            tile: Input tile
            confidence_threshold: Confidence threshold for predictions
            
        Returns:
            Result object from YOLO model
        """
        results = self.model.predict(
            tile, 
            conf=confidence_threshold,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        return results[0] 