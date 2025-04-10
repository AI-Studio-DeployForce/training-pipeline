import os
import numpy as np
import cv2
from ultralytics import YOLO
from config import MODEL_PATH, TILE_SIZE, NUM_TILES, CLASS_COLORS, IOU_THRESHOLD

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