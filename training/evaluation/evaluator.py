import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import gc

from .config import IMAGES_DIR, LABELS_DIR, CONFIDENCE_THRESHOLDS, NUM_WORKERS
from .utils import load_image, load_labels, create_gt_mask
from .model import SegmentationModel
from .metrics import calculate_metrics, aggregate_metrics
from .visualization import plot_performance_curves, print_metrics, find_optimal_threshold

class ModelEvaluator:
    """
    Class to evaluate a segmentation model on a dataset
    """
    def __init__(self, model_path: str = None):
        # forward the path into SegmentationModel
        self.model = SegmentationModel(model_path)
        
    def process_single_image(self, args):
        """
        Process a single image with a specific confidence threshold
        
        Args:
            args: Tuple of (image_file, confidence_threshold)
            
        Returns:
            Dictionary with metrics or None if processing failed
        """
        image_file, confidence = args
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(IMAGES_DIR, image_file)
        label_path = os.path.join(LABELS_DIR, base_name + ".txt")

        # Load image
        image = load_image(image_path)
        if image is None:
            return None

        H, W, _ = image.shape
        if (H != 1024) or (W != 1024):
            return None

        # Load labels and create ground truth mask
        labels = load_labels(label_path, W, H)
        if not labels:
            return None
            
        gt_mask = create_gt_mask(labels, W, H)

        # Generate prediction mask
        pred_mask = self.model.predict_tiles(image, confidence)
        
        # Calculate metrics
        metrics = calculate_metrics(gt_mask, pred_mask)
        
        # Clean up to avoid memory leaks
        del image
        del gt_mask
        del pred_mask
        gc.collect()
        
        return metrics
        
    def evaluate(self):
        """
        Evaluate the model on the dataset across different confidence thresholds
        
        Returns:
            Dictionary with evaluation results
        """
        # Get list of image files
        all_image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])
        
        # Initialize result storage
        results = {
            'class_metrics': {
                'precision': {i: [] for i in range(4)},
                'recall': {i: [] for i in range(4)},
                'f1': {i: [] for i in range(4)}
            },
            'aggregate_metrics': {
                'precision': [],
                'recall': [],
                'f1': []
            },
            'thresholds': CONFIDENCE_THRESHOLDS,
            'optimal': {
                'threshold': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
        }

        # Main progress bar for confidence thresholds
        for conf in tqdm(CONFIDENCE_THRESHOLDS, desc="Testing confidence thresholds"):
            print(f"\nTesting confidence threshold: {conf:.2f}")
            
            # Process images in parallel with limited workers
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Create a list of futures for progress tracking
                futures = [executor.submit(self.process_single_image, (image_file, conf)) 
                          for image_file in all_image_files]
                
                # Process results with progress bar
                all_metrics = []
                for future in tqdm(futures, desc="Processing images", leave=False):
                    result = future.result()
                    if result is not None:
                        all_metrics.append(result)
            
            # Skip if no valid results
            if not all_metrics:
                # Add zeros for all classes and aggregate
                for cls in range(4):
                    results['class_metrics']['precision'][cls].append(0)
                    results['class_metrics']['recall'][cls].append(0)
                    results['class_metrics']['f1'][cls].append(0)
                
                results['aggregate_metrics']['precision'].append(0)
                results['aggregate_metrics']['recall'].append(0)
                results['aggregate_metrics']['f1'].append(0)
                continue
                
            # Aggregate metrics from all images
            aggregated = aggregate_metrics(all_metrics)
            
            # Store per-class metrics
            for cls in range(4):
                if cls in aggregated['per_class']:
                    results['class_metrics']['precision'][cls].append(aggregated['per_class'][cls]['precision'])
                    results['class_metrics']['recall'][cls].append(aggregated['per_class'][cls]['recall'])
                    results['class_metrics']['f1'][cls].append(aggregated['per_class'][cls]['f1'])
                else:
                    # Class not present in this batch
                    results['class_metrics']['precision'][cls].append(0)
                    results['class_metrics']['recall'][cls].append(0)
                    results['class_metrics']['f1'][cls].append(0)
            
            # Store aggregate metrics
            results['aggregate_metrics']['precision'].append(aggregated['weighted']['precision'])
            results['aggregate_metrics']['recall'].append(aggregated['weighted']['recall'])
            results['aggregate_metrics']['f1'].append(aggregated['weighted']['f1'])
            
            # Print results for this threshold
            print_metrics(aggregated)
            
            # Clean up
            del all_metrics
            del futures
            gc.collect()
        
        # Find optimal threshold
        optimal_threshold, optimal_idx = find_optimal_threshold(
            CONFIDENCE_THRESHOLDS, 
            results['aggregate_metrics']['f1']
        )
        
        # Store optimal results
        results['optimal']['threshold'] = optimal_threshold
        results['optimal']['precision'] = results['aggregate_metrics']['precision'][optimal_idx]
        results['optimal']['recall'] = results['aggregate_metrics']['recall'][optimal_idx]
        results['optimal']['f1'] = results['aggregate_metrics']['f1'][optimal_idx]
        
        print(f"\nOptimal confidence threshold: {optimal_threshold:.2f}")
        print(f"At this threshold - Precision: {results['optimal']['precision']:.4f}, "
              f"Recall: {results['optimal']['recall']:.4f}, "
              f"F1: {results['optimal']['f1']:.4f}")
        
        # Plot curves
        plot_performance_curves(
            CONFIDENCE_THRESHOLDS,
            results['class_metrics'],
            results['aggregate_metrics']
        )
        
        return results 