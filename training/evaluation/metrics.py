import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import bgr_mask_to_labels, COLOR_ARRAYS

def calculate_metrics(gt_mask, pred_mask):
    """
    Calculate precision, recall, and F1 score for segmentation masks
    
    Args:
        gt_mask: Ground truth mask
        pred_mask: Prediction mask
        
    Returns:
        Dictionary with per-class and aggregate metrics
    """
    # Convert BGR masks to label masks
    gt_labels = bgr_mask_to_labels(gt_mask, COLOR_ARRAYS)
    pred_labels = bgr_mask_to_labels(pred_mask, COLOR_ARRAYS)

    # Only exclude invalid class 4, but keep and properly handle background (-1)
    valid = (gt_labels != 4) & (pred_labels != 4)
    
    # Get all pixels where ground truth has a damage label (0-3)
    gt_damage_pixels = (gt_labels >= 0) & (gt_labels < 4)
    
    # Get all pixels where prediction has a damage label (0-3)
    pred_damage_pixels = (pred_labels >= 0) & (pred_labels < 4)
    
    # Take valid pixels where either GT or prediction has a damage label
    valid_damage = valid & (gt_damage_pixels | pred_damage_pixels)
    
    y_true = gt_labels[valid_damage]
    y_pred = pred_labels[valid_damage]
    
    # Replace any background (-1) with a special class (999)
    y_true_adjusted = np.copy(y_true)
    y_pred_adjusted = np.copy(y_pred)
    
    y_true_adjusted[y_true == -1] = 999
    y_pred_adjusted[y_pred == -1] = 999

    # Initialize results dictionary
    metrics = {
        'per_class': {},
        'weighted': {'precision': 0, 'recall': 0, 'f1': 0},
        'macro': {'precision': 0, 'recall': 0, 'f1': 0}
    }

    if len(y_true_adjusted) > 0:
        # Get damage classes (0-3)
        damage_classes = sorted(set(range(4)))  # Always include all 4 damage classes
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_true_adjusted, y_pred_adjusted, 
                                             labels=damage_classes, 
                                             average=None, zero_division=0)
        recall_per_class = recall_score(y_true_adjusted, y_pred_adjusted, 
                                       labels=damage_classes, 
                                       average=None, zero_division=0)
        f1_per_class = f1_score(y_true_adjusted, y_pred_adjusted, 
                               labels=damage_classes, 
                               average=None, zero_division=0)
        
        # Store per-class metrics
        for i, cls in enumerate(damage_classes):
            # Count pixels in original ground truth
            pixel_count = int(np.sum(y_true == cls))
            
            metrics['per_class'][cls] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'count': pixel_count
            }
        
        # Calculate weighted average metrics
        metrics['weighted']['precision'] = float(precision_score(y_true_adjusted, y_pred_adjusted, 
                                                              labels=damage_classes, 
                                                              average='weighted', zero_division=0))
        metrics['weighted']['recall'] = float(recall_score(y_true_adjusted, y_pred_adjusted, 
                                                        labels=damage_classes, 
                                                        average='weighted', zero_division=0))
        metrics['weighted']['f1'] = float(f1_score(y_true_adjusted, y_pred_adjusted, 
                                                labels=damage_classes, 
                                                average='weighted', zero_division=0))
        
        # Calculate macro average (unweighted)
        metrics['macro']['precision'] = float(precision_score(y_true_adjusted, y_pred_adjusted, 
                                                          labels=damage_classes, 
                                                          average='macro', zero_division=0))
        metrics['macro']['recall'] = float(recall_score(y_true_adjusted, y_pred_adjusted, 
                                                    labels=damage_classes, 
                                                    average='macro', zero_division=0))
        metrics['macro']['f1'] = float(f1_score(y_true_adjusted, y_pred_adjusted, 
                                            labels=damage_classes, 
                                            average='macro', zero_division=0))
    
    return metrics

def aggregate_metrics(all_metrics):
    """
    Aggregate metrics from multiple images
    
    Args:
        all_metrics: List of metric dictionaries from individual images
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not all_metrics:
        return None
        
    # Initialize aggregated metrics
    aggregated_metrics = {
        'per_class': {},
        'weighted': {'precision': 0, 'recall': 0, 'f1': 0},
        'macro': {'precision': 0, 'recall': 0, 'f1': 0}
    }
    
    # Combine all class occurrences - ensure all damage classes are included
    all_classes = set(range(4))  # Classes 0-3
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
    
    return aggregated_metrics 