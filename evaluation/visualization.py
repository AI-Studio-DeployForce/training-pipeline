import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_STYLES, OUTPUT_PLOT_FILENAME, OUTPUT_PLOT_DPI

def plot_performance_curves(confidence_thresholds, class_metrics, aggregate_metrics):
    """
    Plot performance curves for precision, recall, and F1 score
    
    Args:
        confidence_thresholds: List of confidence thresholds
        class_metrics: Dictionary with per-class metrics
        aggregate_metrics: Dictionary with aggregate metrics
    """
    plt.figure(figsize=(18, 12))

    # Unpack metrics for convenience
    precisions = aggregate_metrics['precision']
    recalls = aggregate_metrics['recall']
    f1_scores = aggregate_metrics['f1']
    
    class_precisions = class_metrics['precision']
    class_recalls = class_metrics['recall']
    class_f1s = class_metrics['f1']

    # 1. Confidence vs Precision (per class)
    plt.subplot(2, 3, 1)
    for cls in range(4):
        plt.plot(confidence_thresholds, class_precisions[cls], 
                color=PLOT_STYLES[cls]['color'], 
                linestyle=PLOT_STYLES[cls]['linestyle'],
                marker=PLOT_STYLES[cls]['marker'],
                label=PLOT_STYLES[cls]['label'])
    plt.plot(confidence_thresholds, precisions, **PLOT_STYLES['avg'])
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Confidence vs Precision (per class)')
    plt.legend()
    plt.grid(True)

    # 2. Confidence vs Recall (per class)
    plt.subplot(2, 3, 2)
    for cls in range(4):
        plt.plot(confidence_thresholds, class_recalls[cls], 
                color=PLOT_STYLES[cls]['color'], 
                linestyle=PLOT_STYLES[cls]['linestyle'],
                marker=PLOT_STYLES[cls]['marker'],
                label=PLOT_STYLES[cls]['label'])
    plt.plot(confidence_thresholds, recalls, **PLOT_STYLES['avg'])
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.title('Confidence vs Recall (per class)')
    plt.legend()
    plt.grid(True)

    # 3. Confidence vs F1 (per class)
    plt.subplot(2, 3, 3)
    for cls in range(4):
        plt.plot(confidence_thresholds, class_f1s[cls], 
                color=PLOT_STYLES[cls]['color'], 
                linestyle=PLOT_STYLES[cls]['linestyle'],
                marker=PLOT_STYLES[cls]['marker'],
                label=PLOT_STYLES[cls]['label'])
    plt.plot(confidence_thresholds, f1_scores, **PLOT_STYLES['avg'])
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('Confidence vs F1 (per class)')
    plt.legend()
    plt.grid(True)

    # 4. Precision vs Recall (aggregate)
    plt.subplot(2, 3, 4)
    plt.plot(recalls, precisions, **PLOT_STYLES['avg'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall (aggregate)')
    plt.grid(True)

    # 5. Precision vs Recall (per class)
    plt.subplot(2, 3, 5)
    for cls in range(4):
        plt.plot(class_recalls[cls], class_precisions[cls], 
                color=PLOT_STYLES[cls]['color'], 
                linestyle=PLOT_STYLES[cls]['linestyle'],
                marker=PLOT_STYLES[cls]['marker'],
                label=PLOT_STYLES[cls]['label'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall (per class)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILENAME, dpi=OUTPUT_PLOT_DPI)
    plt.close()
    
def print_metrics(metrics, prefix=""):
    """
    Print metrics to the console
    
    Args:
        metrics: Dictionary with metrics
        prefix: Optional prefix string for the output
    """
    if not metrics:
        print(f"{prefix}No valid metrics available")
        return
        
    print(f"\n{prefix}Per-class results:")
    for cls in sorted(metrics['per_class'].keys()):
        values = metrics['per_class'][cls]
        class_name = f"Class {cls}"
        print(f"{prefix}{class_name}: Precision={values['precision']:.4f}, "
              f"Recall={values['recall']:.4f}, F1={values['f1']:.4f}, "
              f"Pixels={values['count']}")
    
    print(f"\n{prefix}Weighted Average: "
          f"Precision={metrics['weighted']['precision']:.4f}, "
          f"Recall={metrics['weighted']['recall']:.4f}, "
          f"F1={metrics['weighted']['f1']:.4f}")
    
    print(f"{prefix}Macro Average: "
          f"Precision={metrics['macro']['precision']:.4f}, "
          f"Recall={metrics['macro']['recall']:.4f}, "
          f"F1={metrics['macro']['f1']:.4f}")
    
def find_optimal_threshold(confidence_thresholds, f1_scores):
    """
    Find the optimal confidence threshold based on F1 scores
    
    Args:
        confidence_thresholds: List of confidence thresholds
        f1_scores: List of F1 scores
        
    Returns:
        The optimal threshold and index
    """
    optimal_idx = np.argmax(f1_scores)
    optimal_conf = confidence_thresholds[optimal_idx]
    return optimal_conf, optimal_idx 