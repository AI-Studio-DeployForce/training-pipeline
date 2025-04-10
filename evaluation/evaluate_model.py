"""
Building Damage Assessment Model Evaluation Script

This script evaluates a YOLO segmentation model on a dataset of building damage 
images, calculating precision, recall, and F1 scores across various confidence
thresholds.
"""

from evaluator import ModelEvaluator

def main():
    """
    Main function to run the evaluation
    """
    evaluator = ModelEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
