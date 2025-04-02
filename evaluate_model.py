import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------------------------------------------------------------------
# CONFIG: Flag to control image saving behavior
# ------------------------------------------------------------------------
skip_save = True  # When True, only saves visualization images every 50 iterations

# ------------------------------------------------------------------------
# CONFIG: Folder paths for big 1024x1024 images and their .txt labels
# ------------------------------------------------------------------------
images_dir = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/datasets/original_data_yolo/pre/test/images"  # Folder with 1024x1024 .png images
labels_dir = "/home/diego/Documents/master/S4/AI_studio/training-pipeline/datasets/original_data_yolo/pre/test/labels"  # Folder with corresponding .txt labels

# Temporary folder for 256x256 tiles
temp_dir = "temp_tiles"
os.makedirs(temp_dir, exist_ok=True)

# Folder where we will save final masks
predictions_dir = "predictions_final_256_new_pre"
os.makedirs(predictions_dir, exist_ok=True)

# ------------------------------------------------------------------------
# CONFIG: Load segmentation model (must be YOLO-seg)
# ------------------------------------------------------------------------
model = YOLO("best_256_new.pt")  # e.g. YOLOv9-seg model

# ------------------------------------------------------------------------
# COLOR MAPPING (BGR) for each class
# ------------------------------------------------------------------------
fixed_colors = {
    0: (0, 255, 0),    # Green (No damage)
    1: (0, 255, 255),  # Yellow (Minor damage)
    2: (0, 165, 255),  # Orange (Major damage)
    3: (0, 0, 255)     # Red (Destroyed)
}

# Inverse mapping: BGR -> class index
# We also allow (255,255,255) for "unknown" in case you used that for unmatched polygons.
color_map_inverse = {
    (0, 255, 0): 0,
    (0, 255, 255): 1,
    (0, 165, 255): 2,
    (0, 0, 255): 3,
    (255, 255, 255): 4  # We'll treat this as "ignore" (unknown)
}

# ------------------------------------------------------------------------
# Helper: Convert a BGR mask to a 2D label image
# ------------------------------------------------------------------------
def bgr_mask_to_labels(mask, color_inv):
    """
    mask: HxWx3 (BGR)
    color_inv: dict {(B,G,R): class_index, ...}
    returns: HxW np.int32 array with class indices or -1 if not recognized
    """
    H, W, _ = mask.shape
    label_img = np.full((H, W), -1, dtype=np.int32)

    # Pixel-by-pixel color matching (straightforward but can be slow)
    for r in range(H):
        for c in range(W):
            bgr = tuple(mask[r, c])  # (B,G,R)
            if bgr in color_inv:
                label_img[r, c] = color_inv[bgr]
            else:
                label_img[r, c] = -1  # Not recognized

    return label_img

# ------------------------------------------------------------------------
# Main loop: Process every .png in images_dir
# ------------------------------------------------------------------------
all_image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])

# We will keep track of global metrics across images if you want an average
global_precisions = []
global_recalls = []
global_f1s = []

for idx, image_file in enumerate(all_image_files):
    big_image_path = os.path.join(images_dir, image_file)
    base_name = os.path.splitext(image_file)[0]
    annotation_path = os.path.join(labels_dir, base_name + ".txt")

    print(f"Processing image {idx+1}/{len(all_image_files)}: {base_name}")

    # -------------------------------
    # 1) Load the 1024x1024 image
    # -------------------------------
    big_image = cv2.imread(big_image_path)
    if big_image is None:
        print(f"Warning: Could not read image file: {big_image_path}")
        continue

    H, W, _ = big_image.shape
    if (H != 1024) or (W != 1024):
        print(f"Warning: {base_name} is {W}x{H}, expected 1024x1024.")
        # You can decide to continue or skip

    # -------------------------------
    # 2) Build the Ground-Truth mask
    # -------------------------------
    gt_mask = np.zeros((H, W, 3), dtype=np.uint8)

    if not os.path.exists(annotation_path):
        print(f"Warning: No annotation file for {base_name}. Skipping.")
        continue

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))

        points = []
        for i in range(0, len(coords), 2):
            x_norm = coords[i]
            y_norm = coords[i + 1]
            x_pix = int(x_norm * W)
            y_pix = int(y_norm * H)
            points.append((x_pix, y_pix))

        points = np.array(points, dtype=np.int32)
        color = fixed_colors.get(cls, (255, 255, 255))  # default to white
        cv2.fillPoly(gt_mask, [points], color)

    # -------------------------------
    # 3) Prepare an empty Pred. mask
    # -------------------------------
    pred_mask = np.zeros((H, W, 3), dtype=np.uint8)

    # -------------------------------
    # 4) Split image into 256×256 tiles, run inference
    # -------------------------------
    tile_size = 256
    num_tiles = 4  # 1024 / 256

    for row in range(num_tiles):
        for col in range(num_tiles):
            y1 = row * tile_size
            y2 = (row + 1) * tile_size
            x1 = col * tile_size
            x2 = (col + 1) * tile_size

            tile = big_image[y1:y2, x1:x2]
            tile_path = os.path.join(temp_dir, f"{base_name}_tile_{row}_{col}.png")
            cv2.imwrite(tile_path, tile)

            # Run inference on the tile
            results = model.predict(tile_path, conf=0.2, iou=0.6)

            # Merge tile predictions back into the big pred_mask
            if results[0].masks is not None:
                for seg_polygon, cls_idx in zip(results[0].masks.xy, results[0].boxes.cls):
                    offset_polygon = seg_polygon + [x1, y1]
                    offset_polygon = offset_polygon.astype(np.int32)

                    color = fixed_colors.get(int(cls_idx), (255, 255, 255))
                    cv2.fillPoly(pred_mask, [offset_polygon], color)
            else:
                print(f"No predicted masks found for tile ({row}, {col}).")

    # -------------------------------
    # 5) Compute pixel-level metrics
    # -------------------------------
    # Convert GT and Pred masks to label arrays
    gt_labels = bgr_mask_to_labels(gt_mask, color_map_inverse)
    pred_labels = bgr_mask_to_labels(pred_mask, color_map_inverse)

    # Filter out any invalid (-1) or "unknown" (class 4) pixels
    valid = (
        (gt_labels != -1) & (pred_labels != -1) &
        (gt_labels != 4) & (pred_labels != 4)
    )
    y_true = gt_labels[valid]
    y_pred = pred_labels[valid]

    # If everything is valid, compute metrics (macro average across classes)
    if len(y_true) > 0:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        # If we have no valid pixels, skip
        precision, recall, f1 = 0, 0, 0

    global_precisions.append(precision)
    global_recalls.append(recall)
    global_f1s.append(f1)

    print(f"  -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # -------------------------------
    # 6) (Optional) Visualization
    #    Only show every 50th image
    # -------------------------------
    if idx % 50 == 0:
        # Create visualization directory if it doesn't exist
        vis_dir = "visualization_results"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save the visualization
        plt.figure(figsize=(18, 6))

        # Original big image
        plt.subplot(1, 3, 1)
        plt.title("Big Original Image")
        plt.imshow(cv2.cvtColor(big_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        # Ground Truth mask
        plt.subplot(1, 3, 2)
        plt.title("Big Ground Truth Mask")
        plt.imshow(cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        # Prediction mask
        plt.subplot(1, 3, 3)
        plt.title("Big Prediction Mask")
        plt.imshow(cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(vis_dir, f"visualization_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory
        
        print(f"Saved visualization to {save_path}")

    # -------------------------------
    # 7) Save final masks to predictions_final folder
    # -------------------------------
    if not skip_save or idx % 50 == 0:
        gt_outfile = os.path.join(predictions_dir, f"{base_name}_gt_mask.png")
        pred_outfile = os.path.join(predictions_dir, f"{base_name}_pred_mask.png")
        cv2.imwrite(gt_outfile, gt_mask)
        cv2.imwrite(pred_outfile, pred_mask)

# ------------------------------------------------------------------------
# After processing all images, remove temp folder
# ------------------------------------------------------------------------
shutil.rmtree(temp_dir, ignore_errors=True)
print(f"\nDeleted temporary folder: {temp_dir}")

# ------------------------------------------------------------------------
# (Optional) Show average metrics across all images
# ------------------------------------------------------------------------
if len(global_precisions) > 0:
    avg_precision = np.mean(global_precisions)
    avg_recall = np.mean(global_recalls)
    avg_f1 = np.mean(global_f1s)
    print("\n=== Overall Dataset Metrics (Macro-Average) ===")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")
else:
    print("\nNo images were successfully processed. No metrics to report.")
