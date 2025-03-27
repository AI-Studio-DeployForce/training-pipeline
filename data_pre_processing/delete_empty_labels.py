import os
import random

# Path to the root folder containing train/, test/, valid/
BASE_DIR = "yolov9/data/dataset_256"

# Subfolders to process
SUBSETS = ["train", "test", "valid"]

# We keep 20% of the empty-labeled data and delete 80%
KEEP_RATIO = 0.2

# Set a seed for reproducibility if desired
random.seed(42)

def is_label_file_empty(label_path):
    """Check if a label file is effectively empty (0 bytes or only whitespace)."""
    if os.path.getsize(label_path) == 0:
        return True
    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f]
        # If all stripped lines are empty, treat file as empty
        return not any(lines)

def remove_file_if_exists(path):
    """Helper to remove a file if it exists."""
    if os.path.exists(path):
        os.remove(path)

def main():
    for subset in SUBSETS:
        labels_dir = os.path.join(BASE_DIR, subset, "labels")
        images_dir = os.path.join(BASE_DIR, subset, "images")
        targets_dir = os.path.join(BASE_DIR, subset, "targets")

        if not os.path.isdir(labels_dir):
            print(f"[WARNING] Labels directory not found: {labels_dir}. Skipping {subset}.")
            continue

        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
        
        # Identify which label files are empty
        empty_label_files = []
        for lbl in label_files:
            lbl_path = os.path.join(labels_dir, lbl)
            if is_label_file_empty(lbl_path):
                empty_label_files.append(lbl)

        total_empty = len(empty_label_files)
        if total_empty == 0:
            print(f"[INFO] No empty labels found in {subset}.")
            continue

        # Shuffle to pick a random 20% to keep
        random.shuffle(empty_label_files)
        keep_count = int(total_empty * KEEP_RATIO)
        # These are the label files we keep
        keep_files = empty_label_files[:keep_count]
        # These are the label files we delete
        delete_files = empty_label_files[keep_count:]

        print(f"[INFO] {subset.upper()} - Empty label files found: {total_empty}")
        print(f"        Keeping {keep_count}, Deleting {len(delete_files)}")

        # Remove the "delete_files" and their corresponding images/targets
        for lbl_file in delete_files:
            # Remove the label
            lbl_path = os.path.join(labels_dir, lbl_file)
            remove_file_if_exists(lbl_path)

            # Derive the base name (without extension) to find corresponding image/target
            base_name, _ = os.path.splitext(lbl_file)
            
            # Typical assumption: images/targets have the same base name + .png
            # If your images have .jpg or something else, adjust below as needed.
            img_path = os.path.join(images_dir, base_name + ".png")
            remove_file_if_exists(img_path)

            tgt_path = os.path.join(targets_dir, base_name + ".png")
            remove_file_if_exists(tgt_path)

        print(f"[INFO] Completed deletions for {subset}.\n")

if __name__ == "__main__":
    main()
