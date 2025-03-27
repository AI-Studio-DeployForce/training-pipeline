import os

# Path to the root folder containing train/, test/, valid/
BASE_DIR = "yolov9/data/dataset_256"

# Subfolders to process
SUBSETS = ["train", "test", "valid"]

def is_label_file_invalid(label_path):
    """
    Check if any non-empty line in the label file has fewer than 5 columns.
    In YOLO polygon format, each line should have at least 5 columns:
      class x0 y0 x1 y1
    Returns True if the file is invalid (i.e., has at least one short line).
    """
    if not os.path.exists(label_path):
        return False  # If it doesn't exist, treat it as not relevant
    if os.path.getsize(label_path) == 0:
        return False  # An empty file is not handled here (only lines with < 5 columns)
    
    with open(label_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                # blank line, ignore
                continue
            parts = ln.split()
            # If there's at least one line with fewer than 5 columns
            if len(parts) < 5:
                return True
    
    return False

def remove_file_if_exists(path):
    """Helper function to remove a file if it exists."""
    if os.path.exists(path):
        os.remove(path)

def delete_invalid_labels_inplace():
    for subset in SUBSETS:
        labels_dir = os.path.join(BASE_DIR, subset, "labels")
        images_dir = os.path.join(BASE_DIR, subset, "images")
        targets_dir = os.path.join(BASE_DIR, subset, "targets")

        if not os.path.isdir(labels_dir):
            print(f"[WARNING] Labels directory not found: {labels_dir}. Skipping {subset}.")
            continue

        # Get all .txt label files
        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]

        invalid_count = 0

        for label_file in label_files:
            lbl_path = os.path.join(labels_dir, label_file)
            if is_label_file_invalid(lbl_path):
                # Delete this label file, and the corresponding image/target
                invalid_count += 1
                remove_file_if_exists(lbl_path)

                base_name, _ = os.path.splitext(label_file)
                # Assuming images and targets use .png extension. Adjust if needed.
                img_path = os.path.join(images_dir, base_name + ".png")
                tgt_path = os.path.join(targets_dir, base_name + ".png")
                
                remove_file_if_exists(img_path)
                remove_file_if_exists(tgt_path)

        print(f"[INFO] {subset.upper()}: Removed {invalid_count} label files (with <5 columns).")

if __name__ == "__main__":
    delete_invalid_labels_inplace()
