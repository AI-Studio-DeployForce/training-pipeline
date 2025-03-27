import os

# Adjust these paths as needed
BASE_DIR = "yolov9/data/dataset_256"
SUBSETS = ["train", "test", "valid"]

def count_empty_and_nonempty_labels():
    for subset in SUBSETS:
        label_dir = os.path.join(BASE_DIR, subset, "labels")
        if not os.path.exists(label_dir):
            print(f"[WARNING] {label_dir} does not exist! Skipping...")
            continue
        
        all_label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
        
        empty_count = 0
        nonempty_count = 0
        
        for label_file in all_label_files:
            label_path = os.path.join(label_dir, label_file)
            
            # Check if the file is empty by file size or by content
            if os.path.getsize(label_path) == 0:
                empty_count += 1
            else:
                # Alternatively: read lines to ensure not just whitespace
                with open(label_path, 'r') as f:
                    lines = [ln.strip() for ln in f]
                    # If after stripping, no lines have content, treat as empty
                    if not any(lines):
                        empty_count += 1
                    else:
                        nonempty_count += 1
        
        total_files = len(all_label_files)
        print(f"--- {subset.upper()} ---")
        print(f"Total label files: {total_files}")
        print(f"Empty label files: {empty_count}")
        print(f"Non-empty label files: {nonempty_count}")
        print()

if __name__ == "__main__":
    count_empty_and_nonempty_labels()
