import os

BASE_DIR = "yolov9/data/dataset_256"  # Adjust if needed
SUBSETS = ["train", "test", "valid"]

def fix_labels_to_closed_polygons():
    for subset in SUBSETS:
        labels_dir = os.path.join(BASE_DIR, subset, "labels")
        if not os.path.isdir(labels_dir):
            print(f"[WARNING] {labels_dir} does not exist. Skipping {subset}")
            continue
        
        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
        if not label_files:
            print(f"[INFO] No .txt files found in {labels_dir}")
            continue
        
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            
            # Read lines, skipping empty lines
            with open(label_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            new_lines = []
            changed_any_line = False
            
            for idx, line in enumerate(lines, start=1):
                parts = line.split()
                
                # We need at least 7 tokens: 
                #   1 for class + 6 for x0 y0 x1 y1 x2 y2 (i.e. 3 points)
                if len(parts) < 7:
                    print(f"[WARNING] Dropping line {idx} in '{label_file}' (not enough tokens). Line: '{line}'")
                    changed_any_line = True
                    continue
                
                # parts[0] = class ID, the rest are coordinates
                class_id = parts[0]
                coords = parts[1:]  # all the x,y pairs
                if len(coords) % 2 != 0:
                    print(f"[WARNING] Dropping line {idx} in '{label_file}' (odd number of coords). Line: '{line}'")
                    changed_any_line = True
                    continue
                
                # Convert coordinates to float
                floats = [float(x) for x in coords]
                n_points = len(floats) // 2  # number of (x,y) pairs
                
                if n_points < 3:
                    print(f"[WARNING] Dropping line {idx} in '{label_file}' (fewer than 3 points). Line: '{line}'")
                    changed_any_line = True
                    continue

                # Check if last point == first point
                x_first, y_first = floats[0], floats[1]
                x_last,  y_last  = floats[-2], floats[-1]
                
                # If they're not the same, append the first point to the end
                if not (x_first == x_last and y_first == y_last):
                    floats.extend([x_first, y_first])
                    changed_any_line = True
                
                # Reconstruct the line
                poly_str = " ".join(str(v) for v in floats)
                new_line = f"{class_id} {poly_str}"
                new_lines.append(new_line)
            
            # If we changed or removed any lines, overwrite the file
            if changed_any_line or len(new_lines) != len(lines):
                with open(label_path, "w") as f:
                    for nl in new_lines:
                        f.write(nl + "\n")

if __name__ == "__main__":
    fix_labels_to_closed_polygons()
