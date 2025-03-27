import os
import shutil
import time
import json
from datetime import datetime

def parse_polygon(wkt_str):
    """
    Parse a WKT polygon string.
    Expected format: "POLYGON ((x1 y1, x2 y2, ..., xn yn))"
    Returns a list of (x, y) tuples.
    """
    if not wkt_str.startswith("POLYGON ((") or not wkt_str.endswith("))"):
        raise ValueError("Invalid WKT format")
    # Remove the "POLYGON ((" prefix and the "))" suffix.
    coords_str = wkt_str[len("POLYGON (("):-2]
    coords = []
    # Split coordinate pairs by comma and then by whitespace.
    for pair in coords_str.split(","):
        x_str, y_str = pair.strip().split()
        coords.append((float(x_str), float(y_str)))
    return coords

def convert_json_to_yolo(json_file, default_width=1024, default_height=1024, class_map=None, default_class_id=0):
    """
    Convert a JSON label file to YOLO segmentation format.
    
    For post-disaster JSON, the properties include a "subtype" field and are mapped
    using the provided class_map. For pre-disaster JSON (which only has building segments),
    the default_class_id is used for every instance.
    
    Returns a list of strings in the following YOLO format:
       <class_id> x1 y1 x2 y2 ... xn yn
    where the coordinates are normalized (divided by image width and height).
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Retrieve image dimensions from metadata, or use defaults.
    image_width = data.get("metadata", {}).get("width", default_width)
    image_height = data.get("metadata", {}).get("height", default_height)
    
    yolo_lines = []
    # Process the "xy" features (pixel coordinates).
    features = data.get("features", {}).get("xy", [])
    for feature in features:
        properties = feature.get("properties", {})
        # For post-disaster JSON, use the subtype with class mapping.
        if "subtype" in properties:
            if class_map is None:
                class_map = {
                    "no-damage": 0,
                    "minor-damage": 1,
                    "major-damage": 2,
                    "destroyed": 3
                }
            subtype = properties.get("subtype", "unknown")
            class_id = class_map.get(subtype, -1)
            if class_id == -1:
                # Skip features with an undefined subtype.
                continue
        else:
            # Pre-disaster JSON: assign default class id (e.g., 0 for all buildings)
            class_id = default_class_id
        
        wkt = feature.get("wkt", "")
        try:
            coords = parse_polygon(wkt)
        except ValueError as e:
            print(f"Error parsing polygon in file {json_file}: {e}")
            continue
        
        # Normalize the polygon coordinates.
        norm_coords = []
        for (x, y) in coords:
            norm_x = x / image_width
            norm_y = y / image_height
            norm_coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
        
        # YOLO segmentation format: <class_id> x1 y1 x2 y2 ... xn yn
        line = f"{class_id} " + " ".join(norm_coords)
        yolo_lines.append(line)
    
    return yolo_lines

def process_directory(split, src_root, dst_root, default_class_id=0):
    """
    Process one dataset split (train/valid/test) and convert its JSON labels to YOLO segmentation format.
    
    The script now determines the destination folder based on the file name:
      - If the file name contains "post" then the converted labels are saved to:
            <dst_root>/post/<split>/labels/
      - If the file name contains "pre" then the labels are saved to:
            <dst_root>/pre/<split>/labels/
      - If neither is specified, it defaults to the "pre" folder.
    
    Note: If the split is "valid", the output folder is named "valid".
    """
    src_labels_dir = os.path.join(src_root, split, "labels")
    out_split = "valid" if split == "valid" else split
    dst_labels_dir_post = os.path.join(dst_root, "post", out_split, "labels")
    dst_labels_dir_pre = os.path.join(dst_root, "pre", out_split, "labels")
    os.makedirs(dst_labels_dir_post, exist_ok=True)
    os.makedirs(dst_labels_dir_pre, exist_ok=True)
    
    # Define the class mapping for post-disaster JSON labels.
    class_map = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3
    }
    
    # Iterate over each JSON file in the source labels directory.
    for filename in os.listdir(src_labels_dir):
        if filename.endswith(".json"):
            src_file = os.path.join(src_labels_dir, filename)
            # Convert the JSON file to YOLO segmentation format.
            yolo_lines = convert_json_to_yolo(src_file, class_map=class_map, default_class_id=default_class_id)
            
            # Determine destination folder based on file naming.
            lower_filename = filename.lower()
            if "post" in lower_filename:
                dst_file = os.path.join(dst_labels_dir_post, filename.replace(".json", ".txt"))
            elif "pre" in lower_filename:
                dst_file = os.path.join(dst_labels_dir_pre, filename.replace(".json", ".txt"))
            else:
                # Default behavior: save in "pre" folder if naming doesn't specify.
                dst_file = os.path.join(dst_labels_dir_pre, filename.replace(".json", ".txt"))
            
            # Write the YOLO-formatted labels with a .txt extension.
            with open(dst_file, 'w') as out_f:
                for line in yolo_lines:
                    out_f.write(line + "\n")
    print(f"Processed {split} directory:")
    print(f"  Post-disaster labels: {dst_labels_dir_post}")
    print(f"  Pre-disaster labels: {dst_labels_dir_pre}")

def print_step(step_number, step_description):
    """Print a formatted step header."""
    print("\n" + "="*80)
    print(f"STEP {step_number}: {step_description}")
    print("="*80)

def run_copy_data(post_folder, data_folder):
    """
    Step 1: Copy data to YOLOv9 format (only use post).
    Creates 2 folders: images and targets.
    """
    print_step(1, "Copying data to YOLOv9 format")
    
    # Define the mapping for folder names:
    subset_mapping = {
        "train": "train",
        "valid": "valid",
        "test": "test"
    }

    # Process each subset folder in post
    for subset_post, subset_data in subset_mapping.items():
        # Paths in post folder
        post_labels_folder = os.path.join(post_folder, subset_post, "labels")
        dest_images_folder = os.path.join(post_folder, subset_post, "images")
        dest_targets_folder = os.path.join(post_folder, subset_post, "targets")

        # Create destination folders if they do not exist
        os.makedirs(dest_images_folder, exist_ok=True)
        os.makedirs(dest_targets_folder, exist_ok=True)

        # Paths in data folder
        data_images_folder = os.path.join(data_folder, subset_data, "images")
        data_targets_folder = os.path.join(data_folder, subset_data, "targets")

        print(f"  Processing subset: {subset_post} (data folder: {subset_data})")
        
        # Loop through each label file in the post labels folder
        for label_file in os.listdir(post_labels_folder):
            if label_file.endswith(".txt"):
                # Get base file name (without extension)
                base_name = os.path.splitext(label_file)[0]

                # Build the expected file names in data folder
                image_filename = base_name + ".png"
                target_filename = base_name + "_target.png"

                # Build the full source paths
                src_image_path = os.path.join(data_images_folder, image_filename)
                src_target_path = os.path.join(data_targets_folder, target_filename)

                # Build the full destination paths
                dest_image_path = os.path.join(dest_images_folder, image_filename)
                dest_target_path = os.path.join(dest_targets_folder, target_filename)

                # Copy the image file if it exists
                if os.path.exists(src_image_path):
                    shutil.copy(src_image_path, dest_image_path)
                else:
                    print(f"  Image file not found: {src_image_path}")

                # Copy the target file if it exists
                if os.path.exists(src_target_path):
                    shutil.copy(src_target_path, dest_target_path)
                else:
                    print(f"  Target file not found: {src_target_path}")
    
    print(f"Step 1 completed: Data copied to {post_folder}")
    return post_folder

def run_subsample_images(post_folder):
    """
    Step 2: Run subsample_images.py.
    Delete 40% of the dataset.
    """
    print_step(2, "Subsampling images (deleting 40% of the dataset)")
    
    import random
    random.seed(42)  # For reproducibility
    
    # List of subsets to process
    subsets = ["train", "valid", "test"]

    # Loop through each subset folder
    for subset in subsets:
        # Define the full paths to the subfolders for images, targets, and labels
        images_folder = os.path.join(post_folder, subset, "images")
        targets_folder = os.path.join(post_folder, subset, "targets")
        labels_folder = os.path.join(post_folder, subset, "labels")
        
        # List all .png files in the images folder
        image_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]
        total_images = len(image_files)
        
        # Calculate 40% of the total images
        delete_count = int(total_images * 0.4)
        print(f"  Subset '{subset}': Deleting {delete_count} out of {total_images} images.")
        
        # Randomly select files to delete
        files_to_delete = random.sample(image_files, delete_count)
        
        # Process each selected file
        for image_file in files_to_delete:
            # Derive the base file name (without extension)
            base_name = os.path.splitext(image_file)[0]
            
            # Construct full paths for the image, corresponding target, and label files
            image_path = os.path.join(images_folder, image_file)
            target_file = base_name + "_target.png"
            target_path = os.path.join(targets_folder, target_file)
            label_file = base_name + ".txt"
            label_path = os.path.join(labels_folder, label_file)
            
            # Delete the files if they exist
            if os.path.exists(image_path):
                os.remove(image_path)
            
            if os.path.exists(target_path):
                os.remove(target_path)
            
            if os.path.exists(label_path):
                os.remove(label_path)

    print(f"Step 2 completed: Dataset subsampled in {post_folder}")
    return post_folder

def run_image_windowing(post_folder, window_size=512):
    """
    Step 3: Run image_windowing.py.
    Slice images and targets into smaller windows.
    """
    print_step(3, f"Executing image windowing (size: {window_size}×{window_size})")
    
    import cv2
    import numpy as np
    
    # Define destination directory for windowed images
    dest_dir = post_folder + f"_{window_size}"
    subsets = ["train", "test", "valid"]
    
    orig_size = 1024  # original image size
    num_tiles = orig_size // window_size

    # Ensure destination directories exist
    for subset in subsets:
        os.makedirs(os.path.join(dest_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, subset, "targets"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, subset, "labels"), exist_ok=True)
    
    def convert_contours_to_yolo_format(contours, img_w, img_h):
        """Convert OpenCV contours to YOLO segmentation polygon format."""
        polygons = []
        for cnt in contours:
            # Flatten contour to Nx2
            cnt = cnt.reshape(-1, 2)
            
            # Convert each point to normalized coordinates in [0..1]
            poly = []
            for (x, y) in cnt:
                nx = x / img_w
                ny = y / img_h
                poly.append(nx)
                poly.append(ny)

            # Explicitly close the polygon:
            if len(cnt) > 0:
                # Repeat the first point at the end
                first_x, first_y = cnt[0]  # original coordinates
                poly.append(first_x / img_w)
                poly.append(first_y / img_h)
            
            polygons.append(poly)
        return polygons

    def process_mask_to_yolo_txt(mask_subimg, out_txt_path, tile_width, tile_height):
        """Find contours and convert to YOLO format."""
        # Dictionary that maps actual pixel value to YOLO class
        class_map = {1: 0, 2: 1, 3: 2, 4: 3}
        
        lines = []
        for pixel_val, yolo_class in class_map.items():
            # Create a binary mask for the given pixel value
            bin_mask = np.where(mask_subimg == pixel_val, 255, 0).astype(np.uint8)
            
            # Find contours in the binarized mask
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue  # No polygons for this class in this sub-tile
            
            # Convert each contour to YOLO polygon format
            polygons = convert_contours_to_yolo_format(contours, tile_width, tile_height)
            
            # Build lines: "class x0 y0 x1 y1 x2 y2 ..."
            for poly in polygons:
                poly_str = " ".join(str(round(p, 6)) for p in poly)
                line_str = f"{yolo_class} {poly_str}"
                lines.append(line_str)
        
        if len(lines) == 0:
            # If we found no objects at all, just create an empty .txt
            with open(out_txt_path, "w") as f:
                pass
        else:
            with open(out_txt_path, "w") as f:
                for line in lines:
                    f.write(line + "\n")
    
    # Process each subset
    for subset in subsets:
        print(f"  Processing subset: {subset}")
        images_dir = os.path.join(post_folder, subset, "images")
        targets_dir = os.path.join(post_folder, subset, "targets")
        
        out_images_dir = os.path.join(dest_dir, subset, "images")
        out_targets_dir = os.path.join(dest_dir, subset, "targets")
        out_labels_dir = os.path.join(dest_dir, subset, "labels")
        
        # List all image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)
        processed = 0
        
        for img_name in image_files:
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{total_images} images in {subset}")
                
            # Build full path
            img_path = os.path.join(images_dir, img_name)
            
            # Corresponding target name
            base_name, _ = os.path.splitext(img_name)
            target_name = base_name + "_target.png"
            target_path = os.path.join(targets_dir, target_name)
            
            # Read image and target
            img = cv2.imread(img_path)  # BGR
            if img is None:
                print(f"  Could not read image: {img_path}")
                continue
            
            target_mask = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
            if target_mask is None:
                print(f"  Could not read mask: {target_path}")
                continue
            
            # For each tile in the grid
            for i in range(num_tiles):
                for j in range(num_tiles):
                    # Calculate the slice boundaries
                    x_start = j * window_size
                    x_end = x_start + window_size
                    y_start = i * window_size
                    y_end = y_start + window_size
                    
                    # Slice the image and target
                    sub_img = img[y_start:y_end, x_start:x_end]
                    sub_mask = target_mask[y_start:y_end, x_start:x_end]
                    
                    # Create output file names
                    sub_img_name = f"{base_name}_{i}_{j}.png"
                    sub_target_name = f"{base_name}_{i}_{j}.png"
                    sub_label_name = f"{base_name}_{i}_{j}.txt"
                    
                    # Write out the sub image and target
                    out_img_path = os.path.join(out_images_dir, sub_img_name)
                    cv2.imwrite(out_img_path, sub_img)
                    
                    out_target_path = os.path.join(out_targets_dir, sub_target_name)
                    cv2.imwrite(out_target_path, sub_mask)
                    
                    # Generate new label from sub_mask
                    out_label_path = os.path.join(out_labels_dir, sub_label_name)
                    process_mask_to_yolo_txt(sub_mask, out_label_path, window_size, window_size)
    
    print(f"Step 3 completed: Images windowed and saved to {dest_dir}")
    return dest_dir

def run_analyse_dataset(base_dir):
    """
    Step 4 & 6: Run analyse_dataset.py.
    Count empty and non-empty label files.
    """
    print_step(4, "Analyzing dataset")
    
    subsets = ["train", "test", "valid"]
    
    results = {}
    
    for subset in subsets:
        label_dir = os.path.join(base_dir, subset, "labels")
        if not os.path.exists(label_dir):
            print(f"  [WARNING] {label_dir} does not exist! Skipping...")
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
                # Read lines to ensure not just whitespace
                with open(label_path, 'r') as f:
                    lines = [ln.strip() for ln in f]
                    # If after stripping, no lines have content, treat as empty
                    if not any(lines):
                        empty_count += 1
                    else:
                        nonempty_count += 1
        
        total_files = len(all_label_files)
        print(f"  --- {subset.upper()} ---")
        print(f"  Total label files: {total_files}")
        print(f"  Empty label files: {empty_count}")
        print(f"  Non-empty label files: {nonempty_count}")
        print()
        
        results[subset] = {
            'total': total_files,
            'empty': empty_count,
            'nonempty': nonempty_count
        }
    
    print("Step 4 completed: Dataset analyzed")
    return results

def run_delete_empty_labels(base_dir, keep_ratio=0.2):
    """
    Step 5: Run delete_empty_labels.py.
    Delete empty label files (keeping 20%).
    """
    print_step(5, f"Deleting empty labels (keeping {keep_ratio*100}%)")
    
    import random
    random.seed(42)  # For reproducibility
    
    subsets = ["train", "test", "valid"]
    
    def is_label_file_empty(label_path):
        """Check if a label file is effectively empty."""
        if os.path.getsize(label_path) == 0:
            return True
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f]
            return not any(lines)

    def remove_file_if_exists(path):
        """Helper to remove a file if it exists."""
        if os.path.exists(path):
            os.remove(path)
    
    for subset in subsets:
        labels_dir = os.path.join(base_dir, subset, "labels")
        images_dir = os.path.join(base_dir, subset, "images")
        targets_dir = os.path.join(base_dir, subset, "targets")

        if not os.path.isdir(labels_dir):
            print(f"  [WARNING] Labels directory not found: {labels_dir}. Skipping {subset}.")
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
            print(f"  [INFO] No empty labels found in {subset}.")
            continue

        # Shuffle to pick a random percentage to keep
        random.shuffle(empty_label_files)
        keep_count = int(total_empty * keep_ratio)
        # These are the label files we keep
        keep_files = empty_label_files[:keep_count]
        # These are the label files we delete
        delete_files = empty_label_files[keep_count:]

        print(f"  [INFO] {subset.upper()} - Empty label files found: {total_empty}")
        print(f"          Keeping {keep_count}, Deleting {len(delete_files)}")

        # Remove the "delete_files" and their corresponding images/targets
        for lbl_file in delete_files:
            # Remove the label
            lbl_path = os.path.join(labels_dir, lbl_file)
            remove_file_if_exists(lbl_path)

            # Derive the base name (without extension) to find corresponding image/target
            base_name, _ = os.path.splitext(lbl_file)
            
            # Remove image and target
            img_path = os.path.join(images_dir, base_name + ".png")
            remove_file_if_exists(img_path)

            tgt_path = os.path.join(targets_dir, base_name + ".png")
            remove_file_if_exists(tgt_path)

    print(f"Step 5 completed: Empty labels deleted from {base_dir}")
    return base_dir

def run_remove_invalid_labels(base_dir):
    """
    Step 7: Run remove_invalid_labels.py.
    Remove labels with invalid YOLO format.
    """
    print_step(7, "Removing invalid labels")
    
    subsets = ["train", "test", "valid"]

    def is_label_file_invalid(label_path):
        """
        Check if any non-empty line in the label file has fewer than 5 columns.
        In YOLO polygon format, each line should have at least 5 columns:
          class x0 y0 x1 y1
        """
        if not os.path.exists(label_path):
            return False
        if os.path.getsize(label_path) == 0:
            return False
        
        with open(label_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
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

    for subset in subsets:
        labels_dir = os.path.join(base_dir, subset, "labels")
        images_dir = os.path.join(base_dir, subset, "images")
        targets_dir = os.path.join(base_dir, subset, "targets")

        if not os.path.isdir(labels_dir):
            print(f"  [WARNING] Labels directory not found: {labels_dir}. Skipping {subset}.")
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
                img_path = os.path.join(images_dir, base_name + ".png")
                tgt_path = os.path.join(targets_dir, base_name + ".png")
                
                remove_file_if_exists(img_path)
                remove_file_if_exists(tgt_path)

        print(f"  [INFO] {subset.upper()}: Removed {invalid_count} label files (with <5 columns).")

    print(f"Step 7 completed: Invalid labels removed from {base_dir}")
    return base_dir

def run_fix_annotations(base_dir):
    """
    Step 8: Run fix_annotations.py.
    Make sure polygons are closed.
    """
    print_step(8, "Fixing annotations (ensuring polygons are closed)")
    
    subsets = ["train", "test", "valid"]

    for subset in subsets:
        labels_dir = os.path.join(base_dir, subset, "labels")
        if not os.path.isdir(labels_dir):
            print(f"  [WARNING] {labels_dir} does not exist. Skipping {subset}")
            continue
        
        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
        if not label_files:
            print(f"  [INFO] No .txt files found in {labels_dir}")
            continue
        
        fixed_count = 0
        
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
                    print(f"  [WARNING] Dropping line {idx} in '{label_file}' (not enough tokens).")
                    changed_any_line = True
                    continue
                
                # parts[0] = class ID, the rest are coordinates
                class_id = parts[0]
                coords = parts[1:]  # all the x,y pairs
                if len(coords) % 2 != 0:
                    print(f"  [WARNING] Dropping line {idx} in '{label_file}' (odd number of coords).")
                    changed_any_line = True
                    continue
                
                # Convert coordinates to float
                floats = [float(x) for x in coords]
                n_points = len(floats) // 2  # number of (x,y) pairs
                
                if n_points < 3:
                    print(f"  [WARNING] Dropping line {idx} in '{label_file}' (fewer than 3 points).")
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
                fixed_count += 1
        
        print(f"  [INFO] {subset.upper()}: Fixed {fixed_count} annotation files.")

    print(f"Step 8 completed: Annotations fixed in {base_dir}")
    return base_dir

def main(src_root = "datasets/original_data",
         dst_root = "datasets/original_data_yolo",
         post_folder = "datasets/original_data_yolo/post", 
         window_size = 512, 
         keep_ratio = 0.2):

    # Process each split: train, val, test.
    for split in ["train", "valid", "test"]:
        process_directory(split, src_root=src_root, dst_root=dst_root, default_class_id=0)

    # Handle dataset folder organization
    datasets_dir = "datasets"
    dataset_dir = os.path.join(datasets_dir, "dataset")
    
    # If dataset folder exists, rename it to dataset_old with timestamp
    if os.path.exists(dataset_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_dataset_dir = os.path.join(datasets_dir, f"dataset_old_{timestamp}")
        print(f"\nRenaming existing dataset folder to: {old_dataset_dir}")
        shutil.move(dataset_dir, old_dataset_dir)
    
    # Create fresh dataset folder
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Created fresh dataset folder at: {dataset_dir}")
    
    start_time = time.time()
    
    # Step 1: Copy data to YOLOv9 format
    post_folder = run_copy_data(post_folder, src_root)
    
    # Step 2: Subsample images
    post_folder = run_subsample_images(post_folder)
    
    # Step 3: Image windowing
    windowed_folder = run_image_windowing(post_folder, window_size)
    
    # Step 4: Analyze dataset before deleting empty labels
    print("Dataset analysis BEFORE deleting empty labels:")
    before_stats = run_analyse_dataset(windowed_folder)
    
    # Step 5: Delete empty labels
    windowed_folder = run_delete_empty_labels(windowed_folder, keep_ratio)
    
    # Step 6: Analyze dataset after deleting empty labels
    print("Dataset analysis AFTER deleting empty labels:")
    after_stats = run_analyse_dataset(windowed_folder)
    
    # Step 7: Remove invalid labels
    windowed_folder = run_remove_invalid_labels(windowed_folder)
    
    # Step 8: Fix annotations
    windowed_folder = run_fix_annotations(windowed_folder)
    
    # Move the final processed dataset to the new dataset folder
    final_dataset_name = os.path.basename(windowed_folder)
    final_dataset_path = os.path.join(dataset_dir, final_dataset_name)
    print(f"\nMoving final dataset to: {final_dataset_path}")
    shutil.move(windowed_folder, final_dataset_path)
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*80)
    print(f"PREPROCESSING PIPELINE COMPLETED in {total_time:.2f} seconds")
    print("="*80)
    print(f"Final dataset location: {final_dataset_path}")
    print("="*80)

if __name__ == "__main__":
    main(src_root = "datasets/original_data",
         dst_root = "datasets/original_data_yolo",
         post_folder = "datasets/original_data_yolo/post", 
         window_size = 512, 
         keep_ratio = 0.2) 