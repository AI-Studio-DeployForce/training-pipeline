import os
import json
import argparse

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
    Process one dataset split (train/val/test) and convert its JSON labels to YOLO segmentation format.
    
    The script now determines the destination folder based on the file name:
      - If the file name contains "post" then the converted labels are saved to:
            <dst_root>/post/<split>/labels/
      - If the file name contains "pre" then the labels are saved to:
            <dst_root>/pre/<split>/labels/
      - If neither is specified, it defaults to the "pre" folder.
    
    Note: If the split is "val", the output folder is named "valid".
    """
    src_labels_dir = os.path.join(src_root, split, "labels")
    out_split = "valid" if split == "val" else split
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON labels to YOLO segmentation format and separate post and pre disaster labels based on file naming.")
    parser.add_argument("--src_root", type=str, default="segmentation_dataset", help="Path to the dataset root")
    parser.add_argument("--dst_root", type=str, default="YOLO", help="Path to save YOLO labels")
    args = parser.parse_args()
    
    # Process each split: train, val, test.
    for split in ["train", "val", "test"]:
        process_directory(split, src_root=args.src_root, dst_root=args.dst_root, default_class_id=0)