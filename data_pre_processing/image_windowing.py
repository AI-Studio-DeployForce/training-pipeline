import os
import cv2
import numpy as np

# Adjust these as needed
SOURCE_DIR = "./yolov9/data/dataset"  # Path to your root dataset folder
DEST_DIR = "./yolov9/data/dataset_256"  # Where to save the 256×256 results
SUBSETS = ["train", "test", "valid"]

# Size of the tiles
WINDOW_SIZE = 256
ORIG_SIZE = 1024  # original image is 1024×1024
NUM_TILES = ORIG_SIZE // WINDOW_SIZE  # 4

# Ensure destination directories exist
for subset in SUBSETS:
    os.makedirs(os.path.join(DEST_DIR, subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, subset, "targets"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, subset, "labels"), exist_ok=True)

def save_subimage(img, out_path):
    """Utility to save a subimage (BGR or Gray)."""
    cv2.imwrite(out_path, img)

def convert_contours_to_yolo_format(contours, img_w, img_h):
    """
    Convert OpenCV contours to a YOLO segmentation polygon format:
      x0 y0 x1 y1 ... (normalized coordinates)
    and ensure the polygon is explicitly closed (end point == start point).
    """
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

def process_mask_to_yolo_txt(mask_subimg, out_txt_path, tile_width=256, tile_height=256):
    """
    Find contours for each class c ∈ {1,2,3,4} in the sub-mask,
    map them to YOLO class IDs {0,1,2,3}, and write polygon lines.
    """
    # Dictionary that maps actual pixel value to YOLO class
    # pixel=1 => class=0, pixel=2 => class=1, etc.
    class_map = {1: 0, 2: 1, 3: 2, 4: 3}
    
    lines = []
    for pixel_val, yolo_class in class_map.items():
        # Create a binary mask for the given pixel value
        # (OpenCV findContours needs a binary mask: 0 or 255)
        bin_mask = np.where(mask_subimg == pixel_val, 255, 0).astype(np.uint8)
        
        # Find contours in the binarized mask
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue  # No polygons for this class in this sub-tile
        
        # Convert each contour to YOLO polygon format
        polygons = convert_contours_to_yolo_format(contours, tile_width, tile_height)
        
        # Build lines: "class x0 y0 x1 y1 x2 y2 ..."
        for poly in polygons:
            poly_str = " ".join(str(round(p, 6)) for p in poly)  # Round or keep float
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

def slice_and_annotate_images(subset):
    """
    For each image in subset/images, slice it into 256×256,
    do the same for the target mask,
    generate YOLOv9 annotations in subset/labels.
    """
    images_dir = os.path.join(SOURCE_DIR, subset, "images")
    targets_dir = os.path.join(SOURCE_DIR, subset, "targets")
    
    out_images_dir = os.path.join(DEST_DIR, subset, "images")
    out_targets_dir = os.path.join(DEST_DIR, subset, "targets")
    out_labels_dir = os.path.join(DEST_DIR, subset, "labels")
    
    # List all image files (assuming .png or .jpg, adapt as needed)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in image_files:
        # Build full path
        img_path = os.path.join(images_dir, img_name)
        
        # Corresponding target name (assuming same name with .png extension)
        # Adjust if your naming is different
        base_name, _ = os.path.splitext(img_name)
        target_name = base_name+"_target" + ".png"
        target_path = os.path.join(targets_dir, target_name)
        
        # Read image and target
        img = cv2.imread(img_path)  # BGR
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        target_mask = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
        if target_mask is None:
            print(f"Could not read mask: {target_path}")
            continue
        
        # For each tile in [0..3]×[0..3]
        for i in range(NUM_TILES):
            for j in range(NUM_TILES):
                # Calculate the slice boundaries
                x_start = j * WINDOW_SIZE
                x_end = x_start + WINDOW_SIZE
                y_start = i * WINDOW_SIZE
                y_end = y_start + WINDOW_SIZE
                
                # Slice the image
                sub_img = img[y_start:y_end, x_start:x_end]
                
                # Slice the target
                sub_mask = target_mask[y_start:y_end, x_start:x_end]
                
                # Create output file names
                sub_img_name = f"{base_name}_{i}_{j}.png"
                sub_target_name = f"{base_name}_{i}_{j}.png"
                sub_label_name = f"{base_name}_{i}_{j}.txt"
                
                # Write out the sub image
                out_img_path = os.path.join(out_images_dir, sub_img_name)
                save_subimage(sub_img, out_img_path)
                
                # Write out the sub target
                out_target_path = os.path.join(out_targets_dir, sub_target_name)
                cv2.imwrite(out_target_path, sub_mask)
                
                # Generate new label from sub_mask
                out_label_path = os.path.join(out_labels_dir, sub_label_name)
                process_mask_to_yolo_txt(sub_mask, out_label_path, WINDOW_SIZE, WINDOW_SIZE)

def main():
    for subset in SUBSETS:
        slice_and_annotate_images(subset)

if __name__ == "__main__":
    main()
