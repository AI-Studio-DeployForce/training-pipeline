import os
import shutil
import time
import json
import random
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse

@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""
    src_root: str = "datasets/original_data"
    dst_root: str = "datasets/original_data_yolo"
    post_folder: str = "datasets/original_data_yolo/post"
    window_size: int = 512
    keep_ratio: float = 0.2
    default_width: int = 1024
    default_height: int = 1024
    default_class_id: int = 0

class DataProcessor:
    """Handles data processing utilities like polygon parsing and format conversion."""
    
    @staticmethod
    def parse_polygon(wkt_str: str) -> List[Tuple[float, float]]:
        """Parse a WKT polygon string into a list of (x, y) coordinate tuples."""
        if not wkt_str.startswith("POLYGON ((") or not wkt_str.endswith("))"):
            raise ValueError("Invalid WKT format")
        
        coords_str = wkt_str[len("POLYGON (("):-2]
        coords = []
        
        for pair in coords_str.split(","):
            x_str, y_str = pair.strip().split()
            coords.append((float(x_str), float(y_str)))
        
        return coords

    @staticmethod
    def convert_json_to_yolo(
        json_file: str,
        default_width: int = 1024,
        default_height: int = 1024,
        class_map: Optional[Dict[str, int]] = None,
        default_class_id: int = 0
    ) -> List[str]:
        """Convert a JSON label file to YOLO segmentation format."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        image_width = data.get("metadata", {}).get("width", default_width)
        image_height = data.get("metadata", {}).get("height", default_height)
        
        yolo_lines = []
        features = data.get("features", {}).get("xy", [])
        
        for feature in features:
            properties = feature.get("properties", {})
            
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
                    continue
            else:
                class_id = default_class_id
            
            wkt = feature.get("wkt", "")
            try:
                coords = DataProcessor.parse_polygon(wkt)
            except ValueError as e:
                print(f"Error parsing polygon in file {json_file}: {e}")
                continue
            
            norm_coords = []
            for (x, y) in coords:
                norm_x = x / image_width
                norm_y = y / image_height
                norm_coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
            
            line = f"{class_id} " + " ".join(norm_coords)
            yolo_lines.append(line)
        
        return yolo_lines

class PipelineStep:
    """Base class for pipeline steps."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def execute(self) -> str:
        """Execute the pipeline step. Must be implemented by subclasses."""
        raise NotImplementedError

class DataCopier(PipelineStep):
    """Step 1: Copy data to YOLOv9 format."""
    
    def execute(self) -> str:
        print("\n" + "="*80)
        print("STEP 1: Copying data to YOLOv9 format")
        print("="*80)
        
        subset_mapping = {"train": "train", "valid": "valid", "test": "test"}
        
        for subset_post, subset_data in subset_mapping.items():
            self._process_subset(subset_post, subset_data)
        
        print(f"Step 1 completed: Data copied to {self.config.post_folder}")
        return self.config.post_folder
    
    def _process_subset(self, subset_post: str, subset_data: str) -> None:
        """Process a single subset of the data."""
        post_labels_folder = os.path.join(self.config.post_folder, subset_post, "labels")
        dest_images_folder = os.path.join(self.config.post_folder, subset_post, "images")
        dest_targets_folder = os.path.join(self.config.post_folder, subset_post, "targets")
        
        os.makedirs(dest_images_folder, exist_ok=True)
        os.makedirs(dest_targets_folder, exist_ok=True)
        
        data_images_folder = os.path.join(self.config.src_root, subset_data, "images")
        data_targets_folder = os.path.join(self.config.src_root, subset_data, "targets")
        
        print(f"  Processing subset: {subset_post} (data folder: {subset_data})")
        
        for label_file in os.listdir(post_labels_folder):
            if not label_file.endswith(".txt"):
                continue
            
            base_name = os.path.splitext(label_file)[0]
            self._copy_files(base_name, subset_post, subset_data, 
                           data_images_folder, data_targets_folder,
                           dest_images_folder, dest_targets_folder)
    
    def _copy_files(self, base_name: str, subset_post: str, subset_data: str,
                   data_images_folder: str, data_targets_folder: str,
                   dest_images_folder: str, dest_targets_folder: str) -> None:
        """Copy image and target files for a given base name."""
        image_filename = base_name + ".png"
        target_filename = base_name + "_target.png"
        
        src_image_path = os.path.join(data_images_folder, image_filename)
        src_target_path = os.path.join(data_targets_folder, target_filename)
        
        dest_image_path = os.path.join(dest_images_folder, image_filename)
        dest_target_path = os.path.join(dest_targets_folder, target_filename)
        
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dest_image_path)
        else:
            print(f"  Image file not found: {src_image_path}")
        
        if os.path.exists(src_target_path):
            shutil.copy(src_target_path, dest_target_path)
        else:
            print(f"  Target file not found: {src_target_path}")

class DataSubsampler(PipelineStep):
    """Step 2: Subsample images by deleting 40% of the dataset."""
    
    def execute(self) -> str:
        print("\n" + "="*80)
        print("STEP 2: Subsampling images (deleting 40% of the dataset)")
        print("="*80)
        
        random.seed(42)  # For reproducibility
        subsets = ["train", "valid", "test"]
        
        for subset in subsets:
            self._process_subset(subset)
        
        print(f"Step 2 completed: Dataset subsampled in {self.config.post_folder}")
        return self.config.post_folder
    
    def _process_subset(self, subset: str) -> None:
        """Process a single subset for subsampling."""
        images_folder = os.path.join(self.config.post_folder, subset, "images")
        targets_folder = os.path.join(self.config.post_folder, subset, "targets")
        labels_folder = os.path.join(self.config.post_folder, subset, "labels")
        
        image_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]
        total_images = len(image_files)
        delete_count = int(total_images * 0.4)
        
        print(f"  Subset '{subset}': Deleting {delete_count} out of {total_images} images.")
        
        files_to_delete = random.sample(image_files, delete_count)
        
        for image_file in files_to_delete:
            base_name = os.path.splitext(image_file)[0]
            self._delete_files(base_name, subset)

    def _delete_files(self, base_name: str, subset: str) -> None:
        """Delete image, target, and label files for a given base name."""
        folders = {
            "images": os.path.join(self.config.post_folder, subset, "images"),
            "targets": os.path.join(self.config.post_folder, subset, "targets"),
            "labels": os.path.join(self.config.post_folder, subset, "labels")
        }
        
        for folder_type, folder_path in folders.items():
            file_path = os.path.join(folder_path, f"{base_name}{'.png' if folder_type != 'labels' else '.txt'}")
            if os.path.exists(file_path):
                os.remove(file_path)

class ImageWindower(PipelineStep):
    """Step 3: Slice images and targets into smaller windows."""
    
    def execute(self) -> str:
        print("\n" + "="*80)
        print(f"STEP 3: Executing image windowing (size: {self.config.window_size}×{self.config.window_size})")
        print("="*80)
        
        dest_dir = self.config.post_folder + f"_{self.config.window_size}"
        subsets = ["train", "test", "valid"]
        
        self._create_destination_directories(dest_dir, subsets)
        
        for subset in subsets:
            self._process_subset(subset, dest_dir)
        
        print(f"Step 3 completed: Images windowed and saved to {dest_dir}")
        return dest_dir
    
    def _create_destination_directories(self, dest_dir: str, subsets: List[str]) -> None:
        """Create destination directories for windowed data."""
        for subset in subsets:
            for folder in ["images", "targets", "labels"]:
                os.makedirs(os.path.join(dest_dir, subset, folder), exist_ok=True)
    
    def _process_subset(self, subset: str, dest_dir: str) -> None:
        """Process a single subset for windowing."""
        images_dir = os.path.join(self.config.post_folder, subset, "images")
        targets_dir = os.path.join(self.config.post_folder, subset, "targets")
        
        out_images_dir = os.path.join(dest_dir, subset, "images")
        out_targets_dir = os.path.join(dest_dir, subset, "targets")
        out_labels_dir = os.path.join(dest_dir, subset, "labels")
        
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)
        processed = 0
        
        for img_name in image_files:
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{total_images} images in {subset}")
            
            self._process_image(img_name, subset, dest_dir, images_dir, targets_dir,
                              out_images_dir, out_targets_dir, out_labels_dir)
    
    def _process_image(self, img_name: str, subset: str, dest_dir: str,
                      images_dir: str, targets_dir: str,
                      out_images_dir: str, out_targets_dir: str,
                      out_labels_dir: str) -> None:
        """Process a single image for windowing."""
        img_path = os.path.join(images_dir, img_name)
        base_name, _ = os.path.splitext(img_name)
        target_name = base_name + "_target.png"
        target_path = os.path.join(targets_dir, target_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Could not read image: {img_path}")
            return
        
        target_mask = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
        if target_mask is None:
            print(f"  Could not read mask: {target_path}")
            return
        
        num_tiles = 1024 // self.config.window_size
        
        for i in range(num_tiles):
            for j in range(num_tiles):
                self._process_tile(i, j, img, target_mask, base_name, subset, dest_dir,
                                 out_images_dir, out_targets_dir, out_labels_dir)
    
    def _process_tile(self, i: int, j: int, img: np.ndarray, target_mask: np.ndarray,
                     base_name: str, subset: str, dest_dir: str,
                     out_images_dir: str, out_targets_dir: str,
                     out_labels_dir: str) -> None:
        """Process a single tile from an image."""
        x_start = j * self.config.window_size
        x_end = x_start + self.config.window_size
        y_start = i * self.config.window_size
        y_end = y_start + self.config.window_size
        
        sub_img = img[y_start:y_end, x_start:x_end]
        sub_mask = target_mask[y_start:y_end, x_start:x_end]
        
        sub_img_name = f"{base_name}_{i}_{j}.png"
        sub_target_name = f"{base_name}_{i}_{j}.png"
        sub_label_name = f"{base_name}_{i}_{j}.txt"
        
        cv2.imwrite(os.path.join(out_images_dir, sub_img_name), sub_img)
        cv2.imwrite(os.path.join(out_targets_dir, sub_target_name), sub_mask)
        
        out_label_path = os.path.join(out_labels_dir, sub_label_name)
        self._process_mask_to_yolo_txt(sub_mask, out_label_path)
    
    def _process_mask_to_yolo_txt(self, mask_subimg: np.ndarray, out_txt_path: str) -> None:
        """Convert mask to YOLO format text file."""
        class_map = {1: 0, 2: 1, 3: 2, 4: 3}
        lines = []
        
        for pixel_val, yolo_class in class_map.items():
            bin_mask = np.where(mask_subimg == pixel_val, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            polygons = self._convert_contours_to_yolo_format(contours)
            
            for poly in polygons:
                poly_str = " ".join(str(round(p, 6)) for p in poly)
                line_str = f"{yolo_class} {poly_str}"
                lines.append(line_str)
        
        with open(out_txt_path, "w") as f:
            for line in lines:
                f.write(line + "\n")
    
    def _convert_contours_to_yolo_format(self, contours: List[np.ndarray]) -> List[List[float]]:
        """Convert OpenCV contours to YOLO segmentation polygon format."""
        polygons = []
        for cnt in contours:
            cnt = cnt.reshape(-1, 2)
            poly = []
            for (x, y) in cnt:
                nx = x / self.config.window_size
                ny = y / self.config.window_size
                poly.append(nx)
                poly.append(ny)
            
            if len(cnt) > 0:
                first_x, first_y = cnt[0]
                poly.append(first_x / self.config.window_size)
                poly.append(first_y / self.config.window_size)
            
            polygons.append(poly)
        return polygons

class DatasetAnalyzer(PipelineStep):
    """Step 4 & 6: Analyze dataset statistics."""
    
    def execute(self) -> Dict[str, Dict[str, int]]:
        print("\n" + "="*80)
        print("STEP 4: Analyzing dataset")
        print("="*80)
        
        subsets = ["train", "test", "valid"]
        results = {}
        
        for subset in subsets:
            results[subset] = self._analyze_subset(subset)
        
        print("Step 4 completed: Dataset analyzed")
        return results
    
    def _analyze_subset(self, subset: str) -> Dict[str, int]:
        """Analyze a single subset of the dataset."""
        label_dir = os.path.join(self.config.post_folder, subset, "labels")
        if not os.path.exists(label_dir):
            print(f"  [WARNING] {label_dir} does not exist! Skipping...")
            return {'total': 0, 'empty': 0, 'nonempty': 0}
        
        all_label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
        
        empty_count = 0
        nonempty_count = 0
        
        for label_file in all_label_files:
            label_path = os.path.join(label_dir, label_file)
            
            if os.path.getsize(label_path) == 0:
                empty_count += 1
            else:
                with open(label_path, 'r') as f:
                    lines = [ln.strip() for ln in f]
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
        
        return {
            'total': total_files,
            'empty': empty_count,
            'nonempty': nonempty_count
        }

class EmptyLabelCleaner(PipelineStep):
    """Step 5: Delete empty label files (keeping 20%)."""
    
    def execute(self) -> str:
        print("\n" + "="*80)
        print(f"STEP 5: Deleting empty labels (keeping {self.config.keep_ratio*100}%)")
        print("="*80)
        
        random.seed(42)  # For reproducibility
        subsets = ["train", "test", "valid"]
        
        for subset in subsets:
            self._process_subset(subset)
        
        print(f"Step 5 completed: Empty labels deleted from {self.config.post_folder}")
        return self.config.post_folder
    
    def _process_subset(self, subset: str) -> None:
        """Process a single subset for empty label cleaning."""
        labels_dir = os.path.join(self.config.post_folder, subset, "labels")
        images_dir = os.path.join(self.config.post_folder, subset, "images")
        targets_dir = os.path.join(self.config.post_folder, subset, "targets")
        
        if not os.path.isdir(labels_dir):
            print(f"  [WARNING] Labels directory not found: {labels_dir}. Skipping {subset}.")
            return
        
        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
        empty_label_files = []
        
        for lbl in label_files:
            lbl_path = os.path.join(labels_dir, lbl)
            if self._is_label_file_empty(lbl_path):
                empty_label_files.append(lbl)
        
        total_empty = len(empty_label_files)
        if total_empty == 0:
            print(f"  [INFO] No empty labels found in {subset}.")
            return
        
        random.shuffle(empty_label_files)
        keep_count = int(total_empty * self.config.keep_ratio)
        keep_files = empty_label_files[:keep_count]
        delete_files = empty_label_files[keep_count:]
        
        print(f"  [INFO] {subset.upper()} - Empty label files found: {total_empty}")
        print(f"          Keeping {keep_count}, Deleting {len(delete_files)}")
        
        for lbl_file in delete_files:
            self._delete_files(lbl_file, subset, labels_dir, images_dir, targets_dir)
    
    def _is_label_file_empty(self, label_path: str) -> bool:
        """Check if a label file is effectively empty."""
        if os.path.getsize(label_path) == 0:
            return True
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f]
            return not any(lines)
    
    def _delete_files(self, lbl_file: str, subset: str,
                     labels_dir: str, images_dir: str, targets_dir: str) -> None:
        """Delete image, target, and label files for a given label file."""
        lbl_path = os.path.join(labels_dir, lbl_file)
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
        
        base_name, _ = os.path.splitext(lbl_file)
        img_path = os.path.join(images_dir, base_name + ".png")
        tgt_path = os.path.join(targets_dir, base_name + ".png")
        
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(tgt_path):
            os.remove(tgt_path)

class InvalidLabelCleaner(PipelineStep):
    """Step 7: Remove labels with invalid YOLO format."""
    
    def execute(self) -> str:
        print("\n" + "="*80)
        print("STEP 7: Removing invalid labels")
        print("="*80)
        
        subsets = ["train", "test", "valid"]
        
        for subset in subsets:
            self._process_subset(subset)
        
        print(f"Step 7 completed: Invalid labels removed from {self.config.post_folder}")
        return self.config.post_folder
    
    def _process_subset(self, subset: str) -> None:
        """Process a single subset for invalid label cleaning."""
        labels_dir = os.path.join(self.config.post_folder, subset, "labels")
        images_dir = os.path.join(self.config.post_folder, subset, "images")
        targets_dir = os.path.join(self.config.post_folder, subset, "targets")
        
        if not os.path.isdir(labels_dir):
            print(f"  [WARNING] Labels directory not found: {labels_dir}. Skipping {subset}.")
            return
        
        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
        invalid_count = 0
        
        for label_file in label_files:
            lbl_path = os.path.join(labels_dir, label_file)
            if self._is_label_file_invalid(lbl_path):
                invalid_count += 1
                self._delete_files(label_file, subset, labels_dir, images_dir, targets_dir)
        
        print(f"  [INFO] {subset.upper()}: Removed {invalid_count} label files (with <5 columns).")
    
    def _is_label_file_invalid(self, label_path: str) -> bool:
        """Check if a label file has invalid YOLO format."""
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
                if len(parts) < 5:
                    return True
        
        return False
    
    def _delete_files(self, lbl_file: str, subset: str,
                     labels_dir: str, images_dir: str, targets_dir: str) -> None:
        """Delete image, target, and label files for a given label file."""
        lbl_path = os.path.join(labels_dir, lbl_file)
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
        
        base_name, _ = os.path.splitext(lbl_file)
        img_path = os.path.join(images_dir, base_name + ".png")
        tgt_path = os.path.join(targets_dir, base_name + ".png")
        
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(tgt_path):
            os.remove(tgt_path)

class AnnotationFixer(PipelineStep):
    """Step 8: Fix annotations to ensure polygons are closed."""
    
    def execute(self) -> str:
        print("\n" + "="*80)
        print("STEP 8: Fixing annotations (ensuring polygons are closed)")
        print("="*80)
        
        subsets = ["train", "test", "valid"]
        
        for subset in subsets:
            self._process_subset(subset)
        
        print(f"Step 8 completed: Annotations fixed in {self.config.post_folder}")
        return self.config.post_folder
    
    def _process_subset(self, subset: str) -> None:
        """Process a single subset for annotation fixing."""
        labels_dir = os.path.join(self.config.post_folder, subset, "labels")
        if not os.path.isdir(labels_dir):
            print(f"  [WARNING] {labels_dir} does not exist. Skipping {subset}")
            return
        
        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
        if not label_files:
            print(f"  [INFO] No .txt files found in {labels_dir}")
            return
        
        fixed_count = 0
        
        for label_file in label_files:
            if self._fix_annotation_file(label_file, labels_dir):
                fixed_count += 1
        
        print(f"  [INFO] {subset.upper()}: Fixed {fixed_count} annotation files.")
    
    def _fix_annotation_file(self, label_file: str, labels_dir: str) -> bool:
        """Fix a single annotation file."""
        label_path = os.path.join(labels_dir, label_file)
        
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        
        new_lines = []
        changed_any_line = False
        
        for idx, line in enumerate(lines, start=1):
            parts = line.split()
            
            if len(parts) < 7:
                print(f"  [WARNING] Dropping line {idx} in '{label_file}' (not enough tokens).")
                changed_any_line = True
                continue
            
            class_id = parts[0]
            coords = parts[1:]
            
            if len(coords) % 2 != 0:
                print(f"  [WARNING] Dropping line {idx} in '{label_file}' (odd number of coords).")
                changed_any_line = True
                continue
            
            floats = [float(x) for x in coords]
            n_points = len(floats) // 2
            
            if n_points < 3:
                print(f"  [WARNING] Dropping line {idx} in '{label_file}' (fewer than 3 points).")
                changed_any_line = True
                continue
            
            x_first, y_first = floats[0], floats[1]
            x_last, y_last = floats[-2], floats[-1]
            
            if not (x_first == x_last and y_first == y_last):
                floats.extend([x_first, y_first])
                changed_any_line = True
            
            poly_str = " ".join(str(v) for v in floats)
            new_line = f"{class_id} {poly_str}"
            new_lines.append(new_line)
        
        if changed_any_line or len(new_lines) != len(lines):
            with open(label_path, "w") as f:
                for nl in new_lines:
                    f.write(nl + "\n")
            return True
        
        return False

class PreprocessingPipeline:
    """Main class that orchestrates the preprocessing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.steps = [
            DataCopier(config),
            DataSubsampler(config),
            ImageWindower(config),
            DatasetAnalyzer(config),
            EmptyLabelCleaner(config),
            DatasetAnalyzer(config),
            InvalidLabelCleaner(config),
            AnnotationFixer(config)
        ]
    
    def run(self) -> None:
        """Run the complete preprocessing pipeline."""
        start_time = time.time()
        
        # Process each split: train, val, test
        for split in ["train", "valid", "test"]:
            self._process_directory(split)
        
        # Handle dataset folder organization
        self._setup_dataset_folder()
        
        # Run preprocessing pipeline steps
        current_folder = self.config.post_folder
        for step in self.steps:
            current_folder = step.execute()
        
        # Move contents to final dataset location
        self._move_to_final_location(current_folder)
        
        # Clean up
        shutil.rmtree(current_folder)
        
        # Print final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*80)
        print(f"PREPROCESSING PIPELINE COMPLETED in {total_time:.2f} seconds")
        print("="*80)
        print(f"Final dataset location: {self.config.dst_root}")
        print("="*80)
    
    def _process_directory(self, split: str) -> None:
        """Process one dataset split (train/valid/test)."""
        src_labels_dir = os.path.join(self.config.src_root, split, "labels")
        out_split = "valid" if split == "valid" else split
        dst_labels_dir_post = os.path.join(self.config.dst_root, "post", out_split, "labels")
        dst_labels_dir_pre = os.path.join(self.config.dst_root, "pre", out_split, "labels")
        
        os.makedirs(dst_labels_dir_post, exist_ok=True)
        os.makedirs(dst_labels_dir_pre, exist_ok=True)
        
        class_map = {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3
        }
        
        for filename in os.listdir(src_labels_dir):
            if not filename.endswith(".json"):
                continue
            
            src_file = os.path.join(src_labels_dir, filename)
            yolo_lines = DataProcessor.convert_json_to_yolo(
                src_file, 
                class_map=class_map,
                default_class_id=self.config.default_class_id
            )
            
            lower_filename = filename.lower()
            if "post" in lower_filename:
                dst_file = os.path.join(dst_labels_dir_post, filename.replace(".json", ".txt"))
            elif "pre" in lower_filename:
                dst_file = os.path.join(dst_labels_dir_pre, filename.replace(".json", ".txt"))
            else:
                dst_file = os.path.join(dst_labels_dir_pre, filename.replace(".json", ".txt"))
            
            with open(dst_file, 'w') as out_f:
                for line in yolo_lines:
                    out_f.write(line + "\n")
    
    def _setup_dataset_folder(self) -> None:
        """Setup the final dataset folder."""
        datasets_dir = "datasets"
        dataset_dir = os.path.join(datasets_dir, "dataset")
        
        if os.path.exists(dataset_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            old_dataset_dir = os.path.join(datasets_dir, f"dataset_old_{timestamp}")
            print(f"\nRenaming existing dataset folder to: {old_dataset_dir}")
            shutil.move(dataset_dir, old_dataset_dir)
        
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Created fresh dataset folder at: {dataset_dir}")
    
    def _move_to_final_location(self, current_folder: str) -> None:
        """Move processed data to final dataset location."""
        print(f"\nMoving contents to: {self.config.dst_root}")
        for item in os.listdir(current_folder):
            s = os.path.join(current_folder, item)
            d = os.path.join(self.config.dst_root, item)
            if os.path.isdir(s):
                shutil.move(s, d)
            else:
                shutil.copy2(s, d)

def main(
    src_root: str = "datasets/original_data",
    dst_root: str = "datasets/original_data_yolo",
    post_folder: str = "datasets/original_data_yolo/post",
    window_size: int = 512,
    keep_ratio: float = 0.2,
    default_width: int = 1024,
    default_height: int = 1024,
    default_class_id: int = 0
) -> None:
    """
    Main entry point for the preprocessing pipeline.
    
    Args:
        src_root: Root directory containing source data
        dst_root: Root directory for YOLO format data
        post_folder: Path to post-disaster data folder
        window_size: Size of the windows for image splitting
        keep_ratio: Ratio of empty labels to keep
        default_width: Default image width if not specified in metadata
        default_height: Default image height if not specified in metadata
        default_class_id: Default class ID for pre-disaster buildings
    """
    config = PipelineConfig(
        src_root=src_root,
        dst_root=dst_root,
        post_folder=post_folder,
        window_size=window_size,
        keep_ratio=keep_ratio,
        default_width=default_width,
        default_height=default_height,
        default_class_id=default_class_id
    )
    pipeline = PreprocessingPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the YOLOv9 preprocessing pipeline")
    parser.add_argument("--src-root", default="datasets/original_data",
                      help="Root directory containing source data")
    parser.add_argument("--dst-root", default="datasets/original_data_yolo",
                      help="Root directory for YOLO format data")
    parser.add_argument("--post-folder", default="datasets/original_data_yolo/post",
                      help="Path to post-disaster data folder")
    parser.add_argument("--window-size", type=int, default=512,
                      help="Size of the windows for image splitting")
    parser.add_argument("--keep-ratio", type=float, default=0.2,
                      help="Ratio of empty labels to keep")
    parser.add_argument("--default-width", type=int, default=1024,
                      help="Default image width if not specified in metadata")
    parser.add_argument("--default-height", type=int, default=1024,
                      help="Default image height if not specified in metadata")
    parser.add_argument("--default-class-id", type=int, default=0,
                      help="Default class ID for pre-disaster buildings")
    
    args = parser.parse_args()
    main(**vars(args)) 