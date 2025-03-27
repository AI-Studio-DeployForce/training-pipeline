import os
import shutil

# Define the mapping for folder names:
# In the 'data' folder, 'train' and 'test' remain the same, but 'valid' in post corresponds to 'holdout' in data.
subset_mapping = {
    "train": "train",
    "valid": "holdout",
    "test": "test"
}

# Base folders - [PARAMETERS]
post_folder = "D:/Kuliah_S2/Semester_4/AIS/Code/yolov9-building-evaluation/YOLO_pre_post/post"
data_folder = "D:/Kuliah_S2/Semester_4/AIS/Code/segmentation-model/data"

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

    print(f"Processing subset: {subset_post} (data folder: {subset_data})")
    
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
                print(f"Copied image: {image_filename}")
            else:
                print(f"Image file not found: {src_image_path}")

            # Copy the target file if it exists
            if os.path.exists(src_target_path):
                shutil.copy(src_target_path, dest_target_path)
                print(f"Copied target: {target_filename}")
            else:
                print(f"Target file not found: {src_target_path}")
