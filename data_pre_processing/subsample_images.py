import os
import random

# Base folder where the post subsets are located
post_folder = "./yolov9/data/dataset"

# List of subsets to process
subsets = ["train", "valid", "test"]

# Loop through each subset folder
for subset in subsets:
    # Define the full paths to the subfolders for images, targets, and labels
    images_folder = os.path.join(post_folder, subset, "images")
    targets_folder = os.path.join(post_folder, subset, "targets")
    labels_folder = os.path.join(post_folder, subset, "labels")
    
    # List all .png files in the images folder (assuming images have .png extension)
    image_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]
    total_images = len(image_files)
    
    # Calculate 40% of the total images (using int to get a whole number)
    delete_count = int(total_images * 0.4)
    print(f"Subset '{subset}': Deleting {delete_count} out of {total_images} images.")
    
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
        
        # Delete the image file if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        else:
            print(f"Image not found: {image_path}")
        
        # Delete the target file if it exists
        if os.path.exists(target_path):
            os.remove(target_path)
            print(f"Deleted target: {target_path}")
        else:
            print(f"Target not found: {target_path}")
        
        # Delete the label file if it exists
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Deleted label: {label_path}")
        else:
            print(f"Label not found: {label_path}")
