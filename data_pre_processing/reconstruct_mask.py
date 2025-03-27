import cv2
import numpy as np
import matplotlib.pyplot as plt

annotation_file = "datasets/original_data_yolo/post_512/train/labels/hurricane-florence_00000409_post_disaster_1_0.txt"
image_file = "datasets/original_data_yolo/post_512/train/images/hurricane-florence_00000409_post_disaster_1_0.png"


# Load the image
image = cv2.imread(image_file)
height, width, _ = image.shape

# Parse the annotation file with YOLO format
with open(annotation_file, "r") as file:
    annotations = file.readlines()

# Define a fixed color mapping for each class (B, G, R)
fixed_colors = {
    0: (0, 255, 0),   # Green (No damage)
    1: (0, 255, 255), # Yellow (Minor damage)
    2: (0, 165, 255), # Orange (Major damage)
    3: (0, 0, 255)    # Red (Destroyed)
}

# Initialize an empty mask with 3 channels (for color)
mask = np.zeros((height, width, 3), dtype=np.uint8)

# Loop through each annotation
for annotation in annotations:
    data = annotation.strip().split()
    label = int(data[0])  # Class label
    coordinates = list(map(float, data[1:]))

    # Extract x,y pairs from the coordinates
    points = [
        (int(width * coordinates[i]), int(height * coordinates[i + 1]))
        for i in range(0, len(coordinates), 2)
    ]
    # Convert to a NumPy array of shape (n_points, 1, 2) for fillPoly
    points = np.array(points, dtype=np.int32)

    # Lookup the color for this label; default to white if an unknown label
    color = fixed_colors.get(label, (255, 255, 255))

    # Fill each class polygon with the class's color
    cv2.fillPoly(mask, [points], color=color)

# Display the original image and the colored mask
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reconstructed Mask (Colored by Class)")
# Note that our mask is in BGR; convert to RGB for plotting
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the mask as an image file
output_mask_file = "data_pre_processing/reconstructed_mask_color.png"
cv2.imwrite(output_mask_file, mask)
print(f"Mask saved to {output_mask_file}")