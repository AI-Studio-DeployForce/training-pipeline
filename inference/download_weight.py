import os
import shutil
from clearml import Model

models = Model.query_models(
    model_name="YOLOv9_BuildingDamage_Segmentation",
    tags=["best"],
    only_published=True  
)

if not models:
    raise ValueError("No published models found with the specified name/tag.")

model = models[0]
model_path = model.get_local_copy()

# Target folder
destination_folder = "/home/ec2-user/SageMaker/downloaded_models"
os.makedirs(destination_folder, exist_ok=True)

# Define full destination path
filename = os.path.basename(model_path)
destination_path = os.path.join(destination_folder, filename)

# Copy the model file to the new location
shutil.copy(model_path, destination_path)

print(f"Model copied to: {destination_path}")