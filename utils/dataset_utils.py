import os
from clearml import Dataset
import shutil

def check_and_download_dataset(dataset_name: str, project_name: str, local_path: str) -> bool:
    """
    Check if dataset exists locally and download it if not.
    
    Args:
        dataset_name: Name of the dataset in ClearML
        project_name: Name of the project in ClearML
        local_path: Local path where the dataset should be stored
    
    Returns:
        bool: True if dataset exists or was downloaded successfully, False otherwise
    """
    print(f"\nChecking for dataset: {dataset_name}")
    
    # Check if dataset exists locally
    if os.path.exists(local_path):
        # Verify the required structure exists
        required_folders = ["train", "valid", "test"]
        has_required_structure = all(os.path.exists(os.path.join(local_path, folder)) for folder in required_folders)
        
        if has_required_structure:
            print(f"Dataset {dataset_name} found locally at {local_path}")
            return True
        else:
            print(f"Dataset {dataset_name} exists but missing required structure. Will download from ClearML.")
    
    # Try to get the latest version from ClearML
    try:
        dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True)
        if dataset is None:
            print(f"Dataset {dataset_name} not found in ClearML")
            return False
        
        print(f"Downloading dataset {dataset_name} from ClearML...")
        
        # Create the directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)
        
        # Download to a temporary location first
        temp_path = dataset.get_local_copy()
        
        # Move contents from temp location to desired location
        for item in os.listdir(temp_path):
            s = os.path.join(temp_path, item)
            d = os.path.join(local_path, item)
            if os.path.isdir(s):
                shutil.move(s, d)
            else:
                shutil.copy2(s, d)
        
        # Clean up temporary directory
        shutil.rmtree(temp_path)
        
        print(f"Dataset downloaded successfully to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False 