import os
import kagglehub
import shutil
import splitfolders  # Requires: pip install split-folders

def download_raw_data():
    """Downloads the latest version of the brain tumor MRI dataset."""
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print(f"Dataset downloaded to: {path}")
    return path

def create_binary_structure(raw_path, temp_dir):
    """
    Groups the 4 original classes into binary 'Tumor' vs 'No_Tumor' folders.
    Original: glioma, meningioma, pituitary, notumor
    """
    training_src = os.path.join(raw_path, "Training")
    
    # Define our mapping for the assignment goals
    mapping = {
        "Tumor": ["glioma", "meningioma", "pituitary"],
        "No_Tumor": ["notumor"]
    }

    print("Grouping images into binary classes...")
    for target_class, source_folders in mapping.items():
        target_path = os.path.join(temp_dir, target_class)
        os.makedirs(target_path, exist_ok=True)
        
        for folder in source_folders:
            source_folder_path = os.path.join(training_src, folder)
            if os.path.exists(source_folder_path):
                for img_name in os.listdir(source_folder_path):
                    # Prepend folder name to avoid filename collisions
                    new_name = f"{folder}_{img_name}"
                    shutil.copy(
                        os.path.join(source_folder_path, img_name),
                        os.path.join(target_path, new_name)
                    )

def split_data(temp_dir, output_dir):
    """Splits the binary data into Train, Val, and Test sets (80/10/10)."""
    # seed=42 ensures the split is the same every time you run it
    splitfolders.ratio(temp_dir, output=output_dir, seed=42, ratio=(.8, .1, .1))
    print(f"Data split complete. Files located in: {output_dir}")

def main():
    # 1. Download
    raw_dataset_path = download_raw_data()
    
    # 2. Temporary setup for regrouping
    temp_binary_dir = "./temp_binary_storage"
    if os.path.exists(temp_binary_dir):
        shutil.rmtree(temp_binary_dir)
        
    # 3. Process
    create_binary_structure(raw_dataset_path, temp_binary_dir)
    
    # 4. Final Split (This folder will be used by both YOLO and EfficientNet)
    final_output_dir = "./mri_data_final"
    split_data(temp_binary_dir, final_output_dir)
    
    # 5. Cleanup temporary folder to save space
    shutil.rmtree(temp_binary_dir)
    print("Process finished successfully.")

if __name__ == "__main__":
    main()