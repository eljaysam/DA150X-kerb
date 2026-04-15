import os
import cv2
import shutil
import kagglehub
import numpy as np
import splitfolders  # Requires: pip install split-folders

# --- CONFIGURATION ---
RESOLUTIONS = {
    "YOLO_v10": (640, 640),
    "EFFICIENTNET": (600, 600),
    "MLLM_DATA": (600, 600)
}

def download_raw_data():
    """Downloads the latest version of the brain tumor MRI dataset."""
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print(f"Dataset downloaded to: {path}")
    return path

def process_and_resize(img_path, target_size):
    """Loads, resizes, and normalizes an image."""
    # 1. Load in grayscale (Standard for MRI)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 2. Geometric Resizing (INTER_AREA is best for shrinking medical images)
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # 3. Intensity Normalization (Min-Max Scaling to 0-255 range)
    img_float = resized.astype(np.float32)
    norm = (img_float - np.min(img_float)) / (np.max(img_float) - np.min(img_float) + 1e-8)
    final_img = (norm * 255).astype(np.uint8)
    
    return final_img

def build_processed_dataset(raw_path):
    """Groups and resizes images into model-specific directories."""
    training_src = os.path.join(raw_path, "Training")
    
    mapping = {
        "Tumor": ["glioma", "meningioma", "pituitary"],
        "No_Tumor": ["notumor"]
    }

    # Process for each model's required resolution
    for model_name, size in RESOLUTIONS.items():
        print(f"Creating processed dataset for {model_name} ({size[0]}x{size[1]})...")
        temp_dir = f"./temp_{model_name}"
        
        for target_class, source_folders in mapping.items():
            target_path = os.path.join(temp_dir, target_class)
            os.makedirs(target_path, exist_ok=True)
            
            for folder in source_folders:
                source_folder_path = os.path.join(training_src, folder)
                if os.path.exists(source_folder_path):
                    for img_name in os.listdir(source_folder_path):
                        src_file = os.path.join(source_folder_path, img_name)
                        
                        # Process: Resize and Normalize
                        processed_img = process_and_resize(src_file, size)
                        
                        if processed_img is not None:
                            # Save to temp directory
                            new_name = f"{folder}_{img_name}"
                            cv2.imwrite(os.path.join(target_path, new_name), processed_img)

        # Split into Train, Val, Test (80/10/10)
        output_dir = f"./mri_data_{model_name}"
        splitfolders.ratio(temp_dir, output=output_dir, seed=42, ratio=(.8, .1, .1))
        
        # Cleanup temp storage
        shutil.rmtree(temp_dir)
        print(f"Finished {model_name}. Files located in: {output_dir}")

def main():
    # 1. Download raw data
    raw_dataset_path = download_raw_data()
    
    # 2. Process, Resize, and Split for all models
    build_processed_dataset(raw_dataset_path)
    
    print("\nAll datasets are standardized and ready for training!")

if __name__ == "__main__":
    main()