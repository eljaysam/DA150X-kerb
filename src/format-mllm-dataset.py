import argparse
import os
import shutil

import cv2
import kagglehub
import numpy as np
import splitfolders


MLLM_RESOLUTION = (600, 600)
OUTPUT_DIR = "./mri_data_MLLM_DATA"
TEMP_DIR = "./temp_MLLM_DATA"


def download_raw_data():
    """Downloads the latest version of the brain tumor MRI dataset."""
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print(f"Dataset downloaded to: {path}")
    return path


def process_and_resize(img_path, target_size):
    """Loads, resizes, and normalizes an image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    img_float = resized.astype(np.float32)
    norm = (img_float - np.min(img_float)) / (np.max(img_float) - np.min(img_float) + 1e-8)
    final_img = (norm * 255).astype(np.uint8)

    return final_img


def build_mllm_dataset(raw_path):
    """Builds only the MLLM dataset in 600x600 resolution."""
    training_src = os.path.join(raw_path, "Training")

    mapping = {
        "Tumor": ["glioma", "meningioma", "pituitary"],
        "No_Tumor": ["notumor"],
    }

    print(f"Creating MLLM dataset ({MLLM_RESOLUTION[0]}x{MLLM_RESOLUTION[1]})...")

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    for target_class, source_folders in mapping.items():
        target_path = os.path.join(TEMP_DIR, target_class)
        os.makedirs(target_path, exist_ok=True)

        for folder in source_folders:
            source_folder_path = os.path.join(training_src, folder)
            if not os.path.exists(source_folder_path):
                continue

            for img_name in os.listdir(source_folder_path):
                src_file = os.path.join(source_folder_path, img_name)
                processed_img = process_and_resize(src_file, MLLM_RESOLUTION)

                if processed_img is None:
                    continue

                new_name = f"{folder}_{img_name}"
                cv2.imwrite(os.path.join(target_path, new_name), processed_img)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    splitfolders.ratio(TEMP_DIR, output=OUTPUT_DIR, seed=42, ratio=(0.8, 0.1, 0.1))
    shutil.rmtree(TEMP_DIR)

    print(f"Finished MLLM_DATA. Files located in: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate only the MLLM MRI dataset at 600x600 resolution."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help="Path to already-downloaded raw dataset root (contains Training/).",
    )
    args = parser.parse_args()

    raw_dataset_path = args.raw_path if args.raw_path else download_raw_data()
    build_mllm_dataset(raw_dataset_path)

    print("MLLM dataset regeneration complete.")


if __name__ == "__main__":
    main()
