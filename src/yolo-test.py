from ultralytics import YOLO
import os

# Point to your trained YOLO model (Update the path if needed)
MODEL_PATH = "yolo_tumor/run_1/weights/best.pt"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mri_data_YOLO_v10"))

def main():
    print("Loading trained YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Running evaluation on the unseen Test Set...")
    # The 'split="test"' argument tells it to ignore train/val folders
    metrics = model.val(data=DATA_DIR, split='test', project='yolo_tumor', name='test_results')
    
    print("\nTesting complete! Look inside the 'yolo_tumor/test_results/' folder.")
    print("Your Confusion Matrix and metric tables have been automatically generated as PNGs!")

if __name__ == "__main__":
    main()