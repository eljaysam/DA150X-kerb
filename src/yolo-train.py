from ultralytics import YOLO
import os

# 1. Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the specific YOLO folder your format-datasets.py script created
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "mri_data_YOLO_v10"))

# Verify the dataset directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory not found at: {DATA_DIR}")

# Your format-datasets.py set YOLO resolution to 640x640
IMG_SIZE = 640 
EPOCHS = 25

def main():
    print("Loading YOLOv8 Classification Model...")
    
    # 2. Load the base YOLO Classification model
    # The '.pt' file will automatically download from Ultralytics the first time you run this
    model = YOLO('yolov8s-cls.pt')

    # 3. Train the model
    # Ultralytics natively expects the "root/train/class" folder structure your formatter built!
    print("Starting YOLO training...")
    results = model.train(
        data=DATA_DIR,          # Path to the root dataset folder
        epochs=EPOCHS,          # Maximum epochs (25)
        imgsz=IMG_SIZE,         # Image size (640)
        batch=16,               # YOLO is VRAM efficient; it can easily handle batch size 16
        patience=3,             # Built-in EARLY STOPPING! Stops if no improvement for 3 epochs
        project='yolo_tumor',   # Folder name to save results
        name='run_1'            # Sub-folder for this specific training attempt
    )
    
    print("\nTraining complete!")
    print("Model, metrics, and training curve graphs have been automatically saved in: yolo_tumor/run_1/")

if __name__ == "__main__":
    main()