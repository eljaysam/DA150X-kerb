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

    # 3. Train the model (with ALL augmentations disabled)
    print("Starting YOLO training with augmentations turned OFF...")
    results = model.train(
        data=DATA_DIR,          
        epochs=EPOCHS,          
        imgsz=IMG_SIZE,         
        batch=16,               
        patience=3,             
        project='yolo_tumor',   
        name='run_no_aug',      # Changed the name so you don't overwrite your first run!
        
        # --- NEW: Explicitly Disable All Augmentations ---
        fliplr=0.0,       # 0% chance of horizontal flip
        flipud=0.0,       # 0% chance of vertical flip
        degrees=0.0,      # 0 degree image rotation
        scale=0.0,        # 0% image scaling/zooming
        translate=0.0,    # 0% image shifting
        hsv_h=0.0,        # Disable hue (color) changes
        hsv_s=0.0,        # Disable saturation changes
        hsv_v=0.0,        # Disable value (brightness) changes
        erasing=0.0,      # Disable random pixel dropping
        mosaic=0.0        # Disable YOLO's signature image-mixing
    )
    
    print("\nTraining complete!")
    print("Model, metrics, and training curve graphs have been automatically saved in: yolo_tumor/run_1/")

if __name__ == "__main__":
    main()