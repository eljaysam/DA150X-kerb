import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "mri_data_EFFICIENTNET"))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "efficientnet_b7_brain_tumor.keras"))
IMG_SIZE = (600, 600)
BATCH_SIZE = 4

def main():
    print("1. Loading the trained EfficientNetB7 model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("2. Loading the Test Dataset...")
    # CRITICAL: shuffle=False! We must keep the images in exact order to match them with their labels
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'test'),
        shuffle=False, 
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode='rgb'
    )

    # Extract the actual true labels from the folders (0 for No_Tumor, 1 for Tumor)
    class_names = test_ds.class_names
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    print("3. AI is taking the final exam (Predicting on test data)...")
    # Get raw probability scores from the model (e.g., 0.85, 0.12)
    predictions = model.predict(test_ds)
    
    # Convert probabilities to strict Yes/No (1 or 0). Anything > 50% is a Tumor.
    y_pred = (predictions > 0.5).astype(int).flatten()

    # --- RESULTS FORMULATION ---
    
    print("\n" + "="*50)
    print(" FINAL PERFORMANCE REPORT (TABLE) ")
    print("="*50)
    # This generates a beautiful text table with Accuracy, Precision, Recall, and F1-Score
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # --- CONFUSION MATRIX GENERATION ---
    print("\nGenerating Confusion Matrix Image...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Seaborn makes the matrix look modern and colorful
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('EfficientNetB7 Confusion Matrix (Test Set)')
    plt.ylabel('Actual Truth (Doctor)')
    plt.xlabel('AI Prediction')
    
    # Save the image for your LaTeX document
    plt.savefig('efficientnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved as 'efficientnet_confusion_matrix.png'!")

if __name__ == "__main__":
    main()