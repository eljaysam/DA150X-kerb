from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, models
import os

# 1. Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "mri_data_EFFICIENTNET"))

# Verify the dataset directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory not found at: {DATA_DIR}\nMake sure 'mri_data_EFFICIENTNET' exists in the parent directory.")

IMG_SIZE = (600, 600)
BATCH_SIZE = 4 # Kept very small because B7 requires massive amounts of RAM/VRAM
EPOCHS = 25

def build_model():
    # 2. Load the base EfficientNetB7 model
    # We use 'imagenet' weights for transfer learning, but exclude the top layer
    # so we can add our own "Tumor vs No_Tumor" binary output.
    base_model = EfficientNetB7(
        weights='imagenet', 
        include_top=False, 
        input_shape=(600, 600, 3) 
    )
    
    # Freeze the base model so we don't destroy the pre-trained weights
    base_model.trainable = False

    # 3. Add our custom Classification Head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2), # Helps prevent overfitting
        layers.Dense(1, activation='sigmoid') # 'sigmoid' outputs a probability between 0 and 1
    ])

    # 4. Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy', # Standard loss function for Yes/No classification
        metrics=['accuracy', tf.keras.metrics.Recall(name='sensitivity')]
    )
    
    return model

def main():
    print("Loading datasets...")
    
    # 5. Load the training data directly from your pre-processed folders
    # Note: Even though MRI is grayscale, EfficientNet expects 3 color channels (RGB).
    # color_mode='rgb' automatically duplicates your grayscale channel 3 times.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode='rgb' 
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'val'),
        shuffle=False,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode='rgb'
    )

    # 6. Build and train the model
    print("Building EfficientNetB7 model...")
    model = build_model()

    # --- NEW: Define the Early Stopping Rule ---
    early_stop = EarlyStopping(
        monitor='val_loss',        # Watch the validation error
        patience=3,                # Wait 3 epochs before stopping
        restore_best_weights=True  # Automatically keep the best version!
    )
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop] # Attatched early stopping rule
    )
    
    # 7. Save the trained model
    model.save("efficientnet_b7_brain_tumor.keras")
    print("Model saved successfully!")

    # --- NEW: Generate Visual Learning Curves ---
    plot_training_history(history)




# --- NEW: The Graphing Function ---
def plot_training_history(history):
    """Plots the accuracy and loss curves and saves them as an image."""
    
    # Extract the data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_run = range(1, len(acc) + 1)

    # Create a wide figure with two subplots side-by-side
    plt.figure(figsize=(14, 5))

    # Plot 1: Accuracy (Higher is better)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_run, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs_run, val_acc, label='Validation Accuracy', color='orange', linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot 2: Loss (Lower is better)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_run, loss, label='Training Loss', color='red')
    plt.plot(epochs_run, val_loss, label='Validation Loss', color='green', linestyle='--')
    plt.title('Model Loss (Error Rate)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the graph as a PNG file so you can put it in your LaTeX report!
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved as 'training_curves.png'")

if __name__ == "__main__":
    main()