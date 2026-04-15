import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- NEW: Import the Vertex AI Enterprise SDK ---
import vertexai
from vertexai.generative_models import GenerativeModel, Image

# 1. Configuration
# REPLACE THIS with the exact Project ID from your Google Cloud Console!
PROJECT_ID = "your-project-id-here"
LOCATION = "us-central1" # Standard robust server region

# Initialize Vertex AI (This automatically uses your gcloud authentication)
vertexai.init(project=PROJECT_ID, location=LOCATION)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "mri_data_MLLM_DATA", "test"))

def main():
    print("Loading Vertex AI Generative Model...")
    
    # Using the enterprise endpoint
    model = GenerativeModel("gemini-2.5-flash") 
    
    prompt = (
        "You are an expert radiologist AI. Look at this brain MRI scan. "
        "Is there a tumor present? Answer strictly with the word 'Yes' or 'No'. "
        "Do not provide any other explanation."
    )

    y_true = []
    y_pred = []
    class_names = ["No_Tumor", "Tumor"]

    print("Starting Vertex AI Zero-Shot Evaluation on the Test Set...")
    
    for label_idx, class_name in enumerate(class_names):
        folder_path = os.path.join(TEST_DIR, class_name)
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # --- NEW: Vertex AI uses its own Image loader instead of PIL ---
                img = Image.load_from_file(img_path)
                
                # Send Image + Prompt to Vertex AI
                response = model.generate_content([prompt, img])
                answer = response.text.strip().lower()
                
                # Parse the response
                if 'yes' in answer:
                    prediction = 1
                else:
                    prediction = 0
                
                y_true.append(label_idx)
                y_pred.append(prediction)
                
                print(f"Actual: {class_name} | Gemini (Vertex) Says: {answer}")
                
                # We can lower the sleep timer drastically!
                # Vertex AI handles much higher limits (e.g., 60-150 RPM by default)
                # 1 second is enough to ensure stability without taking hours.
                time.sleep(1) 

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # 3. Generate Results
    print("\n" + "="*50)
    print(" VERTEX AI (GEMINI) PERFORMANCE REPORT ")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Gemini Flash (Vertex AI) Confusion Matrix')
    plt.ylabel('Actual Truth (Doctor)')
    plt.xlabel('AI Prediction')
    
    plt.savefig('gemini_vertex_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved as 'gemini_vertex_confusion_matrix.png'!")

if __name__ == "__main__":
    main()