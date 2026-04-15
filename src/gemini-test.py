import os
import time
import google.generativeai as genai
import PIL.Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dotenv import load_dotenv # <--- Add this

# Load the hidden variables from the .env file
load_dotenv() # <--- Add this

# 1. Configuration & API Setup
# You must get a free API key from Google AI Studio and set it in your environment
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key) 

# Point to the 600x600 MLLM Test folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "mri_data_MLLM_DATA", "test"))

def main():
    print("Loading Gemini API...")
    
    # Initialize the model. Update the string to the exact version you are testing.
    # Note: Use the official API model string for the Flash model you are targeting.
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    # The strict prompt forcing a binary "Yes" or "No"
    prompt = (
        "You are an expert radiologist AI. Look at this brain MRI scan. "
        "Is there a tumor present? Answer strictly with the word 'Yes' or 'No'. "
        "Do not provide any other explanation."
    )

    y_true = []
    y_pred = []
    class_names = ["No_Tumor", "Tumor"]

    print("Starting Zero-Shot Evaluation on the Test Set...")
    
    # 2. Iterate through the exact same Test folders your CNNs used
    for label_idx, class_name in enumerate(class_names):
        folder_path = os.path.join(TEST_DIR, class_name)
        
        # Note: If your test set is massive, consider testing a subset (e.g., [:100]) to respect API rate limits
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # Load image
                img = PIL.Image.open(img_path)
                
                # Send Image + Prompt to Gemini
                response = model.generate_content([prompt, img])
                answer = response.text.strip().lower()
                
                # Parse the response
                if 'yes' in answer:
                    prediction = 1
                else:
                    prediction = 0
                
                y_true.append(label_idx)
                y_pred.append(prediction)
                
                print(f"Actual: {class_name} | Gemini Says: {answer}")
                
                # Crucial: Sleep for 3-5 seconds to avoid hitting API rate limits
                time.sleep(3) 

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # 3. Generate Results (Identical logic to your CNN scripts)
    print("\n" + "="*50)
    print(" GEMINI ZERO-SHOT PERFORMANCE REPORT ")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Using a different color (Purples) to visually distinguish MLLM graphs from CNN graphs in your report
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Gemini Flash Confusion Matrix (Zero-Shot)')
    plt.ylabel('Actual Truth (Doctor)')
    plt.xlabel('AI Prediction')
    
    plt.savefig('gemini_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved as 'gemini_confusion_matrix.png'!")

if __name__ == "__main__":
    main()