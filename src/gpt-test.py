import os
import time
import base64
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Secure Authentication
# This automatically finds your .env file and loads the key securely
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in the .env file. Please check Phase 2!")

# Initialize the Native OpenAI Client
client = OpenAI(api_key=api_key)

# 2. Dataset Configuration
# Pointing to the exact 600x600 Test Folder used by Gemini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "mri_data_MLLM_DATA", "test"))

# Function to encode local images to Base64 (Required by OpenAI API)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    # Targeted model updated based on your specifications
    MODEL_NAME = "gpt-5.4-mini" 
    print(f"Loading Native OpenAI API ({MODEL_NAME})...")
    
    prompt = (
        "You are an expert radiologist AI. Look at this brain MRI scan. "
        "Is there a tumor present? Answer strictly with the word 'Yes' or 'No'. "
        "Do not provide any other explanation."
    )

    y_true = []
    y_pred = []
    class_names = ["No_Tumor", "Tumor"]

    # Progress tracking variables
    total_images = 560
    current_count = 0

    print("Starting Zero-Shot Evaluation...")
    
    for label_idx, class_name in enumerate(class_names):
        folder_path = os.path.join(TEST_DIR, class_name)
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            try:
                base64_image = encode_image(img_path)
                current_count += 1
                
                # 3. Send the Payload directly to OpenAI
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_completion_tokens=10 
                )
                
                answer = response.choices[0].message.content.strip().lower()
                
                if 'yes' in answer:
                    prediction = 1
                else:
                    prediction = 0
                
                y_true.append(label_idx)
                y_pred.append(prediction)
                
                # Prints progress: e.g., [1/560] Actual: Tumor | GPT-5.4-mini Says: yes
                print(f"[{current_count}/{total_images}] Actual: {class_name} | {MODEL_NAME} Says: {answer}")
                
                # 1-second sleep keeps you perfectly within the Tier 1 limits (60 RPM)
                time.sleep(1) 

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # 4. Generate the Final Report and Graphs
    print("\n" + "="*50)
    print(f" {MODEL_NAME.upper()} ZERO-SHOT REPORT ")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Using 'Greens' for OpenAI to contrast with Gemini (Purples) and CNNs (Blues)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{MODEL_NAME} Confusion Matrix (Zero-Shot)')
    plt.ylabel('Actual Truth (Doctor)')
    plt.xlabel('AI Prediction')
    
    plt.savefig('gpt_native_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved as 'gpt_native_confusion_matrix.png'!")

if __name__ == "__main__":
    main()