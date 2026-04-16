import os
import sys

print("========================================")
print(" PRE-FLIGHT DIAGNOSTIC CHECK ")
print("========================================\n")

# --- 1. CHECK LIBRARIES ---
print("1. Checking Dependencies...")
try:
    import dotenv
    import openai
    import matplotlib
    import seaborn
    import sklearn
    print("   ✅ SUCCESS: All required libraries are installed.")
except ImportError as e:
    print(f"   ❌ ERROR: Missing library -> {e}")
    print("   Fix: Run `pip install openai python-dotenv matplotlib seaborn scikit-learn`")
    sys.exit()

# --- 2. CHECK .ENV FILE ---
print("\n2. Checking .env Configuration...")
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    # Mask the key so it's safe to print
    masked_key = f"{api_key[:8]}.......{api_key[-4:]}"
    print(f"   ✅ SUCCESS: Found OpenAI API Key ({masked_key})")
else:
    print("   ❌ ERROR: Could not find OPENAI_API_KEY.")
    print("   Fix: Ensure your file is named exactly '.env' and is in the project root.")
    sys.exit()

# --- 3. CHECK IMAGE PATHS ---
print("\n3. Checking Image Directories...")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "mri_data_MLLM_DATA", "test"))

if os.path.exists(TEST_DIR):
    print(f"   ✅ SUCCESS: Found test directory at {TEST_DIR}")
    
    # Count the images
    image_count = 0
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_count += 1
                
    print(f"   ✅ SUCCESS: Counted exactly {image_count} images.")
    if image_count != 560:
        print(f"   ⚠️ WARNING: Expected 560 images, but found {image_count}.")
else:
    print(f"   ❌ ERROR: Could not find directory at {TEST_DIR}")
    sys.exit()

# --- 4. CHECK API CONNECTION ---
print("\n4. Testing OpenAI Connection (Ping)...")
try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Sending a tiny, 1-token test message
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": "Reply with the exact word: Connected."}],
        max_tokens=5
    )
    reply = response.choices[0].message.content.strip()
    print(f"   ✅ SUCCESS: API is active! Server replied: '{reply}'")
except Exception as e:
    print(f"   ❌ ERROR: API Connection failed.")
    print(f"   Details: {e}")
    sys.exit()

print("\n========================================")
print(" ALL SYSTEMS GO! YOU ARE READY TO RUN ")
print("========================================\n")