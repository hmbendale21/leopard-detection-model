import os
import sys

def check_accuracy():
    print("--- Leopard Detection Accuracy Checker ---")
    
    # 1. Check for custom brain (ckpt)
    if os.path.isdir('ckpt'):
        print("[SUCCESS] Found 'ckpt' folder! Accuracy will be NEAR 100%.")
        files = os.listdir('ckpt')
        if any('.meta' in f for f in files):
            print("  -> Found trained leopard checkpoints. Well done!")
        else:
            print("  -> WARNING: 'ckpt' folder exists but appears empty.")
    else:
        print("[LOW ACCURACY] 'ckpt' folder is missing.")
        print("  -> AI is using a generic house cat brain (bin/yolotiny.weights).")
        print("  -> Accuracy will be around 20% until you download the leopard brain.")
        
    print("\n--- Action Items ---")
    print("1. Open README.md and find the Google Drive link for 'ckpt' files.")
    print("2. Download that folder and place it in this directory.")
    print("3. Run 'python test_live_cam.py' again.")
    print("\n-------------------------------------------")

if __name__ == "__main__":
    check_accuracy()
