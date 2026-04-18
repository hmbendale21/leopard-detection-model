import cv2
import numpy as np
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'yolo11n.pt'  # Nano model (very fast)
CONF_THRESHOLD = 0.5       # Probability required to show a box
IOU_THRESHOLD = 0.45       # NMS threshold
CAMERA_INDEX = 0

# Class IDs for YOLOv11 (MS COCO)
# 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 0: person
TARGET_CLASS = 15          # We use 'cat' as the proxy for leopard in standard YOLO
REJECT_CLASSES = [16, 17, 18, 19, 0] # Dog, Horse, Sheep, Cow, Person (Explicitly Ignore)

def run_detection():
    print(f"--- Loading High-Accuracy YOLO11 Brain ({MODEL_PATH}) ---")
    model = YOLO(MODEL_PATH)
    
    capture = cv2.VideoCapture(CAMERA_INDEX)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # EMA Smoothing state
    smoothed_box = None
    smoothing_factor = 0.6
    
    print("Camera active. Press 'q' to quit.")
    
    while capture.isOpened():
        stime = time.time()
        ret, frame = capture.read()
        if not ret:
            break
            
        # Run inference
        results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        leopard_detected = False
        
        for result in results[0]:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # --- ACCURACY FILTER ---
                # 1. If it's a known non-leopard animal, we REJECT it entirely
                if cls_id in REJECT_CLASSES:
                    continue
                
                # 2. Only accept the feline (leopard proxy) class
                if cls_id != TARGET_CLASS:
                    continue
                    
                leopard_detected = True
                
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                curr_tl = [int(xyxy[0]), int(xyxy[1])]
                curr_br = [int(xyxy[2]), int(xyxy[3])]
                
                # --- EMA SMOOTHING ---
                if smoothed_box is None:
                    smoothed_box = [curr_tl, curr_br]
                else:
                    smoothed_box[0][0] = int(smoothed_box[0][0] * (1 - smoothing_factor) + curr_tl[0] * smoothing_factor)
                    smoothed_box[0][1] = int(smoothed_box[0][1] * (1 - smoothing_factor) + curr_tl[1] * smoothing_factor)
                    smoothed_box[1][0] = int(smoothed_box[1][0] * (1 - smoothing_factor) + curr_br[0] * smoothing_factor)
                    smoothed_box[1][1] = int(smoothed_box[1][1] * (1 - smoothing_factor) + curr_br[1] * smoothing_factor)
                
                tl = tuple(smoothed_box[0])
                br = tuple(smoothed_box[1])
                
                # Draw the UI
                cv2.rectangle(frame, tl, br, (0, 255, 0), 7)
                cv2.putText(frame, f"leopard ({conf:.2f})", (tl[0], tl[1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
                print(f"Detected: LEOPARD (Conf: {conf:.2f})")
                break # We pin to the best single feline detection
        
        # If no leopard found, we slowly fade out the smoothed box
        if not leopard_detected:
            smoothed_box = None
            
        cv2.imshow('Leopard Detection (YOLO11 High Precision)', frame)
        
        # Print speed stats
        fps = 1 / (time.time() - stime + 1e-6)
        # print(f'FPS: {fps:.1f}')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()
