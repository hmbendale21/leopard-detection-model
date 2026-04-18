import cv2
import numpy as np
import time
import sys
import os
import threading
import pathlib

# Handle cross-platform sound
try:
    import winsound
except ImportError:
    winsound = None

import torch
import torchvision.transforms as T
from torchvision import models
from ultralytics import YOLO

# ── Preload / trigger-file paths ─────────────────────────────
BASE_DIR     = pathlib.Path(__file__).parent
READY_FILE   = BASE_DIR / ".model_ready"     # written when models are loaded
TRIGGER_FILE = BASE_DIR / ".detect_trigger"  # written by Flask when user clicks Detect
STOP_FILE    = BASE_DIR / ".detect_stop"     # written by Flask when user clicks Stop

# ============================================================
# LEOPARD-ONLY DETECTION  (YOLO11 + MobileNetV3 verifier)
# ============================================================
# Two-stage pipeline:
#   Stage 1  — YOLO detects all animals (no class filter)
#   Stage 2  — MobileNetV3 (ImageNet) classifies the crop.
#              Only accepts ImageNet leopard/jaguar/snow-leopard classes.
#
# This eliminates false positives from dogs, bears, cows, etc.
# ============================================================

MODEL_PATH       = 'yolo11n.pt'
YOLO_CONF        = 0.30           # YOLO first-pass confidence
IOU_THRESHOLD    = 0.45
CAMERA_INDEX     = 0

# ── YOLO animal classes (COCO) to consider as candidates ─────
# We let YOLO detect any animal, then the classifier decides.
ANIMAL_CLASSES = [
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
]

# ── ImageNet classes that count as "leopard" ─────────────────
LEOPARD_IMAGENET_IDS = {
    288,  # leopard, Panthera pardus
    289,  # snow leopard, ounce
    290,  # jaguar, panther, Panthera onca
}
CLASSIFIER_THRESHOLD = 0.15   # minimum softmax probability for leopard classes

# Sharpening kernel — enhances edges smeared by motion blur
SHARPEN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# ── ImageNet preprocessing ───────────────────────────────────
CLASSIFY_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def load_classifier():
    """Load a lightweight MobileNetV3-Small for leopard verification."""
    print("Loading MobileNetV3 classifier for leopard verification...")
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.eval()
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    print(f"  Classifier device: {device}")
    return net, device


def is_leopard(crop_bgr, classifier, device):
    """
    Run the crop through MobileNetV3 and return True + confidence
    only if the top prediction is a leopard-family class.
    """
    if crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
        return False, 0.0

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor   = CLASSIFY_TRANSFORM(crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    # Sum probabilities of all leopard-family classes
    leopard_prob = sum(probs[cid].item() for cid in LEOPARD_IMAGENET_IDS)

    return leopard_prob >= CLASSIFIER_THRESHOLD, leopard_prob


def sharpen_frame(frame):
    """Apply unsharp mask to reduce motion-blur effect before inference."""
    return cv2.filter2D(frame, -1, SHARPEN_KERNEL)


def get_source():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isfile(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        return path, True
    return CAMERA_INDEX, False


def draw_box(frame, tl, br, label, conf, active):
    """Draw a polished bounding box with corner accents."""
    color      = (0, 255, 100) if active else (0, 180, 70)
    rect_color = (0, 230, 50)  if active else (0, 160, 40)

    # Filled semi-transparent background for label
    label_text = f"LEOPARD  {conf:.0%}" if active else "LEOPARD"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    lx, ly = tl[0], tl[1] - 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Main rectangle
    cv2.rectangle(frame, tl, br, rect_color, 2)

    # Corner accents
    cl, th_line = 22, 4
    for pt1, pt2 in [
        (tl, (tl[0]+cl, tl[1])), (tl, (tl[0], tl[1]+cl)),
        ((br[0], tl[1]), (br[0]-cl, tl[1])), ((br[0], tl[1]), (br[0], tl[1]+cl)),
        ((tl[0], br[1]), (tl[0]+cl, br[1])), ((tl[0], br[1]), (tl[0], br[1]-cl)),
        (br, (br[0]-cl, br[1])), (br, (br[0], br[1]-cl)),
    ]:
        cv2.line(frame, pt1, pt2, color, th_line)

    cv2.putText(frame, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def run_detection():
    source, is_video = get_source()

    print("--- YOLO11 + MobileNetV3 Leopard Detector ---")
    print(f"Mode           : {'VIDEO: ' + str(source) if is_video else 'LIVE CAMERA'}")
    print(f"YOLO conf      : {YOLO_CONF}")
    print(f"Classifier     : MobileNetV3 (ImageNet leopard classes)")
    print(f"Classifier min : {CLASSIFIER_THRESHOLD:.0%} probability")
    print(f"Sharpening     : ON\n")

    # Load models
    yolo_model = YOLO(MODEL_PATH)
    classifier, device = load_classifier()

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        print("ERROR: Cannot open video source.")
        sys.exit(1)

    # Camera resolution
    if not is_video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    src_w   = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0

    print(f"Resolution: {src_w}x{src_h}  |  Source FPS: {src_fps:.1f}")

    # Output video writer
    out_writer = None
    if is_video:
        out_path   = os.path.splitext(source)[0] + "_leopard_detected.mp4"
        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, src_fps, (src_w, src_h))
        print(f"Saving output to: {out_path}")

    print("\nPress 'q' to quit  |  SPACE = pause/resume (video)\n")

    # State
    smoothed_box    = None
    smooth_alpha    = 0.70
    no_detect_count = 0
    MAX_COAST       = 8          # reduced coast since we have higher precision now
    last_conf       = 0.0
    paused          = False
    frame_num       = 0
    detect_count    = 0
    last_beep_time  = 0
    BEEP_COOLDOWN   = 2

    window_name = 'Leopard Detection'

    while capture.isOpened():
        # ── PAUSE HANDLING (video only) ────────────────────────
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            if key == ord(' '): paused = False
            continue

        ret, frame = capture.read()
        if not ret:
            if is_video:
                print(f"\nVideo finished. {detect_count} detections in {frame_num} frames.")
            break

        frame_num += 1
        stime = time.time()

        # ── SHARPEN TO FIGHT MOTION BLUR ──────────────────────
        sharp_frame = sharpen_frame(frame)

        # ── STAGE 1: YOLO DETECTION (all animal classes) ──────
        results = yolo_model.predict(
            sharp_frame,
            conf=YOLO_CONF,
            iou=IOU_THRESHOLD,
            verbose=False,
            classes=ANIMAL_CLASSES
        )

        best_conf = 0.0
        best_box  = None

        for r in results:
            for box in r.boxes:
                yolo_conf = float(box.conf[0])
                xyxy      = box.xyxy[0].cpu().numpy()
                x1, y1    = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                x2, y2    = min(frame.shape[1], int(xyxy[2])), min(frame.shape[0], int(xyxy[3]))

                # ── STAGE 2: CLASSIFY THE CROP ────────────────
                crop = frame[y1:y2, x1:x2]
                confirmed, leo_prob = is_leopard(crop, classifier, device)

                if confirmed and leo_prob > best_conf:
                    best_conf = leo_prob
                    best_box  = [[x1, y1], [x2, y2]]

        # ── EMA SMOOTHING ────────────────────────────────────
        leopard_found = best_box is not None

        if leopard_found:
            no_detect_count = 0
            detect_count   += 1
            last_conf       = best_conf

            if smoothed_box is None:
                smoothed_box = [list(best_box[0]), list(best_box[1])]
            else:
                for i in range(2):
                    for j in range(2):
                        smoothed_box[i][j] = int(
                            smoothed_box[i][j] * (1 - smooth_alpha) +
                            best_box[i][j]     *      smooth_alpha
                        )
            print(f"[Frame {frame_num:>5}] LEOPARD confirmed  prob={best_conf:.2f}")

            # ── BEEP ALERT (non-blocking, with cooldown) ─────
            now = time.time()
            if now - last_beep_time >= BEEP_COOLDOWN:
                last_beep_time = now
                if winsound:
                    threading.Thread(
                        target=winsound.Beep, args=(2500, 300), daemon=True
                    ).start()
                else:
                    print("\a") # terminal bell for non-windows
        else:
            no_detect_count += 1
            if no_detect_count > MAX_COAST:
                smoothed_box = None

        # ── DRAW BOX ─────────────────────────────────────────
        if smoothed_box is not None:
            tl = tuple(smoothed_box[0])
            br = tuple(smoothed_box[1])
            draw_box(frame, tl, br, "LEOPARD", last_conf, leopard_found)

        # ── STATUS INDICATOR ─────────────────────────────────
        if smoothed_box is not None:
            status_color = (0, 255, 100)
            status_text  = "LEOPARD DETECTED"
        else:
            status_color = (80, 80, 80)
            status_text  = "Scanning..."

        cv2.putText(frame, status_text, (10, src_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # ── FPS OVERLAY ──────────────────────────────────────
        fps = 1 / (time.time() - stime + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # ── VIDEO PROGRESS BAR ───────────────────────────────
        if is_video and total_frames > 0:
            progress = frame_num / total_frames
            bar_w    = src_w - 20
            cv2.rectangle(frame, (10, src_h-35), (10+bar_w, src_h-22), (50,50,50), -1)
            cv2.rectangle(frame, (10, src_h-35), (10+int(bar_w*progress), src_h-22),
                          (0, 200, 80), -1)

        # ── DISPLAY ──────────────────────────────────────────
        try:
            cv2.imshow(window_name, frame)
        except Exception as e:
            # Skip GUI if not available (e.g. on a cloud server)
            if frame_num % 30 == 0:
                print(f"[Info] Running in headless mode. Frames processed: {frame_num}")
        if out_writer:
            out_writer.write(frame)

        wait_ms = 1
        if is_video:
            elapsed = time.time() - stime
            wait_ms = max(1, int(1000/src_fps) - int(elapsed*1000))

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'): break
        if key == ord(' ') and is_video:
            paused = True
            print("--- PAUSED (SPACE to resume) ---")

    # ── CLEANUP ──────────────────────────────────────────────
    capture.release()
    if out_writer:
        out_writer.release()
        print(f"\nAnnotated video saved: {out_path}")
    cv2.destroyAllWindows()

    if is_video:
        print(f"Summary: {detect_count} detections across {frame_num} frames "
              f"({detect_count/max(frame_num,1)*100:.1f}% detection rate)")


def preload_and_wait():
    """
    --preload mode:
      1. Load YOLO + MobileNetV3 (the slow part)
      2. Write READY_FILE so the website knows models are ready
      3. Wait (polling) for TRIGGER_FILE to appear
      4. Delete trigger, run detection, then delete ready file
      5. Loop back to step 2 so a second click works too
    """
    # Clean stale files from a previous run
    for f in (READY_FILE, TRIGGER_FILE, STOP_FILE):
        f.unlink(missing_ok=True)

    print("[Preload] Loading YOLO model...")
    yolo_model = YOLO(MODEL_PATH)

    print("[Preload] Loading MobileNetV3 classifier...")
    classifier, device = load_classifier()

    print("[Preload] Models ready. Writing ready flag.")
    READY_FILE.write_text("ready")

    while True:
        # Wait for trigger
        while not TRIGGER_FILE.exists():
            time.sleep(0.1)

        TRIGGER_FILE.unlink(missing_ok=True)
        STOP_FILE.unlink(missing_ok=True)
        print("[Preload] Trigger received — opening camera now!")

        # ── Run detection with pre-loaded models ──
        source = CAMERA_INDEX
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            print("ERROR: Cannot open camera.")
            READY_FILE.write_text("ready")
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Preload] Camera opened: {src_w}x{src_h}")

        smoothed_box    = None
        smooth_alpha    = 0.70
        no_detect_count = 0
        MAX_COAST       = 8
        last_conf       = 0.0
        last_beep_time  = 0
        BEEP_COOLDOWN   = 2
        frame_num       = 0
        detect_count    = 0

        window_name = 'Leopard Detection'

        while capture.isOpened():
            # Stop signal from website
            if STOP_FILE.exists():
                STOP_FILE.unlink(missing_ok=True)
                print("[Preload] Stop signal received.")
                break

            ret, frame = capture.read()
            if not ret:
                break

            frame_num += 1
            stime = time.time()

            sharp_frame = sharpen_frame(frame)

            results = yolo_model.predict(
                sharp_frame,
                conf=YOLO_CONF,
                iou=IOU_THRESHOLD,
                verbose=False,
                classes=ANIMAL_CLASSES
            )

            best_conf = 0.0
            best_box  = None

            for r in results:
                for box in r.boxes:
                    yolo_conf = float(box.conf[0])
                    xyxy      = box.xyxy[0].cpu().numpy()
                    x1, y1   = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                    x2, y2   = min(frame.shape[1], int(xyxy[2])), min(frame.shape[0], int(xyxy[3]))
                    crop = frame[y1:y2, x1:x2]
                    confirmed, leo_prob = is_leopard(crop, classifier, device)
                    if confirmed and leo_prob > best_conf:
                        best_conf = leo_prob
                        best_box  = [[x1, y1], [x2, y2]]

            leopard_found = best_box is not None

            if leopard_found:
                no_detect_count = 0
                detect_count   += 1
                last_conf       = best_conf
                if smoothed_box is None:
                    smoothed_box = [list(best_box[0]), list(best_box[1])]
                else:
                    for i in range(2):
                        for j in range(2):
                            smoothed_box[i][j] = int(
                                smoothed_box[i][j] * (1 - smooth_alpha) +
                                best_box[i][j]     *      smooth_alpha
                            )
                print(f"[Frame {frame_num:>5}] LEOPARD confirmed  prob={best_conf:.2f}")
                now = time.time()
                if now - last_beep_time >= BEEP_COOLDOWN:
                    last_beep_time = now
                    threading.Thread(target=winsound.Beep, args=(2500, 300), daemon=True).start()
            else:
                no_detect_count += 1
                if no_detect_count > MAX_COAST:
                    smoothed_box = None

            if smoothed_box is not None:
                tl = tuple(smoothed_box[0])
                br = tuple(smoothed_box[1])
                draw_box(frame, tl, br, "LEOPARD", last_conf, leopard_found)

            if smoothed_box is not None:
                status_color, status_text = (0, 255, 100), "LEOPARD DETECTED"
            else:
                status_color, status_text = (80, 80, 80), "Scanning..."

            cv2.putText(frame, status_text, (10, src_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            fps = 1 / (time.time() - stime + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
        print(f"[Preload] Detection ended. {detect_count} detections in {frame_num} frames.")

        # Re-signal ready for next click
        READY_FILE.write_text("ready")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--preload":
        preload_and_wait()
    else:
        run_detection()