"""
Leopard Detection Website — Flask Backend (cloud-deployable)

Architecture:
  - Browser captures webcam frames with getUserMedia (client-side, no server camera needed)
  - Each frame is sent to POST /api/detect_frame as a base64 JPEG
  - Server runs the two-stage pipeline: YOLO11 (animal candidates) -> MobileNetV3 (leopard verification)
  - Server returns a bounding box + confidence; browser draws it on an overlay canvas

Models are loaded once, in a background thread, right after the server boots.
"""

import base64
import pathlib
import threading
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

BASE_DIR = pathlib.Path(__file__).parent

# ============================================================
# LEOPARD-ONLY DETECTION  (YOLO11 + MobileNetV3 verifier)
# ============================================================
MODEL_PATH = str(BASE_DIR / "yolo11n.pt")
YOLO_CONF = 0.30
IOU_THRESHOLD = 0.45

ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]  # COCO: cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe

LEOPARD_IMAGENET_IDS = {288, 289, 290}  # leopard, snow leopard, jaguar
CLASSIFIER_THRESHOLD = 0.15

SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

CLASSIFY_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Shared model state ──────────────────────────────────────
_state_lock = threading.Lock()
_models = {"yolo": None, "classifier": None, "device": None, "ready": False, "error": None}


def load_models():
    global _models

    try:
        print("========== LOADING MODELS ==========")

        # Automatically download YOLO if missing
        # Use explicit path so deployers don't rely on CWD
        if not pathlib.Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"YOLO model file not found at {MODEL_PATH}")
        yolo_model = YOLO(MODEL_PATH)

        print("YOLO Loaded")

        # Load a MobileNetV3 classifier. If pretrained weights cannot be
        # downloaded in the deployment environment, fall back to an
        # uninitialized model and log a warning. For production, bundle
        # the weights or provide a local checkpoint to avoid runtime
        # downloads.
        try:
            classifier = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
        except Exception as ex_weights:
            print("Could not load pretrained MobileNet weights; falling back to weights=None")
            print(ex_weights)
            classifier = models.mobilenet_v3_small(weights=None)

        classifier.eval()

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        classifier.to(device)

        with _state_lock:
            _models["yolo"] = yolo_model
            _models["classifier"] = classifier
            _models["device"] = device
            _models["ready"] = True
            _models["error"] = None

        print("MODELS READY")

    except Exception as e:

        print("MODEL LOAD FAILED")
        traceback.print_exc()

        with _state_lock:
            _models["ready"] = False
            _models["error"] = str(e)   


threading.Thread(target=load_models, daemon=True).start()


def sharpen_frame(frame):
    return cv2.filter2D(frame, -1, SHARPEN_KERNEL)


def is_leopard(crop_bgr, classifier, device):
    if crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
        return False, 0.0
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = CLASSIFY_TRANSFORM(crop_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    leopard_prob = sum(probs[cid].item() for cid in LEOPARD_IMAGENET_IDS)
    return leopard_prob >= CLASSIFIER_THRESHOLD, leopard_prob


def decode_base64_image(data_url: str):
    """Accepts either a raw base64 string or a data:image/...;base64,xxxx URL."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():

    with _state_lock:

        return jsonify({

            "models_ready": _models["ready"],

            "error": _models["error"],

            "device": str(_models["device"])
            if _models["device"] else None

        }),200


@app.route("/api/detect_frame", methods=["POST"])
def detect_frame():
    with _state_lock:
        ready = _models["ready"]
        yolo_model = _models["yolo"]
        classifier = _models["classifier"]
        device = _models["device"]

    if not ready:
        return jsonify({"success": False, "message": "Models still loading, please wait..."}), 503

    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    if not image_b64:
        return jsonify({"success": False, "message": "No image provided."}), 400

    try:
        frame = decode_base64_image(image_b64)
        if frame is None:
            return jsonify({"success": False, "message": "Could not decode image."}), 400
    except Exception:
        return jsonify({"success": False, "message": "Invalid image data."}), 400

    stime = time.time()
    sharp_frame = sharpen_frame(frame)

    results = yolo_model.predict(
        sharp_frame, conf=YOLO_CONF, iou=IOU_THRESHOLD, verbose=False, classes=ANIMAL_CLASSES
    )

    best_conf = 0.0
    best_box = None

    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
            x2, y2 = min(frame.shape[1], int(xyxy[2])), min(frame.shape[0], int(xyxy[3]))
            crop = frame[y1:y2, x1:x2]
            confirmed, leo_prob = is_leopard(crop, classifier, device)
            if confirmed and leo_prob > best_conf:
                best_conf = leo_prob
                best_box = [x1, y1, x2, y2]

    elapsed_ms = int((time.time() - stime) * 1000)

    return jsonify({
        "success": True,
        "leopard": best_box is not None,
        "confidence": round(best_conf, 3),
        "box": best_box,
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
        "inference_ms": elapsed_ms,
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 55)
    print("  LEOPARD DETECTION WEBSITE")
    print(f"  Open:  http://127.0.0.1:{port}")
    print("  Models loading in background...")
    print("=" * 55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
