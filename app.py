"""
LeopardEye — Flask Backend (deployment-ready)

Architecture (fixed for the web):
  - AI models load once in a background thread when the process boots.
  - The BROWSER captures the user's own webcam (getUserMedia) and posts
    individual JPEG frames to /api/detect/frame for inference.
  - The server never opens a camera or a GUI window itself — that only
    works on a local desktop, never on a hosted server. This version
    works identically on localhost and on any cloud host.
"""

import base64
import threading
import time

import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

from detector import detector

app = Flask(__name__)
CORS(app)

model_status = {
    "models_ready": False,
    "message": "Loading AI models in background...",
}


def load_models_in_background():
    print("[Server] Loading YOLO11 + MobileNetV3...")
    t0 = time.time()
    detector.load()
    model_status["models_ready"] = True
    model_status["message"] = "Models ready — click Start Detection!"
    print(f"[Server] Models ready in {time.time() - t0:.1f}s")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(model_status.copy())


@app.route("/api/detect/frame", methods=["POST"])
def detect_frame():
    """Accepts one JPEG/PNG frame (base64 data URL or raw bytes) from the
    browser's own webcam and returns the detection result for it."""
    if not model_status["models_ready"]:
        return jsonify({"success": False, "message": "Models still loading, please wait..."}), 503

    try:
        if request.content_type and "application/json" in request.content_type:
            payload = request.get_json(silent=True) or {}
            data_url = payload.get("image", "")
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            raw = base64.b64decode(data_url)
        else:
            raw = request.get_data()

        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "message": "Could not decode image."}), 400

        result = detector.detect(frame)
        result["success"] = True
        return jsonify(result)

    except Exception as exc:  # noqa: BLE001
        print(f"[Server] detect_frame error: {exc}")
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "models_ready": model_status["models_ready"]})


# Kick off model loading as soon as the module is imported, so it works
# both under `python app.py` and under a WSGI server like gunicorn.
threading.Thread(target=load_models_in_background, daemon=True).start()


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  LEOPARD DETECTION WEBSITE")
    print("  Open:  http://127.0.0.1:5000")
    print("  Models loading in background...")
    print("=" * 55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
