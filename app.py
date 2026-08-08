"""
Leopard Detection Website — Flask Backend
- Pre-loads AI models in background on server start
- On Detect click: writes trigger file → camera opens INSTANTLY
- On Stop click: writes stop file → camera closes
"""

import os
import sys
import subprocess
import threading
import time
import pathlib
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR     = pathlib.Path(__file__).parent
READY_FILE   = BASE_DIR / ".model_ready"
TRIGGER_FILE = BASE_DIR / ".detect_trigger"
STOP_FILE    = BASE_DIR / ".detect_stop"

# Track the preload process
preload_proc   = None
detection_lock = threading.Lock()
detection_status = {
    "running":     False,
    "models_ready": False,
    "start_time":  None,
    "message":     "Loading AI models in background..."
}


def launch_preload():
    """Start test_live_cam.py --preload in background when server boots."""
    global preload_proc

    # Clean stale flag files
    for f in (READY_FILE, TRIGGER_FILE, STOP_FILE):
        f.unlink(missing_ok=True)

    python_exe = sys.executable
    script     = str(BASE_DIR / "test_live_cam.py")

    print("[Server] Launching preload process...")
    proc = subprocess.Popen(
        [python_exe, script, "--preload"],
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    preload_proc = proc

    # Watch for READY_FILE
    def watch():
        while True:
            if READY_FILE.exists():
                with detection_lock:
                    detection_status["models_ready"] = True
                    detection_status["running"]      = False
                    detection_status["message"]      = "Models ready — click Detect for instant camera!"
                print("[Server] Models are ready!")
                break
            if proc.poll() is not None:
                print("[Server] Preload process exited unexpectedly.")
                break
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    # Keep models_ready in sync with READY_FILE
    ready = READY_FILE.exists()
    with detection_lock:
        detection_status["models_ready"] = ready
        if ready and not detection_status["running"]:
            detection_status["message"] = "Models ready — click Detect for instant camera!"
        return jsonify(detection_status.copy())


@app.route("/api/detect/start", methods=["POST"])
def start_detection():
    with detection_lock:
        if detection_status["running"]:
            return jsonify({"success": False, "message": "Detection is already running."})
        if not detection_status["models_ready"]:
            return jsonify({"success": False, "message": "Models still loading, please wait a few seconds..."})

        # Write trigger → preload process opens camera immediately
        TRIGGER_FILE.write_text("go")
        detection_status["running"]    = True
        detection_status["start_time"] = time.time()
        detection_status["message"]    = "Camera opening..."

    # Watch for camera to close (READY_FILE re-appears)
    def watch_stop():
        time.sleep(2)   # give camera time to open
        while True:
            if READY_FILE.exists():
                with detection_lock:
                    detection_status["running"] = False
                    detection_status["message"] = "Models ready — click Detect for instant camera!"
                print("[Server] Detection finished, ready for next click.")
                break
            time.sleep(0.5)

    threading.Thread(target=watch_stop, daemon=True).start()
    return jsonify({"success": True, "message": "Camera opening instantly!"})


@app.route("/api/detect/stop", methods=["POST"])
def stop_detection():
    with detection_lock:
        if not detection_status["running"]:
            return jsonify({"success": False, "message": "No detection is running."})
        STOP_FILE.write_text("stop")
        detection_status["running"] = False
        detection_status["message"] = "Stopping..."
    return jsonify({"success": True, "message": "Detection stopped."})


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    # Boot: immediately start loading models in background
    t = threading.Thread(target=launch_preload, daemon=True)
    t.start()

    print("\n" + "="*55)
    print("  LEOPARD DETECTION WEBSITE")
    print("  Open:  http://127.0.0.1:5000")
    print("  Models loading in background...")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
