# 🐆 Leopard Detection System (LeopardEye)

## 🚀 Deploying to the cloud

This app was reworked so the **browser** captures your webcam (via `getUserMedia`) and streams
frames to the Flask backend, which runs the YOLO11 + MobileNetV3 pipeline and sends back a
bounding box. That's what makes it deployable — no camera needs to be attached to the server.

### Option A — Render (recommended, free tier works)
1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com) → **New +** → **Blueprint**, point it at the repo. It will
   read `render.yaml` automatically. Or manually create a **Web Service** with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn -w 1 -k gthread --threads 4 --timeout 120 -b 0.0.0.0:$PORT app:app`
3. Deploy. First boot takes 1-2 min while YOLO + MobileNetV3 download/load.
4. Open the `*.onrender.com` URL Render gives you — allow camera access when prompted.

### Option B — Docker (Railway, Fly.io, any VPS)
```bash
docker build -t leopard-detection .
docker run -p 5000:5000 -e PORT=5000 leopard-detection
```
Push the image / repo to Railway or Fly.io and it will pick up the `Dockerfile` automatically.

### Notes
- **Not compatible with Vercel** — the `torch`/`ultralytics` dependencies are far larger than
  serverless function size limits.
- Free-tier CPU hosting gives roughly 1-2 frames/sec of inference; that's expected and by design
  (`CAPTURE_INTERVAL_MS` in `static/js/main.js` controls the capture rate).
- The site must be served over **HTTPS** (or `localhost`) for browsers to allow camera access —
  Render/Railway/Fly all provide HTTPS automatically.


## 📌 Problem Statement
Leopard intrusions near human settlements pose serious safety risks, especially in areas close to forests and wildlife reserves. Traditional monitoring systems often fail to accurately distinguish leopards from other animals, leading to false alarms or missed detections.

There is a need for a reliable, real-time AI-based system that can accurately detect leopard presence and trigger immediate alerts.

---

## 🎯 Objectives
- To develop a real-time leopard detection system using deep learning  
- To minimize false positives by filtering non-leopard detections  
- To implement a multi-stage AI pipeline for higher accuracy  
- To provide instant alerts upon confirmed detection  
- To build a complete end-to-end system  

---

## 💡 Proposed Solution
The system uses a two-stage detection pipeline:

1. YOLO model detects objects (animals) in real-time  
2. MobileNetV3 verifies whether the detected object is a leopard  

This approach ensures high accuracy and reduces false detections.

---

## ⚙️ System Modules

### 1. Data Processing Module
- Captures live video using webcam  
- Performs frame extraction and preprocessing  

### 2. Detection Module (YOLO)
- Detects objects in each frame  
- Generates bounding boxes  

### 3. Verification Module (MobileNetV3)
- Classifies detected objects as leopard or non-leopard  
- Filters out incorrect detections  

### 4. Alert Module
- Triggers beep sound on leopard detection  
- Displays bounding box with confidence score  

### 5. Web Interface Module
- Interactive UI using HTML, CSS, JavaScript  
- Real-time visualization of detection  

---

## 🧠 Technologies Used
- Python  
- OpenCV  
- NumPy  
- PyTorch  
- YOLO (Ultralytics)  
- MobileNetV3  
- Flask  
- HTML, CSS, JavaScript  
- GitHub  

---

## 🏗️ System Architecture
Camera Input → Frame Processing → YOLO Detection → MobileNetV3 Verification → Alert & Output

---

## 🚀 Implementation Steps
1. Capture video using OpenCV  
2. Preprocess frames  
3. Apply YOLO for object detection  
4. Extract detected objects  
5. Pass to MobileNetV3 for verification  
6. Filter only leopard detections  
7. Display bounding box  
8. Trigger alert sound  
9. Integrate backend using Flask  
10. Build frontend interface  

---

## 📊 Results
- Accurate real-time leopard detection  
- Reduced false positives  
- Fast processing and response  
- Stable performance in live feed  

---

## 🔥 Key Features
- Real-time detection  
- Leopard-specific filtering  
- Two-stage AI pipeline  
- Instant alert system  
- Web-based interface  

---

## 🚧 Limitations
- Performance depends on lighting conditions  
- Requires good camera quality  
- Accuracy depends on model training  

---

## 🚀 Future Scope
- Night vision detection  
- Mobile application integration  
- Cloud-based monitoring  
- SMS/Email alerts  
- Multi-animal detection  

---

## 📈 Outcome
- Developed a complete AI-based detection system  
- Gained hands-on experience in deep learning and computer vision  
- Built a real-world application with practical use cases  

---

## 👨‍💻 Author
Himanshu Bendale  
AI & Data Science Engineering Student
