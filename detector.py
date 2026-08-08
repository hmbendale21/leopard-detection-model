"""
detector.py — Shared two-stage leopard detection pipeline.

Stage 1: YOLO11 finds candidate animal boxes.
Stage 2: MobileNetV3 (ImageNet) confirms leopard/jaguar/snow-leopard.

This module holds no server/camera logic — it just takes a BGR frame
(numpy array) in and returns detection results out. That makes it
usable from a web request handler (app.py) as well as the local
webcam CLI (test_live_cam.py).
"""

import pathlib
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from ultralytics import YOLO

BASE_DIR = pathlib.Path(__file__).parent
MODEL_PATH = str(BASE_DIR / "yolo11n.pt")

YOLO_CONF = 0.30
IOU_THRESHOLD = 0.45

# COCO animal classes YOLO is allowed to propose as candidates
ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]

# ImageNet classes that count as "leopard"
LEOPARD_IMAGENET_IDS = {288, 289, 290}  # leopard, snow leopard, jaguar
CLASSIFIER_THRESHOLD = 0.15

SHARPEN_KERNEL = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
], dtype=np.float32)

CLASSIFY_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LeopardDetector:
    """Loads once, then answers detect() calls cheaply."""

    def __init__(self):
        self.yolo = None
        self.classifier = None
        self.device = None
        self.ready = False

    def load(self):
        self.yolo = YOLO(MODEL_PATH)
        self.classifier = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.classifier.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
        self.ready = True

    def _is_leopard(self, crop_bgr):
        if crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return False, 0.0
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = CLASSIFY_TRANSFORM(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        leopard_prob = sum(probs[cid].item() for cid in LEOPARD_IMAGENET_IDS)
        return leopard_prob >= CLASSIFIER_THRESHOLD, leopard_prob

    def detect(self, frame_bgr):
        """Run the full pipeline on a single BGR frame.

        Returns: {leopard_found, confidence, box: [x1,y1,x2,y2] | None}
        """
        if not self.ready:
            raise RuntimeError("Detector models are not loaded yet")

        sharp = cv2.filter2D(frame_bgr, -1, SHARPEN_KERNEL)
        results = self.yolo.predict(
            sharp,
            conf=YOLO_CONF,
            iou=IOU_THRESHOLD,
            verbose=False,
            classes=ANIMAL_CLASSES,
        )

        best_conf = 0.0
        best_box = None

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                x2 = min(frame_bgr.shape[1], int(xyxy[2]))
                y2 = min(frame_bgr.shape[0], int(xyxy[3]))
                crop = frame_bgr[y1:y2, x1:x2]
                confirmed, prob = self._is_leopard(crop)
                if confirmed and prob > best_conf:
                    best_conf = prob
                    best_box = [x1, y1, x2, y2]

        return {
            "leopard_found": best_box is not None,
            "confidence": round(float(best_conf), 4),
            "box": best_box,
        }


# Single shared instance used by app.py
detector = LeopardDetector()
