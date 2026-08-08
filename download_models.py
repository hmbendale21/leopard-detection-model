"""
Utility to download and save model files into the project directory so locked deploys
(e.g. Render) can include them without requiring outbound downloads at runtime.

Usage (run locally where outbound downloads are allowed):
    python download_models.py

This will create:
 - mobilenet_weights.pth  (torch state_dict for MobileNetV3)
 - yolo11n.pt             (if ultralytics can download it into its cache)

If `yolo11n.pt` is found in an ultralytics cache location it will be copied into
the project `yolo11n.pt` path used by `app.py`.
"""
from pathlib import Path
import shutil
import sys
import traceback

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
MOBILENET_CKPT = BASE_DIR / "mobilenet_weights.pth"

print("Downloading MobileNetV3 pretrained weights and saving locally...")
try:
    import torch
    from torchvision import models

    try:
        classifier = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        torch.save(classifier.state_dict(), MOBILENET_CKPT)
        print(f"Saved MobileNet weights to {MOBILENET_CKPT}")
    except Exception as e:
        print("Failed to download MobileNet pretrained weights via torchvision:")
        traceback.print_exc()
        print("You can still use an uninitialized model, but accuracy will be lower.")

except Exception:
    print("PyTorch/torchvision not available in this environment.")
    traceback.print_exc()
    sys.exit(1)

print("Attempting to trigger ultralytics YOLO model download (if needed)...")
try:
    from ultralytics import YOLO
    # This may download into ultralytics cache
    try:
        _ = YOLO('yolo11n.pt')
    except Exception as e:
        print('ultralytics failed to instantiate YOLO("yolo11n.pt"):', e)

    # Search common ultralytics cache locations for the downloaded file
    home = Path.home()
    candidate_roots = [
        home / '.cache' / 'ultralytics',
        home / '.ultralytics',
        Path('~/.cache/ultralytics').expanduser(),
    ]

    found = None
    for root in candidate_roots:
        if not root.exists():
            continue
        for p in root.rglob('yolo11n.pt'):
            found = p
            break
        if found:
            break

    # As a best-effort, also scan the entire home cache if not found yet (can be slow)
    if not found:
        print('Not found in common cache dirs, doing a wider search under', home / '.cache')
        cache_root = home / '.cache'
        if cache_root.exists():
            for p in cache_root.rglob('yolo11n.pt'):
                found = p
                break

    if found:
        print(f'Found YOLO file in cache: {found} — copying to {MODEL_PATH}')
        shutil.copy2(found, MODEL_PATH)
        print('Copy succeeded.')
    else:
        print('Could not locate yolo11n.pt in ultralytics cache. If ultralytics can download it, run this script again or copy the file manually.')

except Exception:
    print('ultralytics not available or download attempt failed.')
    traceback.print_exc()

print('Done.')
