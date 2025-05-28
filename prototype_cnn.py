import os
import time
import logging
import glob
import math
import numpy as np
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

# ─── TORCH & CNN IMPORTS ───────────────────────────────────────────────────────
import torch
from torch import nn
import torchvision as tv
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from ultralytics import YOLO

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
DEBUG              = False
SAVE_RESULT_IMAGES = False
SCREENSHOT_DIR     = 'images/screenshots/test suite 1'
DATA_DIR_FULLART   = 'images/cards/fullart'
DATA_DIR_STANDARD  = 'images/cards/standard'
OUTPUT_DIR         = 'images/results'
GRID_COLS          = 5
MIDDLE_SPACE       = 20
FONT_PATH          = None

YOLO_MODEL         = 'dataset/card_detector.pt'
CNN_MODEL_PATH     = 'dataset/cnn_best.pth'
CNN_TRAIN_DATA     = 'dataset/cnn/train'
CONF_THRESHOLD     = 0.80

# ─── LOGGING SETUP ──────────────────────────────────────────────────────────────
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
logging.getLogger('ultralytics').setLevel(logging.DEBUG if DEBUG else logging.WARNING)
logger.info("Loading detection & classification pipeline...")

# ─── DEVICE & CNN SETUP ──────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = EfficientNet_B0_Weights.DEFAULT
cnn = efficientnet_b0(weights=weights)

# ─── DATASET CLASSES ────────────────────────────────────────────────────────────
train_ds = tv.datasets.ImageFolder(CNN_TRAIN_DATA)
classes = train_ds.classes
num_classes = len(classes)
in_feats = cnn.classifier[1].in_features
cnn.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, num_classes))
state = torch.load(CNN_MODEL_PATH, map_location=device)
cnn.load_state_dict(state)
cnn.to(device).eval()

# ─── PREPROCESS FOR CLASSIFICATION ──────────────────────────────────────────────
clf_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
])

# ─── THUMB & FONT SETUP ─────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def load_thumb(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')

if FONT_PATH and os.path.exists(FONT_PATH):
    font = ImageFont.truetype(FONT_PATH, 12)
else:
    font = None

# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────────
def run():
    # Clean output dir
    if SAVE_RESULT_IMAGES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for f in glob.glob(os.path.join(OUTPUT_DIR, '*')):
            if f.lower().endswith(('.png', '.jpg')):
                os.remove(f)

    # Build reference map
    ref_map = {}
    for d in (DATA_DIR_FULLART, DATA_DIR_STANDARD):
        for p in glob.glob(os.path.join(d, '*.png')):
            cid = os.path.splitext(os.path.basename(p))[0]
            ref_map[cid] = p

    detector = YOLO(YOLO_MODEL)
    shots = sorted(sum([glob.glob(os.path.join(SCREENSHOT_DIR, f'*.{ext}')) for ext in ('png','jpg')], []))

    for sp in shots:
        orig = Image.open(sp).convert('RGB')
        arr = np.array(orig)

        # Detection
        dt0 = time.time()
        results = detector.predict(arr, conf=CONF_THRESHOLD, verbose=DEBUG)
        det_time = time.time() - dt0

        # Extract boxes with center coordinates
        bboxes = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bboxes.append((x1, y1, x2, y2, cx, cy))

        if not bboxes:
            continue

        # Estimate row height by median box height
        heights = [y2 - y1 for x1,y1,x2,y2,_,_ in bboxes]
        row_h = np.median(heights)
        # Sort first by row index, then by x-center
        sorted_boxes = sorted(
            bboxes,
            key=lambda b: (int(b[1] // (row_h * 0.8)), b[4])
        )

        # Crop in sorted order
        crops = [orig.crop(tuple(map(int, b[:4]))) for b in sorted_boxes]

        # Batch Classification
        ct0 = time.time()
        if crops:
            inputs = torch.stack([clf_tf(c).to(device) for c in crops], dim=0)
            with torch.no_grad():
                logits = cnn(inputs)
            preds = logits.argmax(1).cpu().tolist()
            matches = [ref_map.get(classes[p]) for p in preds]

            # Logging CNN classification results
            if DEBUG:
                for idx, p in enumerate(preds):
                    class_id = classes[p]
                    match_path = ref_map.get(class_id)
                    logger.debug(f"CNN result - crop {idx}: predicted '{class_id}', matched path: {match_path}")
        else:
            matches = []
        class_time = time.time() - ct0

        logger.info(f"{os.path.basename(sp)} det:{det_time:.3f}s cls:{class_time:.3f}s")

        # Composite
        if matches and SAVE_RESULT_IMAGES:
            n = len(matches)
            cols = min(GRID_COLS, n)
            rows = math.ceil(n / cols)
            tw, th = load_thumb(next(iter(ref_map.values()))).size

            # Build right grid
            right = Image.new('RGB', (cols * tw, rows * th), (0, 0, 0))
            draw = ImageDraw.Draw(right)
            for i, m in enumerate(matches):
                if m:
                    thumb = load_thumb(m).resize((tw, th), Image.LANCZOS)
                    r, c = divmod(i, cols)
                    right.paste(thumb, (c * tw, r * th))
                    if font:
                        draw.text((c * tw + 2, r * th + 2), os.path.basename(m), font=font, fill=(255,255,255))

            # Left panel: match entire right height
            left_h = rows * th
            w0, h0 = orig.size
            left_w = int(w0 / h0 * left_h)
            left = orig.resize((left_w, left_h), Image.LANCZOS)

            combined = Image.new('RGB', (left_w + MIDDLE_SPACE + right.width, left_h), (0, 0, 0))
            combined.paste(left, (0, 0))
            combined.paste(right, (left_w + MIDDLE_SPACE, 0))

            out_path = os.path.join(OUTPUT_DIR, os.path.basename(sp))
            combined.save(out_path)

if __name__ == '__main__':
    logger.info("Starting pipeline with batch classification...")
    run()
