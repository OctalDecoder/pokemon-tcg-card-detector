import os
import time
import logging
import glob
import json
import math
import numpy as np
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ─── TORCH & CNN IMPORTS ───────────────────────────────────────────────────────
import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
from ultralytics import YOLO

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
DEBUG              = False
USE_MASTER_CNN     = False
SAVE_RESULT_IMAGES = True
SCREENSHOT_DIR     = 'tests/fixtures'
OUTPUT_DIR         = 'output/tests'
GRID_COLS          = 5
MIDDLE_SPACE       = 20
FONT_PATH          = None

YOLO_MODEL         = 'models/card_detector.pt'
YOLO_CONF_THRESHOLD     = 0.1
BBOX_IOU_THRESH = 0.3 # Bounding box overlap threshold for merging bbox detections

CNN_BASE_DIR = 'models/' # /v3 - vertical scrolling'
CNN_CONF_THRESHOLD  = 0.15
CNN_SUBCATS = ["fullart", "standard"]

# ─── LOGGING SETUP ──────────────────────────────────────────────────────────────
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
logging.getLogger('ultralytics').setLevel(logging.DEBUG if DEBUG else logging.WARNING)
logger.info("Loading detection & classification pipeline...")

# ─── CNN INITIALISATION ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load child networks + json mappings
child_models = {}
child_maps   = {}

for cat in CNN_SUBCATS:
    # --- 1a) load the mapping file for this child ---
    with open(f"{CNN_BASE_DIR}/{cat}_mappings.json") as f:
        raw = json.load(f)
    idx2card  = {int(k):v for k,v in raw["idx_to_card"].items()}
    idx2image = {int(k):v for k,v in raw["idx_to_image"].items()}
    child_maps[cat] = (idx2card, idx2image)

    # --- 1b) build the same architecture you used during distillation ---
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f  = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, len(idx2card))

    # --- 1c) load the weights and eval() ---
    ckpt = torch.load(f"{CNN_BASE_DIR}/cnn_{cat}_student.pth", map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    child_models[cat] = model
    
# ─── MASTER CNN INITIALISATION ────────────────────────────────────────────────
if USE_MASTER_CNN:
    master_ckpt = Path(f'{CNN_BASE_DIR}/cnn_master_best.pth')
    master_sd   = torch.load(master_ckpt, map_location=device)

    # infer number of classes from the saved head
    num_classes = master_sd['classifier.1.weight'].shape[0]

    # build a fresh EfficientNet-B0
    master_cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_feats   = master_cnn.classifier[1].in_features
    master_cnn.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feats, num_classes)
    )

    # load both backbone and head (strict=False so missing keys are ignored)
    master_cnn.load_state_dict(master_sd, strict=False)

    master_cnn.to(device).eval()

# ─── DEVICE & CNN SETUP ──────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = EfficientNet_B0_Weights.DEFAULT

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
    
# ─── MERGE BOUNDING BOX HELPER ─────────────────────────────────────────────────────────
def merge_overlapping_boxes(boxes, iou_thresh=0.3):
    """
    boxes: list of (x1,y1,x2,y2,cx,cy,cat)
    returns merged list in the same format
    boxes are only merged if they share the same `cat` and their IoU > iou_thresh
    """
    def iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = area1 + area2 - inter
        return inter/union if union>0 else 0

    merged = []
    used = [False]*len(boxes)

    for i, b in enumerate(boxes):
        if used[i]:
            continue
        group = [b]
        used[i] = True
        # look for others to merge
        for j, b2 in enumerate(boxes[i+1:], start=i+1):
            if used[j] or b2[6] != b[6]:
                continue
            if iou(b, b2) > iou_thresh:
                group.append(b2)
                used[j] = True

        # combine group into one box
        xs1 = [g[0] for g in group]
        ys1 = [g[1] for g in group]
        xs2 = [g[2] for g in group]
        ys2 = [g[3] for g in group]
        nx1, ny1 = min(xs1), min(ys1)
        nx2, ny2 = max(xs2), max(ys2)
        ncat = b[6]
        ncx = (nx1+nx2)/2
        ncy = (ny1+ny2)/2

        merged.append((nx1, ny1, nx2, ny2, ncx, ncy, ncat))

    return merged

# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────────
def detect(screenshot_dir=SCREENSHOT_DIR, save_results_images=SAVE_RESULT_IMAGES, logging=True, cnn_thresh=CNN_CONF_THRESHOLD, yolo_thresh=YOLO_CONF_THRESHOLD):
    # Clean output dir
    if save_results_images:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for f in glob.glob(os.path.join(OUTPUT_DIR, '*')):
            if f.lower().endswith(('.png', '.jpg')):
                os.remove(f)

    # Get reference map
    if USE_MASTER_CNN:
        with open(f"{CNN_BASE_DIR}/master_mappings.json") as f:
            raw = json.load(f)
        idx_to_card  = {int(k):v for k,v in raw["idx_to_card"].items()}
        idx_to_image = {int(k):v for k,v in raw["idx_to_image"].items()}

    # Prepare screenshots to process
    shots = sorted(sum([glob.glob(os.path.join(screenshot_dir, f'*.{ext}')) for ext in ('png','jpg')], []))
            
    # Initialise YOLO and CNN detectors, pay the startup cost by performing dummy actions
    detector = YOLO(YOLO_MODEL)
    _ = detector.predict(np.zeros((640, 640, 3), dtype=np.uint8), conf=yolo_thresh, verbose=False)
    dummy = torch.zeros((1, 3, 224, 224), device=device)
    with torch.no_grad():
        for cat, mdl in child_models.items():
            _ = mdl(dummy)
            if logging: logger.info(f"  ↳ warmed up child model '{cat}'")

    # Begin detection
    detections = {}
    all0 = time.time()
    for sp in shots:
        bp = os.path.basename(sp)
        tt0 = time.time()
        it0 = time.time()
        orig = Image.open(sp).convert('RGB')
        arr = np.array(orig)
        image_load_time = time.time() - it0

        # Detection
        dt0 = time.time()
        results = detector.predict(arr, conf=yolo_thresh, verbose=DEBUG)

        # Extract boxes with center coordinates
        yolo_names = detector.model.names
        bboxes = []
        for r in results:
            xyxy =  r.boxes.xyxy.cpu().numpy()
            ycls =  r.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), y in zip(xyxy, ycls):
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                cat = yolo_names[y]
                bboxes.append((x1,y1,x2,y2,cx,cy,cat))

        if not bboxes:
            continue
        
        bboxes = merge_overlapping_boxes(bboxes, iou_thresh=BBOX_IOU_THRESH)

        # Estimate row height by median box height
        heights = [y2 - y1 for x1, y1, x2, y2, *rest in bboxes]
        row_h = np.median(heights)
        # Sort first by row index, then by x-center
        sorted_boxes = sorted(
            bboxes,
            key=lambda b: (int(b[1] // (row_h * 0.8)), b[4])
        )

        # Crop in sorted order
        crops = [orig.crop(tuple(map(int, b[:4]))) for b in sorted_boxes]
        det_time = time.time() - dt0
        
        
        ct0 = time.time()
        # Batch Classification
        if USE_MASTER_CNN:
            # flatten *all* crops into one batch
            batch = torch.stack([clf_tf(c) for c in crops], dim=0).to(device)
            with torch.no_grad():
                logits = master_cnn(batch)
                softm  = F.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(1).cpu().tolist()

                # build and filter
                output  = []
                matches = []
                for i,p in enumerate(preds):
                    conf = float(softm[i, p])
                    if conf >= cnn_thresh:
                        output.append(idx_to_card[p])
                        matches.append(idx_to_image[p])
        else:
            cats    = [b[6] for b in sorted_boxes]
            output  = [None] * len(crops)
            matches = [None] * len(crops)
            confs   = [0.0] * len(crops)

            for subcat in set(cats):
                idxs = [i for i,c in enumerate(cats) if c == subcat]
                batch = torch.stack([clf_tf(crops[i]) for i in idxs], dim=0).to(device)
                with torch.no_grad():
                    logits = child_models[subcat](batch)
                    softm  = F.softmax(logits, dim=1).cpu()
                sub_preds = logits.argmax(1).cpu().tolist()

                idx2card, idx2img = child_maps[subcat]
                for local_i, (i,p) in enumerate(zip(idxs, sub_preds)):
                    output[i]  = idx2card[p]
                    matches[i] = idx2img[p]
                    confs[i]   = float(softm[local_i, p])

            # now filter out any below threshold
            filtered_output = []
            filtered_matches = []
            for o,m,c in zip(output, matches, confs):
                if c >= cnn_thresh:
                    filtered_output.append(o)
                    filtered_matches.append(m)

            # use the filtered lists from here on
            output  = filtered_output
            matches = filtered_matches

        detections[bp] = output
        class_time = time.time() - ct0

        tot = time.time() - tt0
        if logging: logger.info(f"{bp:.25s} img:{image_load_time:.3f}s det:{det_time:.3f}s cls:{class_time:.3f}s | total time:{tot:.3f}s")

        # Composite
        if matches and save_results_images:
            sv0 = time.time()
            n = len(matches)
            cols = min(GRID_COLS, n)
            rows = math.ceil(n / cols)
            
            # size our grid cells to match the first real child‐mapping thumbnail
            first_thumb = next((m for m in matches if m), None)
            if first_thumb is None:
                # no valid thumbs? skip saving
                continue
            tw, th = load_thumb(first_thumb).size

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

            out_path = os.path.join(OUTPUT_DIR, bp)
            combined.save(out_path)
            svt = time.time() - sv0
            if logging: logger.info(f"{bp:.25s} | Saved result in {svt:.3f}s")
            
    allt = time.time() - all0
    if logging: logger.info(f"Detection completed in {allt:.3f}s")
    return detections

if __name__ == '__main__':
    full0 = time.time()
    logger.info("Starting pipeline with batch classification...")
    results = detect()
    fullt = time.time() - full0
    logger.info(json.dumps(results).replace("],", "],\n").replace("{", "\n ").replace("}", ""))
    logger.info(f"Full pipeline completed in {fullt:.3f}s")
