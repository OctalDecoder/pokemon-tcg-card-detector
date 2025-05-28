import os
import time
import logging

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
DEBUG              = False                               # DEBUG: Print bootstrap times (uses DEBUG log level)
SAVE_RESULT_IMAGES = True                                # Save results as images to OUTPUT_DIR
SCREENSHOT_DIR     = 'images/screenshots/test suite 1'   # Directory containing screenshots to process
DATA_DIR_FULLART   = 'images/cards/fullart'              # Directory of full art cards
DATA_DIR_STANDARD  = 'images/cards/standard'             # Directory of standard cards
OUTPUT_DIR         = 'images/results'                    # Directory where resulting images are stored
GRID_COLS          = 5                                   # Max number of columns in output images
MIDDLE_SPACE       = 20                                  # Pixels between the screenshot and detected images in image output

YOLO_MODEL         = 'dataset/card_detector.pt'
FULL_EMB_PATH      = 'dataset/fullart_emb.npy'
STD_EMB_PATH       = 'dataset/standard_emb.npy'
CONF_THRESHOLD     = 0.90
STANDARD_BOX_REL   = (0.084, 0.098, 0.916, 0.471)        # Crop region that gets the standard cards art (roughly)
BATCH_SIZE         = 32
FONT_PATH          = None

# ─── LOGGING SETUP ──────────────────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Ultralytics logging
logging.getLogger('ultralytics').setLevel(
    logging.DEBUG if DEBUG else logging.WARNING
)

logger.info("Loading card detection prototype...")

# ─── BOOTSTRAP TIMING HELPER ────────────────────────────────────────────────────
record_times = {}
def record(stage):
    if DEBUG:
        record_times[stage] = time.perf_counter()

# ─── ENVIRONMENT SETUP ──────────────────────────────────────────────────────────
record('start_total')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
record('end_env')

# ─── IMPORTS ────────────────────────────────────────────────────────────────────
record('before_imports')
import faiss
import glob
import math
import numpy as np
import tensorflow as tf
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2 # type: ignore
from tensorflow.keras.preprocessing import image                    # type: ignore
from tensorflow.keras.models import Model                           # type: ignore
from ultralytics import YOLO
record('after_imports')

# ─── EMBEDDING MODEL ────────────────────────────────────────────────────────────
record('before_embedder')
base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)
feature_extractor = Model(inputs=base.input, outputs=base.output)

@tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
def embed_batch(batch_tensor):
    return feature_extractor(batch_tensor, training=False)
record('after_embedder')

def get_embeddings(images, target_size=224, batch_size=BATCH_SIZE):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        arrs = []
        for img_src in batch:
            if isinstance(img_src, str):
                im = image.load_img(img_src, target_size=(target_size, target_size))
                ar = image.img_to_array(im)
            else:
                ar = np.array(img_src.resize((target_size, target_size)))
            arrs.append(ar)
        arrs = np.stack(arrs, axis=0)
        arrs = mobilenet_v2.preprocess_input(arrs)
        tensor = tf.convert_to_tensor(arrs)
        emb = embed_batch(tensor).numpy().astype('float32')
        feats.append(emb)
    return np.vstack(feats)

# ─── PRELOAD REFERENCE PATHS & EMBEDDINGS ────────────────────────────────────────
record('before_paths')
fullart_paths = sorted(glob.glob(os.path.join(DATA_DIR_FULLART, '*.png')))
standard_paths = sorted(glob.glob(os.path.join(DATA_DIR_STANDARD, '*.png')))
logger.debug(f"Found {len(fullart_paths)} full-art, {len(standard_paths)} standard reference images")
record('after_paths')

record('before_full_emb')
if os.path.exists(FULL_EMB_PATH):
    full_emb = np.load(FULL_EMB_PATH)
else:
    full_emb = get_embeddings(fullart_paths)
    np.save(FULL_EMB_PATH, full_emb)
record('after_full_emb')

record('before_std_emb')
def build_standard_embeddings():
    std_crops = []
    for p in standard_paths:
        img = Image.open(p).convert('RGB')
        w, h = img.size
        x1 = int(STANDARD_BOX_REL[0] * w)
        y1 = int(STANDARD_BOX_REL[1] * h)
        x2 = int(STANDARD_BOX_REL[2] * w)
        y2 = int(STANDARD_BOX_REL[3] * h)
        std_crops.append(img.crop((x1, y1, x2, y2)))
    return get_embeddings(std_crops)

if os.path.exists(STD_EMB_PATH):
    std_emb = np.load(STD_EMB_PATH)
else:
    std_emb = build_standard_embeddings()
    np.save(STD_EMB_PATH, std_emb)
record('after_std_emb')

# ─── BUILD FAISS INDICES ─────────────────────────────────────────────────────────
record('before_faiss')
dim = full_emb.shape[1]
index_full = faiss.IndexFlatL2(dim)
index_full.add(full_emb)
index_std = faiss.IndexFlatL2(dim)
index_std.add(std_emb)
record('after_faiss')

# ─── THUMB LOADER & FONT ────────────────────────────────────────────────────────
record('before_thumbs')
@lru_cache(maxsize=None)
def load_thumb(path):
    return Image.open(path).convert('RGB')

if FONT_PATH and os.path.exists(FONT_PATH):
    font = ImageFont.truetype(FONT_PATH, 12)
else:
    font = None
record('after_thumbs')

# ─── INITIALIZE YOLO ─────────────────────────────────────────────────────────────
record('before_detector')
detector = YOLO(YOLO_MODEL)
record('after_detector')

# ─── HELPER: ASPECT-RATIO PADDING ────────────────────────────────────────────────
def resize_with_padding(img: Image.Image, target_size: tuple[int,int]) -> Image.Image:
    '''Resize img to fit within target_size preserving aspect ratio, pad with black.'''
    return ImageOps.pad(img, target_size, color=(0,0,0))

# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────────
def run():
    # Cleanup output directory
    if SAVE_RESULT_IMAGES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for f in glob.glob(os.path.join(OUTPUT_DIR, '*')):
            if f.lower().endswith(('.png', '.jpg')):
                os.remove(f)

    # Gather screenshots
    shots = []
    for ext in ('png','jpg'):
        shots.extend(glob.glob(os.path.join(SCREENSHOT_DIR, f'*.{ext}')))
    shots = sorted(shots)

    # Process each screenshot
    for sp in shots:
        t0 = time.time()
        orig = Image.open(sp).convert('RGB')
        arr = np.array(orig)

        # Detection
        dt0 = time.time()
        results = detector.predict(arr, conf=CONF_THRESHOLD, verbose=DEBUG)
        det_time = time.time() - dt0

        # Collect crops
        crops, cls_ids = [], []
        for r in results:
            for box, cid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                x1,y1,x2,y2 = map(int, box)
                if cid == 0:
                    crops.append(orig.crop((x1,y1,x2,y2)))
                    cls_ids.append(0)
                elif cid == 1:
                    card = orig.crop((x1,y1,x2,y2))
                    cw,ch = card.size
                    x1a,y1a = int(STANDARD_BOX_REL[0]*cw), int(STANDARD_BOX_REL[1]*ch)
                    x2a,y2a = int(STANDARD_BOX_REL[2]*cw), int(STANDARD_BOX_REL[3]*ch)
                    crops.append(card.crop((x1a,y1a,x2a,y2a)))
                    cls_ids.append(1)

        # Classification
        ct0 = time.time()
        matches = []
        if crops:
            embs = get_embeddings(crops)
            for emb, cid in zip(embs, cls_ids):
                idx = (index_full if cid==0 else index_std).search(emb.reshape(1,-1),1)[1][0][0]
                matches.append(fullart_paths[idx] if cid==0 else standard_paths[idx])
        class_time = time.time() - ct0

        logger.info(f"Processed {os.path.basename(sp)} - det: {det_time:.3f}s, cls: {class_time:.3f}s")

        # Composite images
        if matches and SAVE_RESULT_IMAGES:
            n    = len(matches)
            cols = min(GRID_COLS, n)
            rows = math.ceil(n / cols)
            tw, th = load_thumb(fullart_paths[0]).size

            # LEFT panel: one-column padded original
            left_w, left_h = tw, rows * th
            left = Image.new('RGB', (left_w, left_h), (0, 0, 0))
            orig_pad = resize_with_padding(orig, (left_w, left_h))
            left.paste(orig_pad, (0, 0))

            # RIGHT panel: matched thumbs grid
            right = Image.new('RGB', (cols * tw, rows * th), (0, 0, 0))
            draw_r = ImageDraw.Draw(right)
            for i, m in enumerate(matches):
                thumb = resize_with_padding(load_thumb(m), (tw, th))
                r, c = divmod(i, cols)
                right.paste(thumb, (c * tw, r * th))
                if font:
                    draw_r.text((c*tw+2, r*th+2),
                                os.path.basename(m),
                                font=font,
                                fill=(255,255,255))

            # Merge and save
            total_w = left_w + MIDDLE_SPACE + right.width
            total_h = max(left_h, right.height)
            combined = Image.new('RGB', (total_w, total_h), (0,0,0))
            combined.paste(left, (0,0))
            combined.paste(right, (left_w + MIDDLE_SPACE, 0))
            out = os.path.join(OUTPUT_DIR, os.path.basename(sp))
            combined.save(out)
            logger.info(f"Saved composite: {out}")

if __name__ == '__main__':
    logger.info("Card detection prototype successfully loaded ✔️")
    if DEBUG:
        labels = [
            'start_total',
            'before_imports', 'after_imports',
            'before_embedder', 'after_embedder',
            'before_paths', 'after_paths',
            'before_full_emb', 'after_full_emb',
            'before_std_emb', 'after_std_emb',
            'before_faiss', 'after_faiss',
            'before_thumbs', 'after_thumbs',
            'before_detector', 'after_detector',
            'end_total'
        ]
        logger.debug('Bootstrap performance timings:')
        for a, b in zip(labels, labels[1:]):
            dt = record_times.get(b, 0) - record_times.get(a, 0)
            logger.debug(f"  {a} → {b}: {dt:.3f}s")
            
    logger.info("Starting image detection...")
    run()