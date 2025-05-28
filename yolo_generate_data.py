import os
import glob
import random
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Configuration
CARDS_FULLART_DIR      = 'images/cards/fullart'    # directory with full-art card PNGs
CARDS_STANDARD_DIR     = 'images/cards/standard'   # directory with standard card PNGs
TEMPLATES_DIR          = 'images/screenshots/backgrounds'  # directory with background screenshots

OUTPUT_IMG_DIR         = 'dataset/yolo/images/train'   # output synthetic images
OUTPUT_LABEL_DIR       = 'dataset/yolo/labels/train'   # output YOLO labels
OUTPUT_IMG_DIR_VAL     = 'dataset/yolo/images/val'     # output validation images
OUTPUT_LABEL_DIR_VAL   = 'dataset/yolo/labels/val'     # output validation labels

NUM_SAMPLES            = 10000                     # number of synthetic images to generate
BACKGROUND_ONLY_PROPORTION = 0.20                 # proportion of images that should have no cards present
CARDS_PER_IMAGE        = (1, 25)                  # min, max number of cards per synthetic image

# YOLO class indices
CLASS_FULLART          = 0
CLASS_STANDARD         = 1

SIZE_REL_RANGE         = (0.1, 0.8)              # relative size range of card vs template
TARGET_SIZE            = (640, 640)              # target input resolution for YOLO (width, height)
PAD_COLOR              = (114, 114, 114)         # letterbox padding color (YOLO default gray)

# Preload source file lists
fullart_paths = glob.glob(os.path.join(CARDS_FULLART_DIR, '*.png'))
standard_paths = glob.glob(os.path.join(CARDS_STANDARD_DIR, '*.png'))
# Combine into (path, class_id) tuples
card_items = [(p, CLASS_FULLART) for p in fullart_paths] + [(p, CLASS_STANDARD) for p in standard_paths]
if not card_items:
    raise ValueError(f"No card PNGs found in {CARDS_FULLART_DIR} or {CARDS_STANDARD_DIR}")

template_paths = glob.glob(os.path.join(TEMPLATES_DIR, '*.png'))
if not template_paths:
    raise ValueError(f"No template PNGs found in {TEMPLATES_DIR}")

# Ensure output directories exist
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR_VAL, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR_VAL, exist_ok=True)


def boxes_overlap(b1, b2):
    # b = (x1,y1,x2,y2)
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def generate_sample(i):
    random.seed(i)
    tmpl_path = random.choice(template_paths)
    tmpl_orig = Image.open(tmpl_path).convert('RGB')
    orig_w, orig_h = tmpl_orig.size

    # decide if this should be a "background-only" image
    is_background_only = random.random() < BACKGROUND_ONLY_PROPORTION

    # track bounding boxes and their class_ids
    annotations = []  # list of (x1,y1,x2,y2,class_id)
    if not is_background_only:
        count = random.randint(*CARDS_PER_IMAGE)
        for _ in range(count):
            path, cls = random.choice(card_items)
            card = Image.open(path).convert('RGBA')
            w, h = card.size
            scale = min(random.uniform(*SIZE_REL_RANGE) * orig_w / w,
                        random.uniform(*SIZE_REL_RANGE) * orig_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            card_resized = card.resize((new_w, new_h), Image.LANCZOS)

            # attempt to find non-overlapping placement
            for attempt in range(10):
                x1 = random.randint(0, orig_w - new_w)
                y1 = random.randint(0, orig_h - new_h)
                x2, y2 = x1 + new_w, y1 + new_h
                box = (x1, y1, x2, y2)
                if all(not boxes_overlap(box, ann[:4]) for ann in annotations):
                    annotations.append((x1, y1, x2, y2, cls))
                    tmpl_orig.paste(card_resized, (x1, y1), card_resized)
                    break
        # end for cards

    # letterbox resize
    target_w, target_h = TARGET_SIZE
    scale_factor = min(target_w / orig_w, target_h / orig_h)
    resized_w, resized_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
    tmpl_resized = tmpl_orig.resize((resized_w, resized_h), Image.LANCZOS)
    canvas = Image.new('RGB', TARGET_SIZE, PAD_COLOR)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas.paste(tmpl_resized, (pad_x, pad_y))

    # build YOLO label lines
    yolo_lines = []
    for x1, y1, x2, y2, cls in annotations:
        nx1 = x1 * scale_factor + pad_x
        ny1 = y1 * scale_factor + pad_y
        nx2 = x2 * scale_factor + pad_x
        ny2 = y2 * scale_factor + pad_y
        cx = ((nx1 + nx2) / 2) / target_w
        cy = ((ny1 + ny2) / 2) / target_h
        bw = (nx2 - nx1) / target_w
        bh = (ny2 - ny1) / target_h
        yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # train/val split
    to_val = random.random() < 0.1
    img_dir = OUTPUT_IMG_DIR_VAL if to_val else OUTPUT_IMG_DIR
    lbl_dir = OUTPUT_LABEL_DIR_VAL if to_val else OUTPUT_LABEL_DIR

    fname = f'synth_{i:05d}'
    canvas.save(os.path.join(img_dir, fname + '.jpg'), quality=85)
    with open(os.path.join(lbl_dir, fname + '.txt'), 'w') as f:
        f.write("\n".join(yolo_lines))

    return i


# Main execution with parallel generation
if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        for idx in executor.map(generate_sample, range(NUM_SAMPLES)):
            if (idx + 1) % 100 == 0:
                print(f"Generated {idx+1}/{NUM_SAMPLES} samples")
    print('Data generation complete.')
