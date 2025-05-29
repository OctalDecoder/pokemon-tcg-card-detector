import os
import glob
import random
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
CARDS_FULLART_DIR        = 'images/cards/fullart'
CARDS_STANDARD_DIR       = 'images/cards/standard'
TEMPLATES_DIR            = 'images/screenshots/backgrounds'

OUTPUT_IMG_DIR           = 'dataset/yolo/images/train'
OUTPUT_IMG_DIR_VAL       = 'dataset/yolo/images/val'
OUTPUT_LABEL_DIR         = 'dataset/yolo/labels/train'
OUTPUT_LABEL_DIR_VAL     = 'dataset/yolo/labels/val'

COMPONENTS_DIR           = 'images/screenshots/backgrounds/components'
COMPONENT_OVERLAY_PROB   = 0.2            # 20% of cards get a snippet
COMPONENT_SCALE_RANGE    = (0.5, 0.9)     # snippet size relative to card width
COMPONENT_AMOUNT         = 1              # how many snippets per decorated card

NUM_SAMPLES              = 10000
BACKGROUND_ONLY_PROPORTION = 0.20
CARDS_PER_IMAGE          = (1, 25)

CLASS_FULLART            = 0
CLASS_STANDARD           = 1

SIZE_REL_RANGE           = (0.1, 0.8)
TARGET_SIZE              = (640, 640)
PAD_COLOR                = (114, 114, 114)

OFFSCREEN_CHANCE = 0.35
OFFSCREEN_MAX = 0.90
OFFSCREEN_MIN = 0.10


# ─── PRELOAD FILE LISTS ────────────────────────────────────────────────────────
fullart_paths = glob.glob(os.path.join(CARDS_FULLART_DIR, '*.png'))
standard_paths = glob.glob(os.path.join(CARDS_STANDARD_DIR, '*.png'))
card_items = [(p, CLASS_FULLART) for p in fullart_paths] + \
             [(p, CLASS_STANDARD) for p in standard_paths]
if not card_items:
    raise ValueError(f"No card PNGs found in {CARDS_FULLART_DIR} or {CARDS_STANDARD_DIR}")

template_paths = (
    glob.glob(os.path.join(TEMPLATES_DIR, '*.png')) +
    glob.glob(os.path.join(TEMPLATES_DIR, '*.jpg'))
)
template_paths = [p for p in template_paths if p.lower().endswith(('.png', '.jpg'))]
if not template_paths:
    raise ValueError(f"No template images (.png/.jpg) found in {TEMPLATES_DIR}")

component_paths = (
    glob.glob(os.path.join(COMPONENTS_DIR, '*.png')) +
    glob.glob(os.path.join(COMPONENTS_DIR, '*.jpg'))
)
component_paths = [p for p in component_paths if p.lower().endswith(('.png', '.jpg'))]
if not component_paths:
    raise ValueError(f"No component images (.png/.jpg) found in {COMPONENTS_DIR}")


# ─── OUTPUT DIRS ───────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR_VAL, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR_VAL, exist_ok=True)


def boxes_overlap(b1, b2):
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    return not (x12 <= x21 or x11 >= x22 or y12 <= y21 or y11 >= y22)


def generate_sample(sample_idx):
    random.seed(sample_idx)
    tmpl_path = random.choice(template_paths)
    tmpl_orig = Image.open(tmpl_path).convert('RGB')
    orig_w, orig_h = tmpl_orig.size

    is_background_only = random.random() < BACKGROUND_ONLY_PROPORTION
    annotations = []   # un‐clipped bboxes: (x1,y1,x2,y2,class_id)

    if not is_background_only:
        count = random.randint(*CARDS_PER_IMAGE)
        for _ in range(count):
            path, cls = random.choice(card_items)
            card = Image.open(path).convert('RGBA')
            w, h = card.size

            # scale card
            scale = min(
                random.uniform(*SIZE_REL_RANGE) * orig_w / w,
                random.uniform(*SIZE_REL_RANGE) * orig_h / h
            )
            new_w, new_h = int(w * scale), int(h * scale)
            card_resized = card.resize((new_w, new_h), Image.LANCZOS)

            # try placements
            for _ in range(10):
                # decide off-screen placement (35% chance)
                if random.random() < 0.35:
                    # x always fully on screen
                    x1 = random.randint(0, orig_w - new_w)
                    # top or bottom?
                    if random.random() < 0.5:
                        # off-screen at top between 10%-90%
                        y1 = random.randint(
                            -int(new_h * 0.90),
                            -int(new_h * 0.10)
                        )
                    else:
                        # off-screen at bottom between 10%-90%
                        y1 = random.randint(
                            orig_h - new_h + int(new_h * 0.10),
                            orig_h - new_h + int(new_h * 0.90)
                        )
                else:
                    # fully on-screen placement
                    x1 = random.randint(0, orig_w - new_w)
                    y1 = random.randint(0, orig_h - new_h)

                x2, y2 = x1 + new_w, y1 + new_h

                # clamp to visible box
                vis_x1 = max(0, x1)
                vis_y1 = max(0, y1)
                vis_x2 = min(orig_w, x2)
                vis_y2 = min(orig_h, y2)

                # ensure theres still something visible
                if vis_x2 > vis_x1 and vis_y2 > vis_y1 and \
                all(not boxes_overlap((vis_x1, vis_y1, vis_x2, vis_y2), ann[:4])
                    for ann in annotations):

                    # record only the visible portion
                    annotations.append((vis_x1, vis_y1, vis_x2, vis_y2, cls))

                    # paste the full card (so off-screen parts remain hidden)
                    tmpl_orig.paste(card_resized, (x1, y1), card_resized)

                    # optional UI snippet overlay
                    if random.random() < COMPONENT_OVERLAY_PROB:
                        for __ in range(COMPONENT_AMOUNT):
                            comp_path = random.choice(component_paths)
                            comp = Image.open(comp_path).convert('RGBA')
                            cw, ch = comp.size
                            comp_scale = random.uniform(*COMPONENT_SCALE_RANGE) * new_w / cw
                            comp_w, comp_h = int(cw * comp_scale), int(ch * comp_scale)
                            comp_resized = comp.resize((comp_w, comp_h), Image.LANCZOS)

                            cx = random.randint(x1, x2 - comp_w)
                            cy = random.randint(y1, y2 - comp_h)
                            tmpl_orig.paste(comp_resized, (cx, cy), comp_resized)

                    break  # done placing this card


    # letterbox‐resize to TARGET_SIZE
    target_w, target_h = TARGET_SIZE
    scale_factor = min(target_w / orig_w, target_h / orig_h)
    resized_w = int(orig_w * scale_factor)
    resized_h = int(orig_h * scale_factor)
    tmpl_resized = tmpl_orig.resize((resized_w, resized_h), Image.LANCZOS)

    canvas = Image.new('RGB', TARGET_SIZE, PAD_COLOR)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas.paste(tmpl_resized, (pad_x, pad_y))

    # clip annotations to [0,orig_w]×[0,orig_h]
    clipped = []
    for x1, y1, x2, y2, cls in annotations:
        cx1 = max(0, min(orig_w, x1))
        cy1 = max(0, min(orig_h, y1))
        cx2 = max(0, min(orig_w, x2))
        cy2 = max(0, min(orig_h, y2))
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue
        clipped.append((cx1, cy1, cx2, cy2, cls))

    # build YOLO labels
    yolo_lines = []
    for x1, y1, x2, y2, cls in clipped:
        nx1 = x1 * scale_factor + pad_x
        ny1 = y1 * scale_factor + pad_y
        nx2 = x2 * scale_factor + pad_x
        ny2 = y2 * scale_factor + pad_y

        cx = ((nx1 + nx2) / 2) / target_w
        cy = ((ny1 + ny2) / 2) / target_h
        bw = (nx2 - nx1) / target_w
        bh = (ny2 - ny1) / target_h
        yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # train/val split & save
    to_val = random.random() < 0.1
    img_dir = OUTPUT_IMG_DIR_VAL if to_val else OUTPUT_IMG_DIR
    lbl_dir = OUTPUT_LABEL_DIR_VAL if to_val else OUTPUT_LABEL_DIR

    fname = f'synth_{sample_idx:05d}'
    canvas.save(os.path.join(img_dir, fname + '.jpg'), quality=85)
    with open(os.path.join(lbl_dir, fname + '.txt'), 'w') as fw:
        fw.write("\n".join(yolo_lines))

    return sample_idx


if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        for idx in executor.map(generate_sample, range(NUM_SAMPLES)):
            if (idx + 1) % 100 == 0:
                print(f"Generated {idx+1}/{NUM_SAMPLES} samples")
    print('Data generation complete.')
