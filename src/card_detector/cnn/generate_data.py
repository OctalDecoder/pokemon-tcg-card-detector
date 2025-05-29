import os
import glob
import shutil
import random
import numpy as np
from PIL import Image
import albumentations as A
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR   = os.path.join(BASE_DIR, 'images', 'cards')
TRAIN_DIR    = os.path.join(BASE_DIR, 'dataset', 'cnn', 'train')
VAL_DIR      = os.path.join(BASE_DIR, 'dataset', 'cnn', 'val')
VAL_SPLIT    = 0.2
VARIANTS_PER = 150
RANDOM_SEED  = 88
ALLOWED_EXT  = ('.png', '.jpg', '.jpeg', '.bmp')

# ─── COMPONENT OVERLAY CONFIG ───────────────────────────────────────────────────
COMPONENTS_DIR         = os.path.join(BASE_DIR, 'images', 'screenshots', 'backgrounds', 'components')
COMPONENT_ON_IMAGE_PROB = 0.08             # 8% of variants get a snippet
COMPONENT_SCALE_RANGE  = (0.25, 0.5)       # snippet size relative to image width
COMPONENT_AMOUNT       = 1                # how many per decorated image

# gather component paths
component_paths = (
    glob.glob(os.path.join(COMPONENTS_DIR, '*.png')) +
    glob.glob(os.path.join(COMPONENTS_DIR, '*.jpg'))
)
component_paths = [p for p in component_paths if p.lower().endswith(('.png','.jpg'))]
if not component_paths:
    raise ValueError(f"No component images found in {COMPONENTS_DIR!r}")

# ─── AUGMENTATION PIPELINE ──────────────────────────────────────────────────────
augmentor = A.Compose([
    A.OneOf([
        A.Resize(224, 224),
        A.RandomResizedCrop(size=(224,224), scale=(0.5,1.0), ratio=(0.8,1.2)),
    ], p=1.0),
    A.Affine(
        translate_percent={"x":(-0.10,0.10), "y":(-0.90,0.90)},
        p=0.25
    ),
    A.Affine(rotate=(-5,5), shear=(-2,2), p=0.8),
    A.Perspective(scale=(0.01,0.05), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.1, p=0.7),
    A.Blur(blur_limit=3, p=0.3),
], p=1.0)

# ─── WORKER FUNCTION ────────────────────────────────────────────────────────────
def process_image(args):
    img_path, card_id, subdir = args
    try:
        # deterministic per-process
        seed = RANDOM_SEED + os.getpid()
        random.seed(seed)
        np.random.seed(seed)

        print(f"[PID {os.getpid()}] processing {img_path}")

        # load the original card as RGBA so we can overlay
        with Image.open(img_path) as raw:
            base_rgba = raw.convert('RGBA')

        # prepare output dirs
        train_out = os.path.join(TRAIN_DIR, subdir, card_id)
        val_out   = os.path.join(VAL_DIR,   subdir, card_id)
        for d in (train_out, val_out):
            os.makedirs(d, exist_ok=True)

        # generate variants
        for i in range(VARIANTS_PER):
            # 1) start from base
            working = base_rgba.copy()

            # 2) optionally overlay component snippet
            if component_paths and random.random() < COMPONENT_ON_IMAGE_PROB:
                for _ in range(COMPONENT_AMOUNT):
                    comp_path = random.choice(component_paths)
                    with Image.open(comp_path).convert('RGBA') as comp:
                        cw, ch = comp.size
                        img_w, img_h = working.size
                        # scale snippet
                        scale = random.uniform(*COMPONENT_SCALE_RANGE) * img_w / cw
                        comp_w, comp_h = int(cw*scale), int(ch*scale)
                        snippet = comp.resize((comp_w, comp_h), Image.LANCZOS)
                        # skip if too large
                        if comp_w > img_w or comp_h > img_h:
                            continue
                        # random position
                        max_x, max_y = img_w - comp_w, img_h - comp_h
                        cx = random.randint(0, max_x)
                        cy = random.randint(0, max_y)
                        working.paste(snippet, (cx, cy), snippet)

            # 3) convert to RGB numpy for augmentation
            working_rgb = working.convert('RGB')
            img_np = np.array(working_rgb)

            # 4) augment
            aug_np = augmentor(image=img_np)['image']
            out_img = Image.fromarray(aug_np)

            # 5) split & save
            dst = val_out if random.random() < VAL_SPLIT else train_out
            out_p = os.path.join(dst, f"{card_id}_{i}.png")
            out_img.save(out_p)
            out_img.close()

    except Exception as e:
        print(f"[PID {os.getpid()}] ERROR on {img_path}: {e}")

# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # wipe old outputs
    for d in (TRAIN_DIR, VAL_DIR):
        if os.path.exists(d):
            print(f"Removing existing directory: {d}")
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        print(f"→ ensuring output dir exists: {d}")

    # collect all source images
    all_images = []
    for root, _, files in os.walk(SOURCE_DIR):
        for fn in files:
            if fn.lower().endswith(ALLOWED_EXT):
                path = os.path.join(root, fn)
                cid  = os.path.splitext(fn)[0]
                sub  = os.path.relpath(root, SOURCE_DIR).split(os.sep)[0]
                all_images.append((path, cid, sub))
    print(f"Found {len(all_images)} source images in {SOURCE_DIR!r}")

    # launch workers
    cpu_count = multiprocessing.cpu_count()
    n_workers = max(1, int(cpu_count * 0.8))
    print(f"Launching {n_workers}/{cpu_count} workers…")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pool.map(process_image, all_images)

    print("Dataset generation complete.")
