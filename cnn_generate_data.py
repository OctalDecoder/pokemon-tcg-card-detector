import os
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
VARIANTS_PER = 120
RANDOM_SEED  = 132
ALLOWED_EXT  = ('.png', '.jpg', '.jpeg', '.bmp')

# ─── WORKER FUNCTION ────────────────────────────────────────────────────────────
def process_image(args):
    img_path, card_id, subdir = args
    try:
        seed = RANDOM_SEED + os.getpid()
        random.seed(seed)
        np.random.seed(seed)

        print(f"[PID {os.getpid()}] processing {img_path}")

        with Image.open(img_path) as src:
            img_np = np.array(src.convert('RGB'))

        augmentor = A.Compose([
            A.OneOf([
                A.Resize(224, 224),
                A.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            ], p=1.0),
            A.Affine(rotate=(-5, 5), shear=(-2, 2)),
            A.Perspective(scale=(0.01, 0.05)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.1),
            A.Blur(blur_limit=3, p=0.3),
        ], p=1.0)

        train_out = os.path.join(TRAIN_DIR, subdir, card_id)
        val_out   = os.path.join(VAL_DIR,   subdir, card_id)
        for d in (train_out, val_out):
            os.makedirs(d, exist_ok=True)

        for i in range(VARIANTS_PER):
            aug = augmentor(image=img_np)['image']
            dst = val_out if random.random() < VAL_SPLIT else train_out
            out_p = os.path.join(dst, f"{card_id}_{i}.png")
            out_img = Image.fromarray(aug)
            out_img.save(out_p)
            out_img.close()

    except Exception as e:
        print(f"[PID {os.getpid()}] ERROR on {img_path}: {e}")

# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Clean old output dirs only in main process
    for d in (TRAIN_DIR, VAL_DIR):
        if os.path.exists(d):
            print(f"Removing existing directory: {d}")
            shutil.rmtree(d)

    for d in (TRAIN_DIR, VAL_DIR):
        os.makedirs(d, exist_ok=True)
        print(f"→ ensuring output dir exists: {d}")

    all_images = []
    for root, _, files in os.walk(SOURCE_DIR):
        for img_name in files:
            if img_name.lower().endswith(ALLOWED_EXT):
                img_path = os.path.join(root, img_name)
                card_id = os.path.splitext(img_name)[0]
                subdir = os.path.relpath(root, SOURCE_DIR).split(os.sep)[0]
                all_images.append((img_path, card_id, subdir))
    print(f"Found {len(all_images)} source images in {SOURCE_DIR!r}")

    cpu_count = multiprocessing.cpu_count()
    n_workers = max(1, int(cpu_count * 0.8))
    print(f"Launching {n_workers}/{cpu_count} workers…")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pool.map(process_image, all_images)

    print("Dataset generation complete.")
