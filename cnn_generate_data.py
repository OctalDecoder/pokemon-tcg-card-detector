import os
import random
import numpy as np
from PIL import Image
import albumentations as A
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR   = os.path.join(BASE_DIR, 'images', 'cards')
TRAIN_DIR    = os.path.join(BASE_DIR, 'dataset/cnn', 'train')
VAL_DIR      = os.path.join(BASE_DIR, 'dataset/cnn', 'val')
VAL_SPLIT    = 0.2
VARIANTS_PER = 80
RANDOM_SEED  = 42
ALLOWED_EXT  = ('.png', '.jpg', '.jpeg', '.bmp')

# ─── PREPARE OUTPUT DIRS ────────────────────────────────────────────────────────
for d in (TRAIN_DIR, VAL_DIR):
    os.makedirs(d, exist_ok=True)
    print(f"→ ensuring output dir exists: {d}")

# ─── BUILD LIST OF ALL IMAGES ───────────────────────────────────────────────────
all_images = []
for root, _, files in os.walk(SOURCE_DIR):
    for img_name in files:
        if img_name.lower().endswith(ALLOWED_EXT):
            all_images.append((os.path.join(root, img_name),
                               os.path.splitext(img_name)[0]))
print(f"Found {len(all_images)} source images in {SOURCE_DIR!r}")

# ─── WORKER FUNCTION ────────────────────────────────────────────────────────────
def process_image(args):
    img_path, card_id = args
    try:
        # per-process seeding
        seed = RANDOM_SEED + os.getpid()
        random.seed(seed)
        np.random.seed(seed)

        print(f"[PID {os.getpid()}] processing {img_path}")

        # build augmentor
        augmentor = A.Compose([
            A.OneOf([
                A.Resize(224, 224),
                A.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.5, 1.0),
                    ratio=(0.8, 1.2)
                ),
            ], p=1.0),
            A.Affine(rotate=(-5, 5), shear=(-2, 2)),
            A.Perspective(scale=(0.01, 0.05)),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.1
            ),
            A.Blur(blur_limit=3, p=0.3),
        ], p=1.0)

        # load image
        src = Image.open(img_path).convert('RGB')
        img_np = np.array(src)

        # prepare per-card folders
        train_out = os.path.join(TRAIN_DIR, card_id)
        val_out   = os.path.join(VAL_DIR,   card_id)
        for d in (train_out, val_out):
            os.makedirs(d, exist_ok=True)

        # generate variants
        for i in range(VARIANTS_PER):
            aug = augmentor(image=img_np)['image']
            dst = val_out if random.random() < VAL_SPLIT else train_out
            out_p = os.path.join(dst, f"{card_id}_{i}.png")
            Image.fromarray(aug).save(out_p)
            print(f"  saved → {out_p}")

    except Exception as e:
        print(f"[PID {os.getpid()}] ERROR on {img_path}: {e}")

# ─── PARALLEL EXECUTION ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    n_workers = max(1, int(cpu_count * 0.8))
    print(f"Launching {n_workers}/{cpu_count} workers…")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # schedule and wait for all tasks
        pool.map(process_image, all_images)

    print("Dataset generation complete.")
