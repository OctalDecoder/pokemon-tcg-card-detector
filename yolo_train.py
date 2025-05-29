import multiprocessing
from ultralytics import YOLO

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
RESUME_TRAINING = False
MODEL_NAME      = "cd2"
DATA_CONFIG     = 'dataset/cards.yaml'
PRETRAINED_YOLO = 'yolov8n.pt'    # base model
EPOCHS          = 25
IMG_SIZE        = 640
BATCH_SIZE      = 16
DEVICE          = 0              # GPU index, or 'cpu'
LR0             = 1e-2           # DEFAULT: 1e-2 start LR (10× lower than default ~1e-3)
LR_FINAL_FACTOR = 0.01            # DEFAULT: 0.01 final LR = LR0 * LR_FINAL_FACTOR
PATIENCE        = 5              # DEFAULT: OFF early‐stop after this many epochs w/o val‐map gain
# ────────────────────────────────────────────────────────────────────────────────

def train():
    # If we want to resume from our own best.pt rather than the official yolov8n.pt:
    ckpt = f"runs/detect/{MODEL_NAME}/weights/best.pt"
    model_path = ckpt if RESUME_TRAINING else PRETRAINED_YOLO

    # Load the model (either fresh or resumed)
    model = YOLO(model_path)

    # Kick off training
    model.train(
        data       = DATA_CONFIG,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        device     = DEVICE,
        name       = MODEL_NAME,
        resume     = RESUME_TRAINING,
        # ─── hyperparameters ───────────────────────
        lr0         = LR0,
        lrf        = LR_FINAL_FACTOR,
        # patience   = PATIENCE,
        # ────────────────────────────────────────────
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
