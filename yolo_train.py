import multiprocessing
from ultralytics import YOLO

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
MODEL_NAME      = "cd2"
DATA_CONFIG     = 'dataset/yolo/cards.yaml'
PRETRAINED_YOLO = f'runs/detect/{MODEL_NAME}/weights/best.pt'    # additional training model
# PRETRAINED_YOLO = 'yolo8n.pt'    # base model
EPOCHS          = 25
IMG_SIZE        = 640
BATCH_SIZE      = 16
DEVICE          = 0              # GPU index, or 'cpu'
LR0             = 1e-3           # DEFAULT: 1e-2 lower to 1e-3 or 4 for fine tuning
LR_FINAL_FACTOR = 0.01            # DEFAULT: 0.01 final LR = LR0 * LR_FINAL_FACTOR
PATIENCE        = 5              # DEFAULT: OFF early‐stop after this many epochs w/o val‐map gain
# ────────────────────────────────────────────────────────────────────────────────

def train():
    # Load the model (either fresh or resumed)
    model = YOLO(PRETRAINED_YOLO)

    # Kick off training
    model.train(
        data       = DATA_CONFIG,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        device     = DEVICE,
        name       = MODEL_NAME,
        
        # Hyper Parameters
        lr0         = LR0,
        lrf        = LR_FINAL_FACTOR,
        patience   = PATIENCE,
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
