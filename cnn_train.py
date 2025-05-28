import os
import torch
import torchvision as tv
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR    = 'dataset/cnn/train'           # Location to store training images
VAL_DIR    = 'dataset/cnn/val'               # Location to store evaluation images
BEST_MODEL_PATH = 'dataset/cnn/cnn_best.pth' # Location to store best model
BATCH_SIZE  = 32
NUM_WORKERS = min(8, os.cpu_count() // 2)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS1 = 5       # epochs head only
NUM_EPOCHS2 = 5       # epochs fine-tuning
LR_HEAD     = 3e-4
LR_FINE     = 1e-4
WD          = 1e-2

# ─── DATA TRANSFORMS ───────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406],
                         std=[.229, .224, .225]),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406],
                         std=[.229, .224, .225]),
])

# ─── DATA LOADERS ──────────────────────────────────────────────────────────────
train_ds = tv.datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = tv.datasets.ImageFolder(VAL_DIR,   transform=val_tf)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_ds,   batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# ─── MODEL SETUP ───────────────────────────────────────────────────────────────
# 1) Load EfficientNet-B0 with ImageNet weights
weights = EfficientNet_B0_Weights.DEFAULT
model   = efficientnet_b0(weights=weights)

# 2) Freeze the backbone feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# 3) Replace the classifier head
num_classes = len(train_ds.classes)
in_feats    = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_feats, num_classes)
)

model = model.to(DEVICE)

# ─── OPTIMIZER & SCHEDULER ─────────────────────────────────────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_HEAD, weight_decay=WD
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# ─── UTILITY: compute top-k accuracy ────────────────────────────────────────────
def accuracy(output, target, topk=(1,5)):
    """Returns a list of top-k accuracies for the given outputs."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # shape: [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=False)
            res.append((correct_k / batch_size).item())
        return res  # e.g. [top1, top5]

# ─── TRAIN & EVAL LOOPS ────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    epoch_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total = 0

    if train:
        model.train()
    else:
        model.eval()

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train:
            optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        if train:
            loss.backward()
            optimizer.step()

        # accumulate metrics
        bs = xb.size(0)
        epoch_loss += loss.item() * bs
        a1, a5 = accuracy(logits, yb)
        top1_acc += a1 * bs
        top5_acc += a5 * bs
        total += bs

    avg_loss = epoch_loss / total
    avg_top1 = top1_acc / total
    avg_top5 = top5_acc / total
    return avg_loss, avg_top1, avg_top5

if __name__ == '__main__':
    print(f"Training head only for {NUM_EPOCHS1} epochs...")
    for epoch in range(1, NUM_EPOCHS1 + 1):
        train_loss, train_1, train_5 = run_epoch(train_loader, train=True)
        val_loss,   val_1,   val_5   = run_epoch(val_loader,   train=False)
        scheduler.step()

        print(f"[Stage 1][Epoch {epoch}] "
              f"train loss {train_loss:.4f}, top1 {train_1:.4f}, top5 {train_5:.4f} | "
              f"val loss {val_loss:.4f}, top1 {val_1:.4f}, top5 {val_5:.4f}")

    # ─── Optional Stage 2: unfreeze and fine-tune entire model ─────────────────
    print("Unfreezing backbone for fine-tuning...")
    for param in model.features.parameters():
        param.requires_grad = True

    # adjust optimizer for fine tuning
    optimizer = optim.AdamW(model.parameters(), lr=LR_FINE, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val1 = 0.0
    print(f"Fine-tuning for {NUM_EPOCHS2} epochs...")
    for epoch in range(1, NUM_EPOCHS2 + 1):
        train_loss, train_1, train_5 = run_epoch(train_loader, train=True)
        val_loss,   val_1,   val_5   = run_epoch(val_loader,   train=False)
        scheduler.step()

        print(f"[Stage 2][Epoch {epoch}] "
              f"train loss {train_loss:.4f}, top1 {train_1:.4f}, top5 {train_5:.4f} | "
              f"val loss {val_loss:.4f}, top1 {val_1:.4f}, top5 {val_5:.4f}")
        if val_1 > best_val1:
            best_val1 = val_1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best model (val1={best_val1:.4f}) to {BEST_MODEL_PATH}")

    print("Training complete.")
