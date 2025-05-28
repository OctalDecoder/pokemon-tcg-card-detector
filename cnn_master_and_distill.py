"""
cnn_master_and_distill.py  (auto-detects sub-categories)

Workflow
--------
1. **Master** - train an EfficientNet-B0 on the *entire* dataset under
   `dataset/cnn/train` and `dataset/cnn/val`.
2. **Students** - for **every** immediate sub-folder inside `train/` (e.g.
   `fullart`, `standard`, …) distil a MobileNetV3-Small using the master as
   teacher.

Run:
    python cnn_master_and_distill.py --epochs-master 10 --epochs-student 6
Add `--student-only` once the master checkpoint exists to skip re-training it.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports & Config
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from typing import Tuple, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)

# ----------------- Hyper-parameters (CLI-override) ---------------------------
BATCH_SIZE    = 32
WEIGHT_DECAY  = 1e-2
LR_MASTER     = 3e-4
LR_STUDENT    = 2e-4
T_KD          = 4.0   # temperature for distillation
ALPHA_CE      = 0.2   # weight for CE vs KD loss
NUM_WORKERS   = max(2, os.cpu_count() // 2)
IMG_SIZE      = 224

# ----------------- Paths -----------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
DATA_ROOT   = BASE_DIR / "dataset" / "cnn"
TRAIN_DIR   = DATA_ROOT / "train"
VAL_DIR     = DATA_ROOT / "val"
MASTER_CKPT = DATA_ROOT / "cnn_master_best.pth"
RAW_IMG_ROOT = BASE_DIR / "images" / "cards"

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

class FlatDataset(Dataset):
    """
    Walk a list of `parent_dirs`, each of which contains
    one folder per class.  Build samples=(path,label) and
    a unified class_to_idx mapping.
    """
    def __init__(self, parent_dirs, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        idx = 0

        for parent in parent_dirs:
            # e.g. parent = dataset/cnn/train/fullart
            for class_dir in sorted(parent.iterdir()):
                if not class_dir.is_dir():
                    continue
                cname = class_dir.name
                if cname not in self.class_to_idx:
                    self.class_to_idx[cname] = idx
                    idx += 1
                label = self.class_to_idx[cname]

                for img_path in class_dir.glob("*"):
                    # you may wish to filter by suffix: .jpg/.png etc
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def build_flat_loaders(train_parents: List[Path],
                       val_parents:   List[Path]
                      ) -> Tuple[DataLoader, DataLoader, int, dict[str, int], List[Tuple[str, int]]]:
    tr_ds = FlatDataset(train_parents, transform=make_transforms(True))
    va_ds = FlatDataset(val_parents,   transform=make_transforms(False))

    train_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_ld   = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    n_classes = len(tr_ds.class_to_idx)
    class_to_idx = tr_ds.class_to_idx
    samples      = tr_ds.samples   # list of (path, label)

    return train_ld, val_ld, n_classes, class_to_idx, samples

def list_subcategories(root: Path) -> List[str]:
    """Return sorted list of immediate subfolder names inside *root*."""
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def make_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406], [.229,.224,.225]),
        ])
    return transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225]),
    ])

def build_loaders(train_path: Path, val_path: Path) -> Tuple[DataLoader, DataLoader, int]:
    train_ds = tv.datasets.ImageFolder(str(train_path), transform=make_transforms(True))
    val_ds   = tv.datasets.ImageFolder(str(val_path),   transform=make_transforms(False))
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    return train_ld, val_ld, len(train_ds.classes)

def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return (logits.argmax(dim=1) == target).float().mean().item()

# ──────────────────────────────────────────────────────────────────────────────
# Loss & epoch util
# ──────────────────────────────────────────────────────────────────────────────
class DistillationLoss(nn.Module):
    def __init__(self, alpha: float, temperature: float):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
    def forward(self, s_logit, t_logit, y):
        ce = self.ce(s_logit, y)
        kd = self.kl(
            nn.functional.log_softmax(s_logit/self.T, dim=1),
            nn.functional.softmax(t_logit/self.T, dim=1)
        ) * self.T**2
        return self.alpha*ce + (1-self.alpha)*kd

def run_epoch(model, loader, crit, opt=None, distill=False, teacher=None, device="cpu"):
    train = opt is not None
    model.train() if train else model.eval()
    loss_sum = acc_sum = total = 0
    for xb,yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train: opt.zero_grad()
        out = model(xb)
        if distill and teacher is not None:
            with torch.no_grad():
                t_out = teacher(xb)
            loss = crit(out, t_out, yb)
        else:
            loss = crit(out, yb)
        if train:
            loss.backward(); opt.step()
        bs = yb.size(0)
        loss_sum += loss.item()*bs
        acc_sum  += accuracy_top1(out, yb)*bs
        total    += bs
    return loss_sum/total, acc_sum/total

# ──────────────────────────────────────────────────────────────────────────────
# Master training
# ──────────────────────────────────────────────────────────────────────────────

def train_master(epochs:int, device:str):
    # student-category roots:
    master_train_roots = [TRAIN_DIR / cat for cat in list_subcategories(TRAIN_DIR)]
    master_val_roots   = [VAL_DIR   / cat for cat in list_subcategories(VAL_DIR)]

    train_loader, val_loader, nc, class_to_idx, samples = build_flat_loaders(master_train_roots, master_val_roots)
    
    # Generate mappings
    idx_to_card = { idx: name for name, idx in class_to_idx.items() }
    idx_to_image = {}
    for card_name, idx in class_to_idx.items():
        # look in each subfolder (fullart, standard, ...)
        for parent in master_train_roots:
            raw_dir = RAW_IMG_ROOT / parent.name
            matches = list(raw_dir.glob(f"{card_name}.*"))
            if matches:
                idx_to_image[idx] = str(matches[0])
                break
        else:
            idx_to_image[idx] = None

    mapping = {"idx_to_card": idx_to_card,
               "idx_to_image": idx_to_image}
    
    with open(DATA_ROOT / "master_mappings.json", "w") as f:
        json.dump(mapping, f, indent=2)

    # Training Init
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, nc))
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=LR_MASTER, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.StepLR(opt, epochs//2, gamma=0.1)
    crit = nn.CrossEntropyLoss()
    best = 0.0

    # Train
    for ep in range(1, epochs+1):
        tl, ta = run_epoch(model, train_loader, crit, opt, device=device)
        vl, va_acc = run_epoch(model, val_loader, crit, device=device)
        sch.step()
        print(f"[Master {ep}/{epochs}] train {ta:.3f} | val {va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), MASTER_CKPT)
            print("  ↳ new best saved")

# ──────────────────────────────────────────────────────────────────────────────
# Student distillation
# ──────────────────────────────────────────────────────────────────────────────

def distil_student(cat:str, teacher_ckpt:Path, epochs:int, device:str):
    tr_p, va_p = TRAIN_DIR/cat, VAL_DIR/cat
    train_loader, val_loader, nc = build_loaders(tr_p, va_p)
    
    # Generate Mappings
    train_ds = train_loader.dataset
    idx_to_card = { idx: cls
                    for cls, idx in train_ds.class_to_idx.items() }
    raw_dir = RAW_IMG_ROOT / cat
    idx_to_image = {}
    for cls, idx in train_ds.class_to_idx.items():
        matches = list(raw_dir.glob(f"{cls}.*"))
        idx_to_image[idx] = str(matches[0]) if matches else None

    mapping = {"idx_to_card":  idx_to_card,
               "idx_to_image": idx_to_image}

    out_fn = DATA_ROOT / f"{cat}_mappings.json"
    with open(out_fn, "w") as f:
        json.dump(mapping, f, indent=2)

    # ----- Teacher -----
    teacher = efficientnet_b0()
    in_ft = teacher.classifier[1].in_features
    teacher.classifier[1] = nn.Linear(in_ft, nc)

    # load only the backbone weights, drop the old 1098-way head
    ckpt = torch.load(teacher_ckpt, map_location=device)
    for key in ["classifier.1.weight", "classifier.1.bias"]:
        ckpt.pop(key, None)

    missing, unexpected = teacher.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"    ↳ dropped head params, loaded {len(ckpt)} backbone keys")

    teacher.eval().to(device)

    # ----- Student -----
    student = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_fs = student.classifier[3].in_features
    student.classifier[3] = nn.Linear(in_fs, nc)
    student.to(device)

    crit = DistillationLoss(ALPHA_CE, T_KD)
    opt  = optim.AdamW(student.parameters(), lr=LR_STUDENT, weight_decay=WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best=0.; out_ckpt = DATA_ROOT/f"cnn_{cat}_student.pth"
    for ep in range(1,epochs+1):
        train_loss, train_acc = run_epoch(student, train_loader, crit, opt, True, teacher, device)
        val_loss,   val_acc   = run_epoch(student, val_loader, crit, None,  True, teacher, device)
        sch.step()
        print(f"  [Student-{cat} {ep}/{epochs}] "
                f"train {train_acc:.3f} | val {val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save(student.state_dict(), out_ckpt)
            print("    ↳ best saved")
    print(f"Finished '{cat}' - best val {best:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI & main
# ──────────────────────────────────────────────────────────────────────────────

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs-master", type=int, default=10)
    p.add_argument("--epochs-student", type=int, default=6)
    p.add_argument("--student-only", action="store_true")
    return p.parse_args()

if __name__=="__main__":
    args=parse(); dev="cuda" if torch.cuda.is_available() else "cpu"
    if not args.student_only:
        print("Training master …"); train_master(args.epochs_master, dev)
    else:
        assert MASTER_CKPT.exists(), "Master checkpoint missing. Run without --student-only first."
    cats = list_subcategories(TRAIN_DIR)
    print("Sub-categories detected:", ", ".join(cats))
    for c in cats:
        print(f"\nDistilling {c} …"); distil_student(c, MASTER_CKPT, args.epochs_student, dev)
    print("\nAll students distilled ✔")
