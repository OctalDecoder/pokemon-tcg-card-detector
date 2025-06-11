"""cnn/train.py
Utility script to train an EfficientNet-B0 "master" model, then distill 
MobileNetV3-Small "students".
"""

import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)

from card_detector.config import load_config

# Load configuration
CONFIG = load_config(config_path="training.yaml", section="cnn")

# Hyperparameters and settings from config
BATCH_SIZE   = CONFIG["batch_size"]
WEIGHT_DECAY = CONFIG["weight_decay"]
LR_MASTER    = CONFIG["lr_master"]
LR_STUDENT   = CONFIG["lr_student"]
T_KD         = CONFIG["t_kd"]           # Knowledge distillation temperature
ALPHA_CE     = CONFIG["alpha_ce"]       # Blend between CE and KD loss
NUM_WORKERS  = max(2, os.cpu_count() // 2)
IMG_SIZE     = CONFIG["img_size"]

# Paths for data and outputs
RAW_IMG_ROOT = Path(CONFIG["card_images_dir"])
OUTPUT_DIR   = Path(CONFIG["output_dir"], "cnn")
MASTER_CKPT  = OUTPUT_DIR / "cnn_master_best.pth"
MAPPINGS_PATH = OUTPUT_DIR / "cnn_mappings.json"
DATA_ROOT    = Path(CONFIG["training_data_dir"], "cnn")
TRAIN_DIR    = DATA_ROOT / "train"
VAL_DIR      = DATA_ROOT / "val"

# ─────────────── Mappings I/O ───────────────

def load_all_mappings():
    if MAPPINGS_PATH.exists():
        with open(MAPPINGS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_all_mappings(mappings):
    with open(MAPPINGS_PATH, "w") as f:
        json.dump(mappings, f, indent=2)

def save_master_mapping(class_to_idx, master_train_roots):
    """Save mapping for master model classes to card names and example image paths."""
    idx_to_card = {idx: name for name, idx in class_to_idx.items()}
    idx_to_image = {}
    for card_name, idx in class_to_idx.items():
        # Find an example image file for this card in the raw images directory
        for parent in master_train_roots:
            matches = list((RAW_IMG_ROOT / parent.name).glob(f"{card_name}.*"))
            if matches:
                idx_to_image[str(idx)] = str(matches[0])
                break
        else:
            idx_to_image[str(idx)] = None  # No image found for this card
    mappings = load_all_mappings()
    mappings["master"] = {str(k): v for k, v in idx_to_card.items()}
    save_all_mappings(mappings)

def save_student_mapping(category, class_to_idx):
    """Save mapping for a student model (category-specific) classes to card names."""
    idx_to_card = {idx: cls for cls, idx in class_to_idx.items()}
    mappings = load_all_mappings()
    mappings[category] = {str(k): v for k, v in idx_to_card.items()}
    save_all_mappings(mappings)

# ─────────────── Visual Plotting Helpers ────────────────────
def show_training_curves(history, title="Training Curves"):
    """Display training and validation loss/accuracy curves using matplotlib."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ─────────────── Dataset and DataLoader Utilities ───────────────

class FlatDataset(Dataset):
    """
    A Dataset that flattens a set of category folders into one dataset.
    Each parent directory in `parent_dirs` represents a subset of classes.
    This allows combining multiple folders (e.g., multiple training sets).
    """
    def __init__(self, parent_dirs: List[Path], transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        idx = 0
        for parent in parent_dirs:
            for class_dir in sorted(parent.iterdir()):
                if not class_dir.is_dir():
                    continue
                cname = class_dir.name
                if cname not in self.class_to_idx:
                    # Assign a new index to each new class name encountered
                    self.class_to_idx[cname] = idx
                    idx += 1
                label = self.class_to_idx[cname]
                # Add all images from this class directory
                for img_path in class_dir.glob("*"):
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def build_flat_loaders(train_parents: List[Path], val_parents: List[Path]):
    """
    Build training and validation DataLoaders by combining multiple parent directories.
    Returns DataLoader for train and val, number of classes, class_to_idx mapping, and samples list.
    """
    train_dataset = FlatDataset(train_parents, transform=make_transforms(train=True))
    val_dataset   = FlatDataset(val_parents,   transform=make_transforms(train=False))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    n_classes = len(train_dataset.class_to_idx)
    return train_loader, val_loader, n_classes, train_dataset.class_to_idx, train_dataset.samples

def list_subcategories(root: Path) -> List[str]:
    """List all subdirectory names (categories) under the given root path."""
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def make_transforms(train: bool) -> transforms.Compose:
    """Compose image transformations for training or validation."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

def build_loaders(train_path: Path, val_path: Path):
    """
    Build DataLoaders for a single category (for student model training) using ImageFolder.
    """
    train_ds = tv.datasets.ImageFolder(str(train_path), transform=make_transforms(train=True))
    val_ds   = tv.datasets.ImageFolder(str(val_path),   transform=make_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)

# ─────────────── Training Utilities ───────────────

def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch of predictions."""
    with torch.no_grad():
        return (logits.argmax(dim=1) == target).float().mean().item()

class DistillationLoss(nn.Module):
    """
    Custom loss for knowledge distillation: combines cross-entropy with KL-divergence.
    """
    def __init__(self, alpha: float, temperature: float):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
    def forward(self, s_logit, t_logit, y):
        # Standard cross-entropy loss with true labels
        ce_loss = self.ce(s_logit, y)
        # Knowledge distillation loss (KL divergence between student & teacher outputs)
        kd_loss = self.kl(
            nn.functional.log_softmax(s_logit / self.T, dim=1),
            nn.functional.softmax(t_logit / self.T, dim=1)
        ) * (self.T ** 2)
        # Weighted sum of the two losses
        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss

def run_epoch(model, loader, criterion, optimizer=None, distill=False, teacher=None, device="cpu"):
    """
    Run one training or validation epoch.
    Returns the average loss and accuracy.
    If `optimizer` is provided, the model is in training mode; otherwise in eval mode.
    If `distill` is True and a `teacher` model is provided, uses knowledge distillation.
    """
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    loss_sum = 0.0
    acc_sum = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train_mode:
            optimizer.zero_grad()
        # Forward pass
        outputs = model(xb)
        if distill and teacher is not None:
            # Teacher outputs for distillation
            with torch.no_grad():
                teacher_outputs = teacher(xb)
            loss = criterion(outputs, teacher_outputs, yb)
        else:
            loss = criterion(outputs, yb)
        # Backpropagation and optimizer step (if training)
        if train_mode:
            loss.backward()
            optimizer.step()
        # Accumulate loss and accuracy
        batch_size = yb.size(0)
        loss_sum += loss.item() * batch_size
        acc_sum  += accuracy_top1(outputs, yb) * batch_size
        total_samples += batch_size

    avg_loss = loss_sum / total_samples
    avg_acc  = acc_sum / total_samples
    return avg_loss, avg_acc

# ─────────────── Master Model Training ───────────────

def train_master(epochs: int, device: str, resume_master: bool = False, logger=None) -> float:
    """
    Train the master EfficientNet-B0 model on all training categories combined.
    Returns the best validation accuracy achieved.
    """
    # Prepare combined dataset from all category subfolders
    master_train_dirs = [TRAIN_DIR / cat for cat in list_subcategories(TRAIN_DIR)]
    master_val_dirs   = [VAL_DIR / cat for cat in list_subcategories(VAL_DIR)]
    train_loader, val_loader, num_classes, class_to_idx, _ = build_flat_loaders(master_train_dirs, master_val_dirs)

    # Save class-to-card mappings for the master model
    save_master_mapping(class_to_idx, master_train_dirs)

    # Initialize EfficientNet-B0 model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # Replace the classifier head to match our number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    model.to(device)

    # Optionally resume from a checkpoint if available
    if resume_master and MASTER_CKPT.exists():
        checkpoint = torch.load(MASTER_CKPT, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        msg = f"Loaded existing master checkpoint: {MASTER_CKPT}"
        logger.info(msg) if logger else print(msg)

    # Set up optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR_MASTER, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 2, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # Training loop
    try:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        for ep in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device=device)
            val_loss, val_acc    = run_epoch(model, val_loader, criterion, device=device)
            scheduler.step()
            
            # Record metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Log epoch results
            msg = (f"[Master {ep}/{epochs}] "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} | "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}")
            logger.info(msg) if logger else print(msg)

            # Check for new best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MASTER_CKPT)
                msg_best = "    ↳ new best saved"
                logger.info(msg_best) if logger else print(msg_best)
    except KeyboardInterrupt:
        logger.warning(f"Distillation for Master was terminated. Saving progress and exiting.")

    # Finalize master training
    show_training_curves(history)
    final_msg = f"Master training finished. Best validation accuracy: {best_val_acc:.3f}"
    logger.info(final_msg) if logger else print(final_msg)
    return best_val_acc

# ─────────────── Student Model Distillation ───────────────

def distil_student(category: str, teacher_ckpt: Path, epochs: int, device: str, logger=None) -> float:
    """
    Train (distill) a MobileNetV3-Small student model for the given category,
    using the master model (EfficientNet-B0) as teacher.
    Returns the best validation accuracy for this student.
    """
    train_path = TRAIN_DIR / category
    val_path   = VAL_DIR / category

    # Build DataLoaders for this category
    train_loader, val_loader, num_classes = build_loaders(train_path, val_path)
    # Save class-to-card mappings for this student's classes
    train_dataset = train_loader.dataset  # ImageFolder dataset
    save_student_mapping(category, train_dataset.class_to_idx)

    # Load teacher model (EfficientNet-B0) and adapt to this category's classes
    teacher = efficientnet_b0(weights=None)  # initialize without pre-trained weights for safety
    in_features_t = teacher.classifier[1].in_features
    teacher.classifier[1] = nn.Linear(in_features_t, num_classes)
    # Load the master model weights (except final layer) into teacher model
    teacher_ckpt_data = torch.load(teacher_ckpt, map_location=device)
    # Remove any classifier weights from checkpoint to avoid size mismatch
    teacher_ckpt_data.pop("classifier.1.weight", None)
    teacher_ckpt_data.pop("classifier.1.bias", None)
    teacher.load_state_dict(teacher_ckpt_data, strict=False)
    teacher.eval().to(device)

    # Initialize student MobileNetV3-Small model with ImageNet weights, replace classifier head
    student = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features_s = student.classifier[3].in_features
    student.classifier[3] = nn.Linear(in_features_s, num_classes)
    student.to(device)

    # Set up distillation loss, optimizer, and scheduler
    criterion = DistillationLoss(alpha=ALPHA_CE, temperature=T_KD)
    optimizer = optim.AdamW(student.parameters(), lr=LR_STUDENT, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    student_ckpt = DATA_ROOT / f"cnn_{category}_student.pth"

    # Distillation training loop
    try:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        for ep in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(student, train_loader, criterion, optimizer,
                                            distill=True, teacher=teacher, device=device)
            val_loss, val_acc = run_epoch(student, val_loader, criterion,
                                        distill=True, teacher=teacher, device=device)
            scheduler.step()
            
            # Record metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Log epoch results for this student
            msg = (f"[Student-{category} {ep}/{epochs}] "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} | "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}")
            logger.info(msg) if logger else print(msg)

            # Check for new best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(student.state_dict(), student_ckpt)
                msg_best = "    ↳ best saved"
                logger.info(msg_best) if logger else print(msg_best)\
    
    except KeyboardInterrupt:
        logger.warning(f"Distillation for {category} interrupted by user (KeyboardInterrupt). Saving progress and exiting.")

    # Finalize student training for this category
    show_training_curves(history)
    final_msg = f"Finished distilling '{category}' - best val accuracy: {best_val_acc:.3f}"
    logger.info(final_msg) if logger else print(final_msg)
    return best_val_acc

# ─────────────── Main Training Function ───────────────

def train_cnn(args, logger=None):
    """
    Train the master model (if not in student-only mode), then distill student models for each category.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train master model if required
    if not args.student_only:
        logger.info("Training master model...") if logger else print("Training master model...")
        import time
        start_time = time.time()
        
        epochs_master = args.epochs_master if args.epochs_master is not None else CONFIG["epochs_master"]
        epochs_student = args.epochs_student if args.epochs_student is not None else CONFIG["epochs_student"]
        
        best_master_acc = train_master(epochs_master, device, resume_master=args.resume_master, logger=logger)
        elapsed = time.time() - start_time
        # Format elapsed time
        hours = int(elapsed // 3600); minutes = int((elapsed % 3600) // 60); seconds = elapsed % 60
        if hours > 0:
            time_str = f"{hours}h {minutes:02d}m {seconds:05.2f}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds:05.2f}s"
        else:
            time_str = f"{seconds:.2f}s"
        done_msg = (f"Master model training completed in {time_str}. "
                    f"Best val accuracy: {best_master_acc:.3f}")
        logger.info(done_msg) if logger else print(done_msg)
    else:
        # If only distilling students, ensure master checkpoint is available
        assert MASTER_CKPT.exists(), "Master checkpoint missing. Run without --student-only first."

    # List categories and distill a student model for each
    categories = list_subcategories(TRAIN_DIR)
    logger.info(f"Sub-categories detected: {', '.join(categories)}") if logger else print("Sub-categories detected:", ", ".join(categories))

    for cat in categories:
        # Announce start of distillation for this category
        logger.info(f"Distilling category '{cat}'...") if logger else print(f"\nDistilling category '{cat}'...")
        import time
        start_c = time.time()
        best_val = distil_student(cat, MASTER_CKPT, epochs_student, device, logger=logger)
        elapsed_c = time.time() - start_c
        # Format elapsed time for this category
        hours_c = int(elapsed_c // 3600); minutes_c = int((elapsed_c % 3600) // 60); seconds_c = elapsed_c % 60
        if hours_c > 0:
            time_str_c = f"{hours_c}h {minutes_c:02d}m {seconds_c:05.2f}s"
        elif minutes_c > 0:
            time_str_c = f"{minutes_c}m {seconds_c:05.2f}s"
        else:
            time_str_c = f"{seconds_c:.2f}s"
        done_msg = (f"Completed distilling '{cat}' in {time_str_c}. "
                    f"Best val accuracy: {best_val:.3f}")
        logger.info(done_msg) if logger else print(done_msg)

    # All done
    logger.info("All students distilled ✔") if logger else print("\nAll students distilled ✔")
