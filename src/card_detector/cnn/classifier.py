"""
cnn_classifier.py

CNN-based classifier for per-category card recognition.

Classes:
    CnnClassifier:
        - __init__(subcats, cnn_model_dir, conf_threshold=0.15, device='cuda'):
            Initialize with a list of subcategories, load mapping JSON and MobileNetV3 models 
            from `cnn_model_dir`, and set confidence threshold and device.
        - _load_models(cnn_model_dir):
            Load `cnn_mappings.json`, convert keys to int, instantiate and load each MobileNetV3 
            checkpoint (cnn_<subcat>_student.pth), and store models and label maps.
        - classify(images, cats):
            Given a list of PIL images and corresponding YOLO category IDs, batch-transform 
            by subcategory, run through the appropriate MobileNetV3 model, apply softmax, 
            filter by `conf_threshold`, and return a list of high-confidence card IDs.

Usage Example:
    from card_detector.cnn.classifier import CnnClassifier

    subcategories = [1, 2, 3]
    cnn_dir = "models/cnn"
    classifier = CnnClassifier(subcategories, cnn_model_dir=cnn_dir, conf_threshold=0.2, device="cuda")

    # images: list of PIL.Image crops, cats: matching list of subcategory IDs
    labels = classifier.classify(images, cats)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
import json

class CnnClassifier:
    def __init__(self, subcats, cnn_model_dir, conf_threshold=0.15, device='cuda'):
        self.subcats = subcats
        self.device = device
        self.conf_threshold = conf_threshold
        self.child_models = {}
        self.child_maps = {}
        self.clf_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
        ])
        self._load_models(cnn_model_dir)
        
        print("CUDA available:", torch.cuda.is_available())
        print("YOLO model device:", next(self.child_models["standard"].parameters()).device)

    def _load_models(self, cnn_model_dir):
        with open(f"{cnn_model_dir}/cnn_mappings.json") as f:
            raw = json.load(f)
        for cat in self.subcats:
            self.child_maps[cat] = {int(k): v for k, v in raw[cat].items()}
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            in_f = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_f, len(raw[cat]))
            ckpt = torch.load(f"{cnn_model_dir}/cnn_{cat}_student.pth", map_location=self.device)
            model.load_state_dict(ckpt)
            model.to(self.device).eval()
            self.child_models[cat] = model

    def classify(self, images, cats):
        output = [None] * len(images)
        confs = [0.0] * len(images)
        for subcat in set(cats):
            idxs = [i for i, c in enumerate(cats) if c == subcat]
            batch = torch.stack([self.clf_tf(images[i]) for i in idxs], dim=0).to(self.device)
            with torch.no_grad():
                logits = self.child_models[subcat](batch)
                softm = F.softmax(logits, dim=1).cpu()
            sub_preds = logits.argmax(1).cpu().tolist()
            for local_i, (i, p) in enumerate(zip(idxs, sub_preds)):
                output[i] = self.child_maps[subcat][p]
                confs[i] = float(softm[local_i, p])
        filtered_output = []
        for o, c in zip(output, confs):
            if c >= self.conf_threshold:
                filtered_output.append(o)
        return filtered_output
