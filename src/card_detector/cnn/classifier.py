# detectors/cnn_classifier.py
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

    def _load_models(self, cnn_model_dir):
        with open(f"{cnn_model_dir}/cnn_mappings.json") as f:
            raw = json.load(f)["students"]
        for cat in self.subcats:
            raw_student = raw[cat]
            idx2card = {int(k): v for k, v in raw_student["idx_to_card"].items()}
            idx2image = {int(k): v for k, v in raw_student["idx_to_image"].items()}
            self.child_maps[cat] = (idx2card, idx2image)
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            in_f = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_f, len(idx2card))
            ckpt = torch.load(f"{cnn_model_dir}/cnn_{cat}_student.pth", map_location=self.device)
            model.load_state_dict(ckpt)
            model.to(self.device).eval()
            self.child_models[cat] = model

    def classify(self, images, cats):
        output = [None] * len(images)
        matches = [None] * len(images)
        confs = [0.0] * len(images)
        for subcat in set(cats):
            idxs = [i for i, c in enumerate(cats) if c == subcat]
            batch = torch.stack([self.clf_tf(images[i]) for i in idxs], dim=0).to(self.device)
            with torch.no_grad():
                logits = self.child_models[subcat](batch)
                softm = F.softmax(logits, dim=1).cpu()
            sub_preds = logits.argmax(1).cpu().tolist()
            idx2card, idx2img = self.child_maps[subcat]
            for local_i, (i, p) in enumerate(zip(idxs, sub_preds)):
                output[i] = idx2card[p]
                matches[i] = idx2img[p]
                confs[i] = float(softm[local_i, p])
        filtered_output = []
        filtered_matches = []
        for o, m, c in zip(output, matches, confs):
            if c >= self.conf_threshold:
                filtered_output.append(o)
                filtered_matches.append(m)
        return filtered_output, filtered_matches
