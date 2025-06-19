import json
import torch
import torch.nn.utils.prune as prune
from torchvision.models import mobilenet_v3_small

def prune_model_structured(model, amount=0.1):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    return model

def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
    return model

if __name__ == "__main__":
    models = ["standard", "fullart"]
    
    for model_name in models:
        model = mobilenet_v3_small()
        in_f = model.classifier[3].in_features
        with open(f"output/cnn/cnn_mappings.json") as f:
            raw = json.load(f)
        num_classes = len(raw[model_name])
        print(f"{model_name}: {num_classes} classes")
        model.classifier[3] = torch.nn.Linear(in_f, num_classes)
        model.load_state_dict(torch.load(f"output/cnn/cnn_{model_name}_student.pth", map_location="cpu"))
        model.eval()

        pruned_model = prune_model_structured(model, amount=0.1)
        remove_pruning(pruned_model)
        torch.save(pruned_model.state_dict(), f"output/cnn/cnn_{model_name}_student.pth")
