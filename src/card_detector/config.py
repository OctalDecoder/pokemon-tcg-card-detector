import yaml
from pathlib import Path
from copy import deepcopy

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

if "shared" in cfg:
    shared = cfg["shared"]
    for key in cfg:
        if key != "shared" and isinstance(cfg[key], dict):
            # Only update dicts (avoid updating lists etc.)
            cfg[key] = {**deepcopy(shared), **cfg[key]}
