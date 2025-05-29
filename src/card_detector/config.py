# src/card_detector/config.py
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
