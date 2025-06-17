import yaml
from pathlib import Path
from copy import deepcopy

DEFAULT_CONFIG_NAME = "config.yaml"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

def load_config(config_path=None, section=None):
    """
    Loads a config YAML file, supports 'shared' dict merging.
    - config_path: str | Path | None
      If None, loads the default config.
      If basename only, loads from configs/ directory.
    - section: str
      If None, loads the entire config
      If given, loads the config section
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_DIR / DEFAULT_CONFIG_NAME
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = DEFAULT_CONFIG_DIR / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "shared" in cfg:
        shared = cfg["shared"]
        for key in cfg:
            if key != "shared" and isinstance(cfg[key], dict):
                cfg[key] = {**deepcopy(shared), **cfg[key]}
    
    if section is not None:
        return cfg[section]
    return cfg

# Backwards Compatibility
cfg = load_config()
