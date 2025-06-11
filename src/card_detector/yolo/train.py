"""yolo/train.py

Training script for a YOLOv8 nano model
"""

def train(args, logger):
    """
    Train a YOLOv8 model, using either config or CLI overrides.
    """
    import multiprocessing
    from pathlib import Path
    import shutil
    from ultralytics import YOLO
    
    from card_detector.config import cfg

    ycfg: dict = cfg["yolo"].copy()

    # Override YAML with CLI if specified
    if args.epochs is not None:      ycfg["epochs"]      = args.epochs
    if args.batch_size is not None:  ycfg["batch_size"]  = args.batch_size
    if args.img_size is not None:    ycfg["img_size"]    = args.img_size
    if args.device is not None:      ycfg["device"]      = args.device
    if args.model_name is not None:  ycfg["model_name"]  = args.model_name
    if args.lr0 is not None:         ycfg["lr0"]         = args.lr0
    if args.lr_final_factor is not None: ycfg["lr_final_factor"] = args.lr_final_factor
    if args.patience is not None:    ycfg["patience"]    = args.patience
    if args.data_config is not None: ycfg["data_config"] = args.data_config
    if args.yolo_model is not None:  ycfg["yolo"]        = args.yolo_model

    multiprocessing.freeze_support()
    
    model_name = ycfg["model"]  # e.g. "yolov8n.pt"
    model_path = Path(ycfg["training_data_dir"]) / "yolo"
    model_path.mkdir(parents=True, exist_ok=True)
    full_model_path = model_path / model_name

    if full_model_path.exists():
        model = YOLO(str(full_model_path))
        first_run = False
    else:
        # Trigger download of model
        logger.warning(f"Model '{full_model_path}' not found. Reverting to default.")
        model = YOLO(model_name)
        first_run = True

    model.train(
        project  = ycfg["output_dir"] + "yolo",
        data     = ycfg["data_config"],
        epochs   = ycfg["epochs"],
        imgsz    = ycfg["img_size"],
        batch    = ycfg["batch_size"],
        device   = ycfg["device"],
        name     = ycfg["model_name"],
        lr0      = ycfg["lr0"],
        lrf      = ycfg["lr_final_factor"],
        patience = ycfg["patience"],
    )

    # Move downloaded model to data after first training
    if first_run:
        src = Path(model_name)
        if src.exists():
            dest = full_model_path
            logger.info(f"Moving {src} -> {dest}")
            shutil.move(str(src), str(dest))
        else:
            logger.warning(f"Expected downloaded model at {src}, but not found!")
