"""yolo/train.py

Utility script for training a YOLOv8 model
"""

import multiprocessing
from ultralytics import YOLO
from card_detector.config import cfg

if __name__ == '__main__':
    ycfg = cfg["yolo"]
    
    multiprocessing.freeze_support()
    model = YOLO(ycfg["yolo"])

    # Kick off training
    model.train(
        data   = ycfg["data_config"],
        epochs = ycfg["epochs"],
        imgsz  = ycfg["img_size"],
        batch  = ycfg["batch_size"],
        device = ycfg["device"],
        name   = ycfg["model_name"],
        lr0    = ycfg["lr0"],
        lrf    = ycfg["lr_final_factor"],
        patience=ycfg["patience"],
    )
