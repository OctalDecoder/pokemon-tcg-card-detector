import logging
from card_detector.config import cfg
from card_detector.detectors.screenshot_pipeline import ScreenshotPipeline

# Set up logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
pcfg = cfg["prototype"]
logger.setLevel(logging.DEBUG if pcfg["debug"] else logging.INFO)
logging.getLogger('ultralytics').setLevel(logging.DEBUG if pcfg["debug"] else logging.WARNING)

# Prepare configs
yolo_cfg = {
    "yolo_model": pcfg["yolo_model"],
    "yolo_conf_thresh": pcfg["yolo_conf_thresh"],
    "bbox_iou_thresh": pcfg["bbox_iou_thresh"]
}
cnn_cfg = {
    "cnn_subcats": pcfg["cnn_subcats"],
    "cnn_base_dir": pcfg["cnn_base_dir"],
    "cnn_conf_threshold": pcfg["cnn_conf_thresh"]
}

# Run pipeline
pipeline = CardDetectionPipeline(yolo_cfg, cnn_cfg, pcfg, logger=logger)
logger.info("Starting pipeline with batch classification...")
results = pipeline.process_images()
logger.info(results)
