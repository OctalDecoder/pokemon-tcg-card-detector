import logging
from card_detector.config import cfg
from card_detector.detectors.screenshot_pipeline import ScreenshotPipeline

# Set up logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
screenshot_pipeline_cfg = cfg["screenshot_pipeline"]
logger.setLevel(logging.DEBUG if screenshot_pipeline_cfg["debug"] else logging.INFO)
logging.getLogger('ultralytics').setLevel(logging.DEBUG if screenshot_pipeline_cfg["debug"] else logging.WARNING)

# Prepare configs
yolo_cfg = {
    "yolo_model": screenshot_pipeline_cfg["yolo_model"],
    "yolo_conf_thresh": screenshot_pipeline_cfg["yolo_conf_thresh"],
    "bbox_iou_thresh": screenshot_pipeline_cfg["bbox_iou_thresh"]
}
cnn_cfg = {
    "classifiers": screenshot_pipeline_cfg["classifiers"],
    "cnn_model_dir": screenshot_pipeline_cfg["cnn_model_dir"],
    "cnn_conf_threshold": screenshot_pipeline_cfg["cnn_conf_thresh"]
}

# Run pipeline
pipeline = ScreenshotPipeline(yolo_cfg, cnn_cfg, screenshot_pipeline_cfg, logger=logger)
logger.info("Starting pipeline with batch classification...")
results = pipeline.process_images()
logger.info(results)
