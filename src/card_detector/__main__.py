def main():
    import argparse
    import logging
    from card_detector.config import cfg
    from card_detector.pipeline.screenshot_pipeline import ScreenshotPipeline
    from card_detector.pipeline.video_pipeline import VideoPipeline

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["screenshot", "video"],
        default="screenshot",
        help="Detection mode",
    )
    args = parser.parse_args()

    pcfg = cfg["video_pipeline"] if args.mode == "video" else cfg["screenshot_pipeline"]

    logger.setLevel(logging.DEBUG if pcfg.get("debug", False) else logging.INFO)
    logging.getLogger("ultralytics").setLevel(
        logging.DEBUG if pcfg.get("debug", False) else logging.WARNING
    )

    yolo_cfg = {
        "yolo_model": pcfg["yolo_model"],
        "yolo_conf_thresh": pcfg["yolo_conf_thresh"],
        "bbox_iou_thresh": pcfg["bbox_iou_thresh"],
    }
    cnn_cfg = {
        "classifiers": pcfg["classifiers"],
        "cnn_model_dir": pcfg["cnn_model_dir"],
        "cnn_conf_threshold": pcfg["cnn_conf_thresh"],
    }

    if args.mode == "video":
        pipeline = VideoPipeline(yolo_cfg, cnn_cfg, pcfg, logger=logger)
        logger.info("Starting video pipeline...")
        results = pipeline.process_videos()
        if results is not None:
            logger.info(results)
            logger.info(f"Total results from all videos: {len(results)}")
            logger.warning("There is a minor bug where the video finishes but the classifications are still queued. The above output will always be the full TRUE list.")
    else:
        pipeline = ScreenshotPipeline(yolo_cfg, cnn_cfg, pcfg, logger=logger)
        logger.info("Starting pipeline with batch classification...")
        results = pipeline.process_images()
        if results is not None:
            logger.info(results)


if __name__ == "__main__":
    main()
