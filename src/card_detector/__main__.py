def main():
    import argparse
    import logging
    from card_detector.config import cfg
    from card_detector.pipeline.screenshot_pipeline import ScreenshotPipeline
    from card_detector.pipeline.video_pipeline import VideoPipeline
    from card_detector.util.logging import print_section_header

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
    parser.add_argument(
        "-v",
        "--video",
        action="store_true",
        help="Shortcut for --mode video",
    )
    parser.add_argument(
        "-hl",
        "--headless",
        action="store_true",
        help="Run detection without rendering video or saving results",
    )
    args = parser.parse_args()
    
    if args.video:
        args.mode = "video"

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
    
    if args.headless:
        pcfg["save_results"] = False
        pcfg["display_video"] = False
        pcfg["record_video"] = False

    if args.mode == "video":
        pipeline = VideoPipeline(yolo_cfg, cnn_cfg, pcfg, logger=logger)
        results = pipeline.process_videos()
        if results is not None:
            total = 0
            print_section_header("Results")
            for video, cards in results.items():
                logger.info(f"{video}: {sorted(cards, key=lambda s: (s.split()[0], int(s.split()[1]))) if cards else 'No cards detected.'}")
                total += len(cards)
            logger.info(f"Total cards detected: {total}")

    else:
        pipeline = ScreenshotPipeline(yolo_cfg, cnn_cfg, pcfg, logger=logger)
        logger.info("Starting pipeline with batch classification...")
        results = pipeline.process_images()
        if results is not None:
            logger.info(results)


if __name__ == "__main__":
    main()
