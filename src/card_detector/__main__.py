""" src/card_detector/__main__.py

Entry point for the application.
Handles:
  - Detection
  - Training
  - Data Generation (TODO)
"""

import argparse
import logging

def main():
    from card_detector.config import cfg

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(prog="card-detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- DETECT subcommand ---
    detect_parser = subparsers.add_parser("detect", aliases=["d", "D"], help="Run detection pipeline")
    detect_parser.add_argument("--mode", choices=["screenshot", "video"], default="screenshot", help="Detection mode")
    detect_parser.add_argument("-v", "--video", action="store_true", help="Shortcut for --mode video")
    detect_parser.add_argument("-t", "--turbo", action="store_true", help="Turbo mode for video")
    detect_parser.add_argument("-hl", "--headless", action="store_true", help="No display or saving")

    # --- TRAIN subcommand ---
    train_parser = subparsers.add_parser("train", aliases=["t", "T"], help="Train a model")
    train_subparsers = train_parser.add_subparsers(dest="model", required=True)

    # -- YOLO training args
    yolo_parser = train_subparsers.add_parser("yolo", help="Train YOLOv8 model")
    yolo_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    yolo_parser.add_argument("--batch-size", type=int, help="Batch size")
    yolo_parser.add_argument("--img-size", type=int, help="Image size (YOLO only)")
    yolo_parser.add_argument("--device", type=str, help="CUDA device or 'cpu'")
    yolo_parser.add_argument("--model-name", type=str, help="Model save/run name")
    yolo_parser.add_argument("--lr0", type=float, help="Initial learning rate")
    yolo_parser.add_argument("--lr-final-factor", type=float, help="Final LR factor")
    yolo_parser.add_argument("--patience", type=int, help="Early stopping patience")
    yolo_parser.add_argument("--data-config", type=str, help="Path to data YAML config")
    yolo_parser.add_argument("--yolo-model", type=str, help="Path to YOLO model YAML or weights")

    # -- CNN training args
    cnn_parser = train_subparsers.add_parser("cnn", help="Train CNN with distillation")
    cnn_parser.add_argument("--epochs-master", type=int, help="Epochs for master CNN", default=10)
    cnn_parser.add_argument("--epochs-student", type=int, help="Epochs for student CNN(s)", default=5)
    cnn_parser.add_argument("--student-only", action="store_true", help="Only train students (skip master)", default=False)
    cnn_parser.add_argument("--resume-master", action="store_true", help="Resume master CNN from checkpoint", default=False)

    args = parser.parse_args()

    # ---- Command dispatch ----
    if args.command in ("detect", "d", "D"):
        # ---- handle detection pipeline ----
        from card_detector.pipeline.screenshot_pipeline import ScreenshotPipeline
        from card_detector.pipeline.video_pipeline import VideoPipeline
        from card_detector.util.logging import print_section_header

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
        if args.turbo:
            pcfg["turbo"] = True

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

    elif args.command in ("train", "t", "T"):
        if args.model == "yolo":
            from card_detector.yolo.train import train as train_yolo
            train_yolo(args, logger)
        elif args.model == "cnn":
            from card_detector.cnn.train import train_cnn
            train_cnn(args, logger)
        else:
            logger.error("Unknown model for training.")


if __name__ == "__main__":
    main()
