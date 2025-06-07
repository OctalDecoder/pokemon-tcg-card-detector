"""
video_pipeline.py

Real-time video processing pipeline for card detection.

Classes:
    VideoPipeline:
        - __init__(yolo_cfg, cnn_cfg, pcfg, logger=None):
            Initialize YOLO detector, CNN classifier, shared CardDB, and threading resources.
        - _worker_loop():
            Internal method to start the ClassifierWorker loop in a separate thread.
        - _draw_fps(frame):
            Compute and overlay FPS estimate on the given frame.
        - _draw_live_detections(frame):
            Overlay recent card detections on the given frame.
        - process_videos(video_dir=None, logging=True) -> Set[str]:
            Process all `.mp4` files in `video_dir`. For each video:
                - Read frames, run YOLO to get bounding boxes.
                - Every Nth frame (based on target FPS), extract crops and enqueue for CNN classification.
                - Overlay FPS and live detections if `display` or `save_results` is enabled.
                - Optionally save annotated video to `output_dir`.
            Returns a set of detected card IDs across all videos.

Usage Example:
    from card_detector.pipeline.video_pipeline import VideoPipeline
    # yolo_cfg, cnn_cfg, pcfg = configs()
    # my_logger = MyLogger()
    pipeline = VideoPipeline(yolo_cfg, cnn_cfg, pcfg, logger=my_logger)
    detected_cards = pipeline.process_videos()
"""

from __future__ import annotations

import time
import threading
import shutil
from queue import Queue
from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict

import cv2
from PIL import Image

from card_detector.database.database import CardDB
from card_detector.cnn.classifier import CnnClassifier
from card_detector.yolo.detector import YoloDetector
from .screenshot_pipeline import merge_overlapping_boxes
from card_detector.ui.video_overlay import draw_fps_overlay, draw_live_detections_overlay
from card_detector.pipeline.video_classification import ClassifierWorker


class VideoPipeline:
    """Pipeline for real-time card detection from video files."""

    def __init__(
        self, yolo_cfg: dict, cnn_cfg: dict, pcfg: dict, logger: Optional[object] = None
    ) -> None:
        import torch

        self.logger = logger
        self.pcfg = pcfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.yolo = YoloDetector(
            model_path=yolo_cfg["yolo_model"],
            conf_thresh=yolo_cfg["yolo_conf_thresh"],
            iou_thresh=yolo_cfg["bbox_iou_thresh"],
            debug=pcfg.get("debug", False),
            device=self.device,
        )

        self.cnn = CnnClassifier(
            subcats=cnn_cfg["classifiers"],
            cnn_model_dir=cnn_cfg["cnn_model_dir"],
            conf_threshold=cnn_cfg.get("cnn_conf_threshold", 0.15),
            device=self.device,
        )

        # Share DB connection across threads for name lookups
        self.card_db = CardDB(pcfg["database"], check_same_thread=False)

        self.queue: "Queue[Tuple[Image.Image, str]]" = Queue()
        self.seen_cards: Set[str] = set()

        # Detection FPS = YOLO+CNN process rate, from config
        self.detection_fps = pcfg.get("target_fps", 30)
        self.batch_size = pcfg.get("cnn_batch_size", 8)

        self.display_video = pcfg.get("display_video", False)
        self.win_name = "Video" if self.display_video else None

        self.record_video = pcfg.get("record_video", False)
        self.output_dir = Path(pcfg.get("output_dir", "output")) / "video_pipeline"
        if self.record_video:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_fps = pcfg["show_fps"]
        self.show_classifications = pcfg["show_classifications"]
        self.show_bboxes = pcfg["show_bboxes"]
        
        self.turbo = pcfg["turbo"]
        self.detection_frame_skip = pcfg["detection_skip"]
        self.classification_frame_skip = pcfg["classification_skip"]

        self.frame_idx = 0
        self.det_time = 0.0
        self.display_time = 0.0

        self.stop_event = threading.Event()

        # For FPS counter overlay
        self._fps_times: List[float] = []
        self._fps_window: int = 30  # Number of frames to average for FPS display

        # For detected card overlay: {card_id: (display_string, detected_time)}
        self._live_detections: Dict[str, Tuple[str, float]] = {}
        self._detection_display_secs: float = 2.0

        # Lock for live detection overlay, since detected in worker thread
        self._overlay_lock = threading.Lock()

        # Refactored worker for classification
        self.worker = ClassifierWorker(
            queue=self.queue,
            cnn=self.cnn,
            card_db=self.card_db,
            seen_cards=self.seen_cards,
            overlay_lock=self._overlay_lock,
            live_detections=self._live_detections,
            batch_size=self.batch_size,
            stop_event=self.stop_event,
        )

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        self.worker.loop()

    # ------------------------------------------------------------------
    def _draw_fps(self, frame):
        now = time.time()
        self._fps_times.append(now)
        if len(self._fps_times) > self._fps_window:
            self._fps_times.pop(0)
        if len(self._fps_times) >= 2:
            fps_est = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])
            draw_fps_overlay(frame, fps_est)

    def _draw_live_detections(self, frame):
        now = time.time()
        # Copy and clean-up old detections
        with self._overlay_lock:
            expired = [cid for cid, (_, t) in self._live_detections.items() if (now - t) > self._detection_display_secs]
            for cid in expired:
                del self._live_detections[cid]
            live_items = list(self._live_detections.values())
        draw_live_detections_overlay(frame, live_items)

    # ------------------------------------------------------------------
    def process_videos(
        self, video_dir: Optional[str] = None, logging: bool = True
    ) -> Set[str]:
        """Process all `.mp4` files in `video_dir` and return detected cards."""

        video_dir = video_dir or self.pcfg["video_dir"]
        videos = sorted(Path(video_dir).glob("*.mp4"))

        all_start = time.time()
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()
        for vp in videos:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                self.logger.info(f"Failed to open video {vp}")
                continue

            # --- FPS split: display (playback/saving) and detection (processing) ---
            display_fps = cap.get(cv2.CAP_PROP_FPS) or 30  # For display and saving video, match original FPS or fallback to 30
            detection_fps = self.detection_frame_skip
            self.frame_idx = 0

            if self.logger and logging:
                self.logger.info(f"Processing video {vp.name} at {display_fps:.2f} display FPS, {detection_fps} detection FPS")

            # Set up video writer if saving results
            if self.record_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = self.output_dir / (vp.stem + "_detected.mp4")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_writer = cv2.VideoWriter(str(out_path), fourcc, display_fps, (width, height))
            else:
                video_writer = None

            self.seen_cards.clear()
            self.frame_idx = 0
            self.det_frame_idx = 0
            
            while True:
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                
                if self.turbo and self.frame_idx % self.detection_frame_skip > 0:
                    self.frame_idx += 1
                    continue

                # Yolo Detection
                if self.frame_idx % self.detection_frame_skip == 0:
                    bboxes = self.yolo.detect(frame)
                    bboxes = merge_overlapping_boxes(
                        bboxes, iou_thresh=self.pcfg["bbox_iou_thresh"]
                    )

                    for x1, y1, x2, y2, *_rest, cat in bboxes:
                        if self.show_bboxes and self.display_video:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                str(cat),
                                (int(x1), max(0, int(y1) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )
                        # Queue crops for CNN
                        if self.det_frame_idx % self.detection_frame_skip == 0:
                            x1_, y1_, x2_, y2_ = map(int, (x1, y1, x2, y2))
                            bgr_crop = frame[y1_:y2_, x1_:x2_]
                            if bgr_crop.size != 0:  # ensure non-empty crop
                                rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
                                pil_crop = Image.fromarray(rgb_crop)
                                self.queue.put((pil_crop, cat))

                    self.det_time += time.time() - frame_start
                    self.det_frame_idx += 1
                
                dispt = time.time()
                # --- FPS & Detected Card Overlays ---
                if self.display_video or self.record_video:
                    if self.show_fps:
                        self._draw_fps(frame)
                    if self.show_classifications:
                        self._draw_live_detections(frame)


                self.frame_idx += 1

                if video_writer is not None:
                    video_writer.write(frame)

                if self.display_video:
                    cv2.imshow(self.win_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                self.display_time = time.time() - dispt

            cap.release()
            if video_writer is not None:
                video_writer.release()
            if self.display_video:
                cv2.destroyWindow(self.win_name)

            # Print per-video seen cards and clear
            if self.seen_cards:
                self.logger.info(f"\n==> Cards detected in '{vp.name}':")
                self.logger.info(sorted(self.seen_cards, key=lambda x: (x.split()[0], int(x.split()[-1]))))
                self.logger.info(f"Cards detected: {len(self.seen_cards)}")
            else:
                self.logger.info(f"\n==> No cards detected in '{vp.name}'.")
            self.seen_cards.clear()

        # Flush remaining queue
        self.stop_event.set()
        worker.join()

        total_time = time.time() - all_start

        if self.display_video:
            cv2.destroyAllWindows()

        if self.logger and logging:
            self.logger.info(
                f"Processed {len(videos)} videos in {total_time:.2f}s | "
                f"det={self.det_time:.2f}s clf={self.worker.clf_time:.2f}s dispt={self.display_time:.2f}s"
            )
