"""Real-time video processing pipeline.

This module implements a simple frame-by-frame pipeline similar to
``screenshot_pipeline`` but designed for video input. Each frame is
processed by YOLO to obtain bounding boxes which are queued for CNN
classification. While the next frame is being prepared we attempt to
empty the queue so the system can keep up with the target FPS.
"""

from __future__ import annotations

import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Optional, Set, Tuple, List

import cv2
from PIL import Image

from card_detector.database.database import CardDB
from card_detector.cnn.classifier import CnnClassifier
from card_detector.yolo.detector import YoloDetector
from .screenshot_pipeline import merge_overlapping_boxes


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

        self.card_db = CardDB(pcfg["database"])

        self.queue: "Queue[Tuple[Image.Image, str]]" = Queue()
        self.seen_cards: Set[str] = set()

        self.target_fps = pcfg.get("target_fps", 30)
        self.batch_size = pcfg.get("cnn_batch_size", 8)

        self.display = pcfg.get("display", False)
        self.win_name = "Video" if self.display else None

        self.det_time = 0.0
        self.clf_time = 0.0

        self.stop_event = threading.Event()

    # ------------------------------------------------------------------
    def _classify_from_queue(self, budget: float = float("inf")) -> None:
        """Process queued crops in batches within ``budget`` seconds."""

        start = time.time()
        while (time.time() - start) < budget and not self.queue.empty():
            imgs: List[Image.Image] = []
            cats: List[str] = []
            for _ in range(min(self.batch_size, self.queue.qsize())):
                try:
                    crop, cat = self.queue.get_nowait()
                except Empty:
                    break
                imgs.append(crop)
                cats.append(cat)
            if not imgs:
                break
            t0 = time.time()
            labels = self.cnn.classify(imgs, cats)
            self.clf_time += time.time() - t0
            for card_id in labels:
                if card_id not in self.seen_cards:
                    self.seen_cards.add(card_id)
                    name = self.card_db.get_name_by_seriesid_id(*card_id.split(" "))
                    print(f"New card {card_id} {name} detected")

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        """Background thread for continuously processing the queue."""

        while not self.stop_event.is_set() or not self.queue.empty():
            if self.queue.empty():
                time.sleep(0.01)
                continue
            self._classify_from_queue()

    # ------------------------------------------------------------------
    def process_videos(
        self, video_dir: Optional[str] = None, logging: bool = True
    ) -> Set[str]:
        """Process all ``.mp4`` files in ``video_dir`` and return detected cards."""

        video_dir = video_dir or self.pcfg["video_dir"]
        videos = sorted(Path(video_dir).glob("*.mp4"))

        all_start = time.time()
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()
        for vp in videos:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                print(f"Failed to open video {vp}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
            interval = max(1, round(fps / self.target_fps))
            frame_idx = 0

            if self.logger and logging:
                self.logger.info(f"Processing video {vp.name} at {fps:.2f} FPS")

            while True:
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                bboxes = self.yolo.detect(frame)
                bboxes = merge_overlapping_boxes(
                    bboxes, iou_thresh=self.pcfg["bbox_iou_thresh"]
                )

                for x1, y1, x2, y2, *_rest, cat in bboxes:
                    if self.display:
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
                    if frame_idx % interval == 0:
                        crop = Image.fromarray(frame[int(y1) : int(y2), int(x1) : int(x2)])
                        self.queue.put((crop, cat))

                det_elapsed = time.time() - frame_start
                self.det_time += det_elapsed

                frame_idx += 1

                if self.display:
                    cv2.imshow(self.win_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                
            cap.release()
            if self.display:
                cv2.destroyWindow(self.win_name)

        # Flush remaining queue
        self.stop_event.set()
        worker.join()

        total_time = time.time() - all_start

        if self.display:
            cv2.destroyAllWindows()

        if self.logger and logging:
            self.logger.info(
                f"Processed {len(videos)} videos in {total_time:.2f}s | "
                f"det={self.det_time:.2f}s clf={self.clf_time:.2f}s"
            )

        return self.seen_cards
