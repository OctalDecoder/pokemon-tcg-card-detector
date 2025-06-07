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
"""

from __future__ import annotations

import time
import threading
import shutil
from queue import Queue, Full
from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict

import cv2
from PIL import Image

from card_detector.database.database import CardDB
from card_detector.cnn.classifier import CnnClassifier
from card_detector.yolo.detector import YoloDetector
from card_detector.ui.video_overlay import draw_fps_overlay, draw_live_detections_overlay, draw_bounding_box
from .screenshot_pipeline import merge_overlapping_boxes
from .video_classification import ClassifierWorker

# -------------------------
# Module-Level Defaults/Constants
# -------------------------
DEFAULT_CODEC = "mp4v"
DEFAULT_OUTPUT_SUBDIR = "video_pipeline"
DEFAULT_DISPLAY_FPS = 30
DEFAULT_QUEUE_MAXSIZE = 128
DEFAULT_VIDEO_EXT = "*.mp4"
DEFAULT_FPS_WINDOW = 30
DEFAULT_DETECTION_DISPLAY_SECS = 2.0

class VideoPipeline:
    """Pipeline for real-time card detection from video files."""

    def __init__(
        self, 
        yolo_cfg: dict, 
        cnn_cfg: dict, 
        pcfg: dict, 
        logger: Optional['logging.Logger'] = None
    ) -> None:
        import torch

        self.logger = logger
        self.pcfg = pcfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Config variables (read from pcfg or fall back to module-level defaults)
        self.display_fps = pcfg.get("display_fps", None)
        self.recording_fps = pcfg.get("recording_fps", None)
        self.native_fps = None  # Determined per video
        self.detection_fps = 0
        self.detection_frame_skip = pcfg.get("detection_skip", 4)
        self.classification_frame_skip = pcfg.get("classification_skip", 0)
        self.turbo = pcfg.get("turbo", False)
        self.batch_size = pcfg.get("cnn_batch_size", 8)
        self.queue_maxsize = pcfg.get("detection_queue_maxsize", DEFAULT_QUEUE_MAXSIZE)

        # Rendering/display config
        self.show_fps = pcfg.get("show_fps", True)
        self.show_classifications = pcfg.get("show_classifications", True)
        self.show_bboxes = pcfg.get("show_bboxes", True)
        self.display_video = pcfg.get("display_video", False)
        self.record_video = pcfg.get("record_video", False)
        self.win_name = "Video" if self.display_video else None
        self.output_dir = Path(pcfg.get("output_dir", "output")) / DEFAULT_OUTPUT_SUBDIR
        if self.record_video:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Models & DB
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
        self.card_db = CardDB(pcfg["database"], check_same_thread=False)

        # State & Queues
        self.detection_queue: "Queue[Tuple[Image.Image, str]]" = Queue(maxsize=self.queue_maxsize)
        self.seen_cards: Set[str] = set()
        self.all_seen_cards: Set[str] = set()
        self.frame_idx = 0
        self.det_time = 0.0
        self.render_time = 0.0
        self.stop_event = threading.Event()

        # Overlay/Display state
        self._fps_times: List[float] = []
        self._fps_window: int = DEFAULT_FPS_WINDOW
        self._live_detections: Dict[str, Tuple[str, float]] = {}
        self._detection_display_secs: float = DEFAULT_DETECTION_DISPLAY_SECS
        self._overlay_lock = threading.Lock()

        # Classifier Worker
        self.classifier_worker = ClassifierWorker(
            queue=self.detection_queue,
            cnn=self.cnn,
            card_db=self.card_db,
            seen_cards=self.seen_cards,
            all_seen_cards=self.all_seen_cards,
            overlay_lock=self._overlay_lock,
            live_detections=self._live_detections,
            batch_size=self.batch_size,
            stop_event=self.stop_event,
        )

    def _worker_loop(self) -> None:
        self.classifier_worker.loop()

    def _draw_fps(self, frame):
        if self.show_fps: 
            now = time.time()
            self._fps_times.append(now)
            if len(self._fps_times) > self._fps_window:
                self._fps_times.pop(0)
            if len(self._fps_times) >= 2:
                fps_est = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])
                draw_fps_overlay(frame, fps_est)

    def _draw_live_detections(self, frame):
        if self.show_classifications: 
            now = time.time()
            with self._overlay_lock:
                expired = [cid for cid, (_, t) in self._live_detections.items() if (now - t) > self._detection_display_secs]
                for cid in expired:
                    del self._live_detections[cid]
                live_items = list(self._live_detections.values())
            draw_live_detections_overlay(frame, live_items)

    def process_videos(
        self, 
        video_dir: Optional[str] = None, 
        logging: bool = True
    ) -> Set[str]:
        """Process all `.mp4` files in `video_dir` and return detected cards."""

        video_dir = video_dir or self.pcfg["video_dir"]
        videos = sorted(Path(video_dir).glob(DEFAULT_VIDEO_EXT))

        all_start = time.time()
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()

        try:
            for vp in videos:
                # Try-except per video for robust resource cleanup/logging
                try:
                    cap = cv2.VideoCapture(str(vp))
                    if not cap.isOpened():
                        if self.logger:
                            self.logger.info(f"Failed to open video '{vp}'.")
                        continue

                    self.native_fps = cap.get(cv2.CAP_PROP_FPS)
                    self.display_fps = self.display_fps or self.native_fps or DEFAULT_DISPLAY_FPS
                    self.recording_fps = self.recording_fps or self.native_fps or DEFAULT_DISPLAY_FPS
                    self.detection_fps = max(1, self.display_fps // (self.detection_frame_skip + 1))
                    
                    if self.logger and logging:
                        self.logger.info(
                            f"Processing video '{vp.name}' @ {self.native_fps if self.native_fps else 'DEFAULTED_TO_30_'} FPS | "
                            f"display_fps={self.display_fps:.2f} | recording_fps={self.recording_fps:.2f} | detection_fps={self.detection_fps}"
                        )

                    # Video Writer
                    if self.record_video:
                        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_CODEC)
                        out_path = self.output_dir / (vp.stem + "_detected.mp4")
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        video_writer = cv2.VideoWriter(str(out_path), fourcc, self.recording_fps, (width, height))
                    else:
                        video_writer = None

                    # State reset
                    self.seen_cards.clear()
                    self.frame_idx = 0

                    while not self.stop_event.is_set():
                        frame_start = time.time()
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if self.turbo and self.frame_idx % self.detection_frame_skip > 0:
                            self.frame_idx += 1
                            continue

                        if self.frame_idx % self.detection_frame_skip == 0:
                            bboxes = self.yolo.detect(frame)
                            bboxes = merge_overlapping_boxes(
                                bboxes, iou_thresh=self.pcfg["bbox_iou_thresh"]
                            )
                            for x1, y1, x2, y2, *_rest, cat in bboxes:
                                if self.show_bboxes and self.display_video:
                                    draw_bounding_box(frame, (x1, y1, x2, y2), str(cat))
                                # Enqueue for CNN with queue-size protection
                                x1_, y1_, x2_, y2_ = map(int, (x1, y1, x2, y2))
                                bgr_crop = frame[y1_:y2_, x1_:x2_]
                                if bgr_crop.size != 0:
                                    rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
                                    pil_crop = Image.fromarray(rgb_crop)
                                    try:
                                        self.detection_queue.put_nowait((pil_crop, cat))
                                    except Full:
                                        if self.logger:
                                            self.logger.warning(
                                                f"Detection queue full (max {self.queue_maxsize}), skipping a crop at frame {self.frame_idx}."
                                            )
                            self.det_time += time.time() - frame_start

                        render_time = time.time()
                        if self.display_video or self.record_video:
                            self._draw_fps(frame)
                            self._draw_live_detections(frame)
                        if video_writer is not None:
                            video_writer.write(frame)
                        if self.display_video:
                            cv2.imshow(self.win_name, frame)
                            # Graceful early exit by user
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                if self.logger:
                                    self.logger.info("Exit requested by user (q). Stopping pipeline.")
                                self.stop_event.set()
                                break
                        self.render_time = time.time() - render_time

                        if not self.turbo and self.display_fps:
                            elapsed = time.time() - frame_start
                            frame_duration = 1.0 / self.display_fps
                            to_sleep = frame_duration - elapsed
                            if to_sleep > 0:
                                time.sleep(to_sleep)

                        if self.logger and self.pcfg.get("debug", False):
                            self.logger.debug(
                                f"Frame {self.frame_idx}: "
                                f"Queue size={self.detection_queue.qsize()} | "
                                f"det_time={self.det_time:.3f}s | render_time={self.render_time:.3f}s"
                            )
                        self.frame_idx += 1

                    cap.release()
                    if video_writer is not None:
                        video_writer.release()
                    if self.display_video:
                        cv2.destroyWindow(self.win_name)

                    # Per-video detected cards
                    if self.seen_cards:
                        if self.logger:
                            self.logger.info(
                                f"\n==> Cards detected in '{vp.name}':\n"
                                f"{sorted(self.seen_cards, key=lambda x: (x.split()[0], int(x.split()[-1])))}\n"
                                f"Cards detected: {len(self.seen_cards)}"
                            )
                    else:
                        if self.logger:
                            self.logger.info(f"\n==> No cards detected in '{vp.name}'.")

                    self.seen_cards.clear()

                except Exception as e:
                    if self.logger:
                        self.logger.exception(f"Exception while processing '{vp}': {e}")
                finally:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    if self.display_video:
                        try:
                            cv2.destroyWindow(self.win_name)
                        except Exception:
                            pass

            # End for videos
            self.stop_event.set()
            worker.join(timeout=5)
        except KeyboardInterrupt:
            self.stop_event.set()
            if self.logger:
                self.logger.info("KeyboardInterrupt received. Exiting pipeline gracefully.")
            worker.join(timeout=5)
        finally:
            if self.display_video:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        total_time = time.time() - all_start
        if self.logger and logging:
            self.logger.info(
                f"Processed {len(videos)} videos in {total_time:.2f}s | "
                f"det={self.det_time:.2f}s | clf={self.classifier_worker.clf_time:.2f}s | dispt={self.render_time:.2f}s"
            )
        return self.all_seen_cards
