"""Screenshot processing pipeline.

This module implements a simple screenshot card detection pipeline. Each screenshot is
processed by YOLO to obtain bounding boxes. These bounding boxes are used to crop cards
and pass into the CNN classifier. The CNN classifier then determines the card ID and uses
it to gather information about the card from the sqllite database.
"""

import os
import math
import time
from typing import List, Tuple, Dict, Optional
import numpy as np
from functools import lru_cache
from pathlib import Path
from PIL import Image

from card_detector.database.database import CardDB
from card_detector.cnn.classifier import CnnClassifier
from card_detector.yolo.detector import YoloDetector


def merge_overlapping_boxes(
    boxes: List[Tuple[float, float, float, float, float, float, int]],
    iou_thresh: float = 0.3
) -> List[Tuple[float, float, float, float, float, float, int]]:
    """Merge boxes that overlap above an IoU threshold and belong to the same category."""
    def iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    merged = []
    used = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        group = [b]
        used[i] = True
        for j, b2 in enumerate(boxes[i + 1:], start=i + 1):
            if used[j] or b2[6] != b[6]:
                continue
            if iou(b, b2) > iou_thresh:
                group.append(b2)
                used[j] = True
        xs1 = [g[0] for g in group]
        ys1 = [g[1] for g in group]
        xs2 = [g[2] for g in group]
        ys2 = [g[3] for g in group]
        nx1, ny1 = min(xs1), min(ys1)
        nx2, ny2 = max(xs2), max(ys2)
        ncat = b[6]
        ncx = (nx1 + nx2) / 2
        ncy = (ny1 + ny2) / 2
        merged.append((nx1, ny1, nx2, ny2, ncx, ncy, ncat))
    return merged


class ScreenshotPipeline:
    """Pipeline for detecting, classifying, and visualizing card detections in screenshots."""

    THUMB_SIZE: Tuple[int, int] = (184, 256)  # Default thumbnail size (w, h)
    OUTPUT_SUBDIR: str = "screenshot_pipeline"
    FALLBACK_COLOR: Tuple[int, int, int] = (80, 80, 80)  # Fallback color for missing images

    def __init__(self, yolo_cfg: dict, cnn_cfg: dict, pcfg: dict, logger: Optional[object] = None):
        import torch
        self.pcfg = pcfg
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YoloDetector(
            model_path=yolo_cfg["yolo_model"],
            conf_thresh=yolo_cfg["yolo_conf_thresh"],
            iou_thresh=yolo_cfg["bbox_iou_thresh"],
            debug=pcfg.get("debug", False),
            device=self.device
        )
        self.cnn = CnnClassifier(
            subcats=cnn_cfg["classifiers"],
            cnn_model_dir=cnn_cfg["cnn_model_dir"],
            conf_threshold=cnn_cfg.get("cnn_conf_threshold", 0.15),
            device=self.device
        )
        # Allow cross-thread access as classification may run in a worker
        self.card_db = CardDB(pcfg["database"], check_same_thread=False)

    @lru_cache(maxsize=256)
    def get_db_thumb(self, series_id: str, card_id: str) -> Image.Image:
        """Fetches and resizes card image from the DB. Returns fallback if missing."""
        img = self.card_db.get_image_blob_by_seriesid_id(series_id, card_id)
        if img:
            return img.convert('RGB').resize(self.THUMB_SIZE, Image.LANCZOS)
        else:
            print(f"Failed to fetch image for: {series_id} {card_id}")
            return Image.new('RGB', self.THUMB_SIZE, self.FALLBACK_COLOR)

    def process_images(
        self,
        screenshot_dir: Optional[str] = None,
        save_results_images: Optional[bool] = None,
        logging: bool = True
    ) -> Dict[str, List[str]]:
        """
        Process screenshots: detects cards, classifies them, and (optionally) saves result images.

        Returns:
            Dictionary mapping screenshot filenames to list of detected card IDs (as strings).
        """
        pcfg = self.pcfg
        screenshot_dir = screenshot_dir or pcfg["screenshot_dir"]
        save_results_images = save_results_images if save_results_images is not None else pcfg["save_results"]
        output_dir = os.path.join(pcfg["output_dir"], self.OUTPUT_SUBDIR)

        # Clean output dir
        if save_results_images:
            os.makedirs(output_dir, exist_ok=True)
            for f in Path(output_dir).glob("*"):
                if f.suffix.lower() in (".png", ".jpg"):
                    try:
                        f.unlink()
                    except Exception as e:
                        print(f"Could not delete file {f}: {e}")

        # Prepare screenshots
        shots = sorted(Path(screenshot_dir).glob("*.png")) + sorted(Path(screenshot_dir).glob("*.jpg"))
        detections: Dict[str, List[str]] = {}

        all0 = time.time()
        for sp in shots:
            bp = sp.name
            tt0 = time.time()
            try:
                with Image.open(sp) as orig_img:
                    orig = orig_img.convert('RGB')
            except Exception as e:
                print(f"Failed to open image {sp}: {e}")
                continue

            arr = np.array(orig)

            # Detection
            try:
                bboxes = self.yolo.detect(arr)
            except Exception as e:
                print(f"Detection failed for {bp}: {e}")
                continue

            if not bboxes:
                continue

            bboxes = merge_overlapping_boxes(bboxes, iou_thresh=pcfg["bbox_iou_thresh"])
            heights = [y2 - y1 for x1, y1, x2, y2, *rest in bboxes]
            row_h = np.median(heights)
            sorted_boxes = sorted(
                bboxes,
                key=lambda b: (int(b[1] // (row_h * 0.8)), b[4])
            )
            crops = [orig.crop(tuple(map(int, b[:4]))) for b in sorted_boxes]
            cats = [b[6] for b in sorted_boxes]

            try:
                output = self.cnn.classify(crops, cats)
            except Exception as e:
                print(f"CNN classification failed for {bp}: {e}")
                continue

            detections[bp] = [f"{s} - {self.card_db.get_name_by_seriesid_id(*s.split(' '))}" for s in output]

            # Save results image
            if output and save_results_images:
                n = len(output)
                cols = min(pcfg["grid_cols"], n)
                rows = math.ceil(n / cols)
                first_thumb = next((m for m in output if m), None)
                if first_thumb is None:
                    continue

                tw, th = self.THUMB_SIZE
                right = Image.new('RGB', (cols * tw, rows * th), (0, 0, 0))
                for i, m in enumerate(output):
                    if m:
                        thumb = self.get_db_thumb(*m.split(" "))
                        r, c = divmod(i, cols)
                        right.paste(thumb, (c * tw, r * th))

                left_h = rows * th
                w0, h0 = orig.size
                left_w = int(w0 / h0 * left_h)
                left = orig.resize((left_w, left_h), Image.LANCZOS)
                combined = Image.new('RGB', (left_w + pcfg["middle_space"] + right.width, left_h), (0, 0, 0))
                combined.paste(left, (0, 0))
                combined.paste(right, (left_w + pcfg["middle_space"], 0))
                out_path = os.path.join(output_dir, bp)
                try:
                    combined.save(out_path)
                except Exception as e:
                    print(f"Failed to save combined image {out_path}: {e}")

            if logging and self.logger:
                self.logger.info(f"{bp:.25s} processed")
        allt = time.time() - all0
        if logging and self.logger:
            instances = sum([len(cards) for cards in detections.values()])
            self.logger.info(f"Detected {instances} cards from {len(shots)} screenshots in {allt:.3f} seconds ({len(shots)/allt:.3f}FPS)")
        return detections
