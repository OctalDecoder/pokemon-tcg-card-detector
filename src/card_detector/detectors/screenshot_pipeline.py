# detectors/card_detection_pipeline.py
import os
import math
import glob
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache

from card_detector.yolo.detector import YoloDetector
from card_detector.cnn.classifier import CnnClassifier

def merge_overlapping_boxes(boxes, iou_thresh=0.3):
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

@lru_cache(maxsize=None)
def load_thumb(path: str):
    return Image.open(path).convert('RGB')

class ScreenshotPipeline:
    def __init__(self, yolo_cfg, cnn_cfg, pcfg, logger=None):
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
            subcats=cnn_cfg["cnn_subcats"],
            cnn_base_dir=cnn_cfg["cnn_base_dir"],
            conf_threshold=cnn_cfg.get("cnn_conf_threshold", 0.15),
            device=self.device
        )
        # Font
        self.font = None
        if pcfg.get("font_path") and os.path.exists(pcfg["font_path"]):
            self.font = ImageFont.truetype(pcfg["font_path"], 12)

    def process_images(self, screenshot_dir=None, save_results_images=None, logging=True):
        pcfg = self.pcfg
        screenshot_dir = screenshot_dir or pcfg["screenshot_dir"]
        save_results_images = save_results_images if save_results_images is not None else pcfg["save_result_images"]

        # Clean output dir
        if save_results_images:
            os.makedirs(pcfg["output_dir"], exist_ok=True)
            for f in glob.glob(os.path.join(pcfg["output_dir"], '*')):
                if f.lower().endswith(('.png', '.jpg')):
                    os.remove(f)
        # Prepare screenshots
        shots = sorted(sum([glob.glob(os.path.join(screenshot_dir, f'*.{ext}')) for ext in ('png', 'jpg')], []))
        detections = {}

        all0 = time.time()
        for sp in shots:
            bp = os.path.basename(sp)
            tt0 = time.time()
            orig = Image.open(sp).convert('RGB')
            arr = np.array(orig)

            # Detection
            bboxes = self.yolo.detect(arr)
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
            output, matches = self.cnn.classify(crops, cats)
            detections[bp] = output

            # Save results image
            if matches and save_results_images:
                n = len(matches)
                cols = min(pcfg["grid_cols"], n)
                rows = math.ceil(n / cols)
                first_thumb = next((m for m in matches if m), None)
                if first_thumb is None:
                    continue
                tw, th = load_thumb(first_thumb).size
                right = Image.new('RGB', (cols * tw, rows * th), (0, 0, 0))
                draw = ImageDraw.Draw(right)
                for i, m in enumerate(matches):
                    if m:
                        thumb = load_thumb(m).resize((tw, th), Image.LANCZOS)
                        r, c = divmod(i, cols)
                        right.paste(thumb, (c * tw, r * th))
                        if self.font:
                            draw.text((c * tw + 2, r * th + 2), os.path.basename(m), font=self.font, fill=(255, 255, 255))
                left_h = rows * th
                w0, h0 = orig.size
                left_w = int(w0 / h0 * left_h)
                left = orig.resize((left_w, left_h), Image.LANCZOS)
                combined = Image.new('RGB', (left_w + pcfg["middle_space"] + right.width, left_h), (0, 0, 0))
                combined.paste(left, (0, 0))
                combined.paste(right, (left_w + pcfg["middle_space"], 0))
                out_path = os.path.join(pcfg["output_dir"], bp)
                combined.save(out_path)
            if logging and self.logger:
                self.logger.info(f"{bp:.25s} processed")
        allt = time.time() - all0
        if logging and self.logger:
            instances = sum([len(cards) for cards in detections.values()])
            self.logger.info(f"Detected {instances} cards from {len(shots)} screenshots in {allt:.3f} seconds ({len(shots)/allt:.3f}FPS)")
        return detections
