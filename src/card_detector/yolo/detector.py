from ultralytics import YOLO
import numpy as np
import time
class YoloDetector:
    def __init__(self, model_path, conf_thresh=0.1, iou_thresh=0.3, device='cuda', debug=False):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device  # Save the device string
        self.debug = debug
        self.det_time = 0.0
        self.crop_time = 0.0
        
        # Move model onto GPU at load time
        _ = self.model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            conf=conf_thresh,
            device=device,
            verbose=debug
        )
    
    def detect(self, image):
        det_start = time.time()
        results = self.model.predict(
            image,
            conf=self.conf_thresh,
            device=self.device,
            verbose=self.debug
        )
        self.det_time += time.time() - det_start
        
        crop_start = time.time()
        bboxes = []
        yolo_names = self.model.model.names
        for r in results:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ycls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), y in zip(xyxy, ycls):
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                cat = yolo_names[y]
                bboxes.append((x1, y1, x2, y2, cx, cy, cat))
        self.crop_time += time.time() - crop_start
        return bboxes
