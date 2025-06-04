from ultralytics import YOLO
import numpy as np

class YoloDetector:
    def __init__(self, model_path, conf_thresh=0.1, iou_thresh=0.3, device='cuda', debug=False):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.debug = debug
        
        # Move model onto GPU so its ready for use (saves time on the first frame detection later)
        _ = self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), conf=conf_thresh, verbose=False)
    
    def detect(self, image):
        results = self.model.predict(image, conf=self.conf_thresh, verbose=self.debug)
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
        return bboxes
