from ultralytics import YOLO
import os

MODEL_NAME = 'cd23'

# 1. Load your trained model
model = YOLO(f'runs/detect/{MODEL_NAME}/weights/best.pt')  # path to best weights

# 2. Directory of real-world screenshots to test
INPUT_DIR = 'images/screenshots/test suite 1'
# The project and name will determine where YOLO saves the annotated outputs
PROJECT = 'runs/inference'
NAME = 'card_detector_test'

# Ensure project directory exists
os.makedirs(os.path.join(PROJECT, NAME), exist_ok=True)

# 3. Run YOLOv8 inference: saves annotated images and optional text
model.predict(
    source=INPUT_DIR,
    imgsz=640,
    conf=0.90,
    save=True,           # save annotated images
    save_txt=False,      # skip saving raw text labels (since you have ground truth)
    project=PROJECT,
    name=NAME
)

print(f"Inference complete. Check {os.path.join(PROJECT, NAME)} for results.")