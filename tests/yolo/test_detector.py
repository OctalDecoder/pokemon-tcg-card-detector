from ultralytics import YOLO
import os

MODEL_NAME = 'cd27 - more off screen variance'

model = YOLO(f'../runs/detect/{MODEL_NAME}/weights/best.pt')  # path to weights

INPUT_DIR = 'tests/fixtures'
PROJECT = 'output/yolo/test_results' # TODO: Discover where this is placed now
NAME = 'card_detector_test'

os.makedirs(os.path.join(PROJECT, NAME), exist_ok=True)

# Run YOLOv8 inference: saves annotated images and optional text
model.predict(
    source=INPUT_DIR,
    imgsz=640,
    conf=0.1,
    save=True,           # save annotated images
    save_txt=False,      # skip saving raw text labels
    project=PROJECT,
    name=NAME
)

print(f"Inference complete. Check {os.path.join(PROJECT, NAME)} for results.")
