# src/card_detector/__main__.py
import argparse
# from .yolo.detector import CardDetector
# from .cnn.classifier import CardClassifier
# from .video.processor import annotate_video
from .proto.prototype import detect

def main():
    p = argparse.ArgumentParser(description="Card Detector Pipeline")
    # p.add_argument("--input",  "-i", required=True, help="Path to input video/image")
    # p.add_argument("--output", "-o", required=True, help="Path for output")
    args = p.parse_args()

    # if args.input ends with .mp4 → video path
    detect()

if __name__ == "__main__":
    main()
