import pytest
import yaml
from pathlib import Path
from card_detector.pipeline.video_pipeline import VideoPipeline

# Load tests config at module scope
with open("configs/tests.yaml") as f:
    testcfg = yaml.safe_load(f)
    TEST_CASES = [
        (
            case["video"],
            case.get("fixture") or case["video"].replace('.mp4', '.yaml'),
            case.get("min_accuracy", testcfg.get("detection_accuracy_threshold", 0.90)),
        )
        for case in testcfg.get("tests", [])
    ]
    videos_dir = testcfg["videos_dir"]
    fixtures_dir = testcfg["fixtures_dir"]

def jaccard_index(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 1.0

@pytest.fixture(scope="session")
def pipeline_results():
    from card_detector.config import cfg
    # Set up config overrides for test runs
    pcfg = cfg["video_pipeline"]
    yolo_cfg = {
        "yolo_model": pcfg["yolo_model"],
        "yolo_conf_thresh": pcfg["yolo_conf_thresh"],
        "bbox_iou_thresh": pcfg["bbox_iou_thresh"],
    }
    cnn_cfg = {
        "classifiers": pcfg["classifiers"],
        "cnn_model_dir": pcfg["cnn_model_dir"],
        "cnn_conf_threshold": pcfg["cnn_conf_thresh"],
    }

    # Test-time overrides
    pcfg["video_dir"] = videos_dir
    pcfg["display_video"] = False
    pcfg["record_video"] = False
    pcfg["turbo"] = True

    pipeline = VideoPipeline(yolo_cfg, cnn_cfg, pcfg)
    return pipeline.process_videos(video_dir=videos_dir, logging=False)

@pytest.mark.parametrize(
    "video_file,fixture_file,min_accuracy",
    TEST_CASES
)
def test_pipeline_detections(video_file, fixture_file, min_accuracy, pipeline_results):
    gt_file = Path(fixtures_dir) / fixture_file
    with open(gt_file) as f:
        expected = set(yaml.safe_load(f)["expected"])

    preds = pipeline_results.get(video_file)
    assert preds is not None, f"No detections returned for {video_file}!"

    accuracy = jaccard_index(preds, expected)
    print(f"\nDetected: {len(preds)}, Expected: {len(expected)}, Jaccard: {accuracy:.3f}")

    if accuracy < min_accuracy:
        missing = expected - preds
        extra = preds - expected
        print(
            f"Accuracy {accuracy:.2f} below threshold {min_accuracy:.2f}\n"
            f"Missing items: {sorted(missing)}\n"
            f"Unexpected items: {sorted(extra)}"
        )
    
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.2f} below threshold {min_accuracy:.2f}"
