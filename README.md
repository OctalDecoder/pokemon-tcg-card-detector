# 🃏 Pokemon TCG Pocket Card Detection

---

## 📋 Table of Contents

- [🛠️ General TODO](#️-general-todo)
- [🚀 Project Summary](#-project-summary)
- [📸 Overview & Examples](#-overview--examples)
- [📁 Repository Setup Guide (WIP)](#-repository-setup-guide-wip)
- [🧪 Real Time Processing Research Suggestions](#-real-time-processing-research-suggestions)

---

## 🛠️ General TODO

- [x] Move global variables into `config.yaml`
- [x] Unify mappings for CNN into a single file
- [x] Move all images into a database with all the webscraped data
- [x] Remove URLs from CNN mappings
- [ ] Create DB table for each subcategory
- [x] Update database types, currently all `string`
- [x] ~~Combine yolo.yaml into config.yaml~~ (more maintainable as is)
- [x] Refactor `prototype.py` into separate classes:

  - [x] **YOLO** logic
  - [x] **CNN** logic

- [x] ~~Complete `Makefile` (build, test, CLI shortcuts)~~ (now in `pyproject.toml`)
- [ ] Write and run tests for:

  - [ ] `cnn/` modules
  - [ ] `yolo/` modules
  - [ ] `screenshot_pipeline.py`
  - [ ] `video_pipeline.py`
  - [ ] ~~`train/` scripts~~
  - [ ] ~~`generate_data/` scripts~~

- [ ] Get all tests passing (fix thresholds, edge cases, etc.)
- [x] ~~Create requirements.txt~~ (now in `pyproject.toml`)
- [x] ~~Verify setup.py is production-ready~~ (now in `pyproject.toml`)
- [x] Implement detection classes:

  - [x] **ImageDetection** (single images)
  - [x] **VideoDetection** (frame-by-frame)

- [ ] Extend video detection to handle images in a folder as the input stream
- [ ] Evaluate moving to mobile-capable student models (TinyML-friendly)
- [ ] Investigate using the model on cards it’s never seen or trained on (no CNN classification)
- [ ] Prune and quantise the student CNNs
- [ ] Add `pHash` duplicate filtering to reduce processing time
- [ ] Test integration of new cards from the latest expansion pack
- [ ] Add CI/CD workflows
- [ ] Finalize README so it's a true "README" (not just a to-do list)
- [ ] SLEEP 😴

---

## 🚀 Project Summary

This repository implements a **high-performance Pokemon TCG Pocket card detection and classification pipeline** for real-time screenshot and video input.
The workflow is fully asynchronous and optimized for speed, accuracy, and scalability.

> _Future plans to expand to the full **Pokemon TCG**._

---

## 📸 Overview & Examples

### 1. Screenshot/Video Ingestion

- Screenshots are pulled from `screenshot_dir` and videos from `videos_dir` (configurable in `config.yaml`).
- Asynchronous pipeline ensures frame acquisition never blocks processing.

_Example: Pack Opening Screenshot_ <img src="assets/2.jpg" width="300">

_Example: Pack Opening Video Demo_ <img src="assets/card_opening.gif" width="300">

---

### 2. YOLOv8 Detection & Tracking

- YOLOv8 identifies cards in each frame, outputting bounding boxes, scores, and class labels (“fullart” vs. “standard”).
- Integrated ByteTrack/SORT assigns persistent `track_id`s for robust tracking. _(TODO)_

_Example: Pack Summary Screenshot_ <img src="assets/3.png" width="800">

_Example: Dex Scrolling Video Demo_ <img src="assets/dex_scrolling.gif" width="220">

---

### 3. Fast Duplicate Filtering (pHash Cache)

- Each crop is hashed with a 64-bit perceptual hash (`pHash`). _(TODO)_
- An LRU cache maps each hash to its last label/timestamp. _(TODO)_
- If a crop matches a cached hash recently, CNN inference is skipped. _(TODO)_

---

### 4. Pruned, Quantized & Distilled CNN Classification

- Uncached crops are classified by fast, mobile-ready student CNNs.
- Models distilled from EfficientNet-B0, then pruned and quantized for sub-10 ms inference. _(TODO)_

---

### 5. Asynchronous, Pipelined Architecture

- Detection (YOLO) and classification (CNN) run in parallel CUDA streams.
- A thread-safe queue decouples detection/classification.
- Adaptive batching and buffer limits keep the system responsive, even with many cards.

---

### 6. Configuration & Testing

- Configs (`config.yaml`, `yolo.yaml`, `cnn.yaml`) are hierarchical and centralized.
- Unit tests and sample screenshots reside in `tests/fixtures`.
- `Makefile`, `pytest`, and `requirements.txt` automate setup and CI/CD.

---

### 7. Future Directions

- Mobile model optimization (TinyML)
- Automated ingestion for new expansion packs
- GPU memory profiling, latency tuning, and mixed-precision enhancements
- Continuous benchmarking to maintain <100ms end-to-end latency at high card counts

---

### 📷 Other Visual Examples

_Battle Screenshot_ <img src="assets/5.jpg" width="800">

_Card Dex Screenshot_ <img src="assets/4.png" width="800">

---

## 📁 Repository Setup Guide (WIP)

> **This guide describes how to configure, structure, and use this repo.** > _WIP: Feedback and edits welcome!_

### 1. Prerequisites

- Python ≥ 3.12
- [pip](https://pip.pypa.io/en/stable/) (latest recommended)
- `git` (for cloning, version control)

---

### 2. Installation

```bash
# Clone the repository
git clone https://gitlab.com/OctalDecoder/pokemon-tcg-pocket-card-detection.git
cd Pokemon\ TCG\ Pocket\ Card\ Detection/

# Install the requirements
pip install -e .
```

---

### 3. Screenshots for Testing

- Place images to be processed in:
  `tests/fixtures/`
- **To change this location:**
  Edit `config.yaml` → `shared` → `screenshot_dir`.

---

### 4. Data & Intermediate Model Files

- High-resolution TCG card images live in:
  `data/raw/cards/`
- All temporary checkpoints, embeddings, and intermediate files are created in:
  `data/`

---

### 5. Database Setup

- **Download or obtain** the database (`cards.db`) from \[TBD download link or instructions].
- Place it in:
  `models/cards.db`
- **To change this location:**
  Edit `config.yaml` → `shared` → `database`

---

### 6. Configuration Structure

- The main configuration file is `config.yaml`.
- The `shared` section merges into all other config sections:

  - _Best practice_: Use `shared` for common variables.
  - **Override priority:**
    `shared` values are populated into all other sections. These individual config sections (e.g., `yolo`, `cnn`) take precedence over `shared` for conflicting values.

---

### 7. Card Image Directory & Naming Convention

- **Base directory:**
  Controlled by `config.yaml` → `shared` → `card_images_dir`
- **Class subfolders:**
  Must match names in `config.yaml` → `shared` → `classifiers` (e.g., `fullart`, `standard`)
- **File naming:**
  Place images inside the relevant classifier subfolder, named as `[SeriesID] [CardID].png`

**Example** (for `card_images_dir: data/raw/cards`, `classifiers: fullart, standard`):

```
data/raw/cards/standard/A2b 32.png
data/raw/cards/fullart/S5 100.png
```

---

### 8. Output Directories

- By default, processed images, results, and generated data will be written to:

  - Output dir specified in `config.yaml` → `shared` → `output_dir`

- Subdirectories (e.g., `output/screenshot_pipeline/`) are automatically created as needed.

---

### 9. Running the Pipeline

To process screenshots and generate results:

```bash
card-detector
```

---

### 10. Work-in-Progress Notes

- Documentation is evolving; structure and locations may shift.
- Database download location is **TBD**.
- Please keep `config.yaml` and directory structure updated if you move or rename files.

---

### ⚡ Quick Reference

| Purpose                   | Path              | Config Key               |
| ------------------------- | ----------------- | ------------------------ |
| Test images (screenshots) | `tests/fixtures/` | `shared.screenshot_dir`  |
| Raw card images           | `data/raw/cards/` | `shared.card_images_dir` |
| Database file             | `models/cards.db` | `shared.database`        |
| Output/results            | `output/`         | `shared.output_dir`      |

---

## 🧪 Real Time Processing Research Suggestions

### YOLOv8 Tracking + pHash Filtering

- **Integrate tracking into YOLOv8**

  - Enable Ultralytics’ built-in tracker (e.g., ByteTrack) so that each detected bounding box gets a persistent `track_id`.
  - For each new detection, immediately compute a perceptual hash (`pHash`) of the cropped card. This should take \~1 ms per crop if using a fast hashing library (e.g., [imagehash](https://github.com/JohannesBuchner/imagehash) with OpenCV preprocess).
  - Maintain a **hash → (label, last_seen_frame)** hashtable (LRU-evicted when capacity > 256). After computing `pHash`, look up any existing entry whose Hamming distance ≤ δ (tuning threshold, e.g. δ = 5 for a 64-bit hash).

    - If a match exists and its `last_seen_frame` is within the last 5 frames, **skip** the CNN forward pass and reuse that label.
    - Otherwise, schedule a CNN inference, then store `(hash, label, current_frame)` in both the per-track cache and the global hashtable.

- **Per-track caching**

  - Maintain `track_id → (hash, label, last_seen_frame)`. On each new frame:

    1. If `track_id` is already active AND (frame_difference ≤ 5) AND (Hamming distance between current `pHash` and cached `hash` ≤ δ), reuse `label`.
    2. If the cached label’s CNN-confidence dropped below 0.5 (adaptive CNN invocation), force a fresh CNN run even if hash matched.
    3. Otherwise (new track or hash drift), run CNN and update `(hash, label, last_seen_frame)` for that `track_id`.

- **Hashtable (global cache) management**

  - Use a fixed-size LRU (capacity = 256). When inserting a new `(hash → label)`, evict the least recently used entry once capacity is exceeded.
  - Update `last_seen_frame` on every reuse to keep hot entries in cache.
  - This prevents the hash table from growing unbounded and ensures lookups remain O(1).

- **Tuning considerations**

  - Choose `pHash` bit-length and δ so that small lighting or viewpoint changes still fall under the threshold, but different cards do not collide. We may need to experiment with 64-bit versus 128-bit hashing.
  - If two cards have near-identical artwork (e.g., reprints), consider switching to a small embedding (128 D) extracted from a pruned CNN intermediate layer for stricter matching.
  - Always verify that the cost of computing `pHash` (1 ms) plus the dictionary lookup (< 0.1 ms) remains < the cost of a full CNN (10–20 ms).

---

### Pruning, Quantization & Distillation

- **Model Pruning**

  - Apply PyTorch’s structured channel pruning (`torch.nn.utils.prune.ln_structured`) on convolutional layers to remove low-magnitude filters. Aim to prune \~30–40% of channels, then fine-tune on the card dataset to recover any lost accuracy.
  - After pruning, measure FLOPs reduction. Target a \~1.5–2× speedup on GPU inference.

- **Quantization**

  - Export the distilled CNN to ONNX or TorchScript and run **FP16 inference** using NVIDIA’s TensorRT or PyTorch AMP. FP16 often yields \~2× speed improvements over FP32 with negligible accuracy drop.
  - Optionally, calibrate for **INT8** quantization (use a representative calibration set of \~500 card crops). INT8 can be \~3× faster, but watch for accuracy loss.
  - Benchmark:

    1. Run a 224×224 crop through FP32 → measure baseline (\~20–30 ms).
    2. Convert to FP16 → measure (\~10–15 ms).
    3. Calibrate to INT8 → measure (\~7–10 ms).

  - Automate quantization in the training script so new student models are automatically exported and optimized.

- **Knowledge Distillation into a Lightweight Student**

  - Teacher: the current EfficientNet-B0 (FP32).
  - Student: a smaller backbone such as **MobileNetV3-Small**, **ShuffleNetV2 (1.0×)**, or **SqueezeNet1.1**.
  - **Training workflow**:

    1. For each training image, compute teacher logits (soft targets).
    2. Train student with combined loss = CrossEntropy(hard labels) + KLDiv(teacher_logits, student_logits) at temperature T (e.g., T = 4).
    3. Fine-tune student until top-1 accuracy drop ≤ 1–2% compared to teacher.

  - After distillation, benchmark student at 224×224 crop: expect \~5–7 ms per inference on a modern GPU.

- **Input Resolution Tuning**

  - If YOLO crops are tight around each card, resize crops to **128×128** or **160×160** before feeding the student. Lower resolution often only costs \~1–2% accuracy but halves FLOPs.
  - Benchmark: compare student accuracy at 128×128 vs. 224×224 on a held-out set of 1,000 card crops. If acceptable, standardize on the smaller resolution.

- **Single CNN per YOLO Class**

  - We have two YOLO output classes (“fullart” vs. “standard”). Maintain a separate student CNN for each. Each student can be specialized:

    - **Fullart_CNN** trained only on full-art card crops.
    - **Standard_CNN** trained only on standard card crops.

  - Benefits: fewer output classes per model allows a smaller classification head and potentially slightly faster inference.

- **Benchmark & Iterate**

  - After pruning/distillation/quantization, measure:

    1. End-to-end inference time for N new crops (e.g., N=10) on GPU.
    2. Compare accuracy to baseline unpruned model.

  - Tweak prune sparsity or student architecture until we hit \~5 ms per crop with ≥ 97% classification accuracy.

---

### Asynchronous Pipeline & GPU Pipelines

- **Decouple Detection from Classification**

  1. **YOLO Stage (GPU Stream #1)**

     - Continuously run YOLOv8 detection+tracking on each incoming screenshot (frame). Each result yields `(bbox, track_id, class_label_raw, confidence_raw)`.
     - For each track: compute `pHash` on the CPU immediately (spawn a worker thread if needed).
     - If hash matches cache (distance ≤ δ and cached confidence > 0.8), tag it as “label_ready” and skip scheduling to the CNN queue.
     - Else, enqueue the crop to the **Classification Queue** (GPU Stream #2).

  2. **Classification Stage (GPU Stream #2)**

     - Consume crops batch-wise (e.g., batch size 16 or 32) from the queue. Preallocate a single GPU tensor buffer for all incoming crops each iteration.
     - Launch a single batched forward pass for all pending crops (using the appropriate student CNN based on YOLO class).
     - After inference, update per-track cache and global hashtable with `(hash → label, confidence, last_seen_frame)`.
     - Return labels to the main thread to overlay on the display or write out results.

- **Asynchronous I/O & Pipelining**

  - While **Stream #1** (YOLO) is processing Frame N, **Stream #2** (CNN) should be classifying crops from Frame N−1.
  - Use CUDA events or Python’s
