def print_section_header(title, width=100, sep='-'):
    side_len = (width - len(title) - 2) // 2
    line = f"\n{sep * side_len} {title} {sep * side_len}\n"
    print(line)
    
def print_video_pipeline_settings(video_pipeline):
    """
    Print the settings of the VideoPipeline instance as a two-column table.
    """
    import torch.cuda as cuda
    
    # Gather all relevant settings in (name, value) pairs
    settings = [
        ("Cuda Avaliable", f"{cuda.is_available()}"),
        ("YOLOv8 Device", next(video_pipeline.yolo.model.model.parameters()).device),
        ("CNN Device", next(video_pipeline.cnn.child_models["standard"].parameters()).device),
        ("Debug Mode", video_pipeline.debug),
        ("Display FPS", video_pipeline.display_fps),
        ("Recording FPS", video_pipeline.recording_fps),
        ("Native FPS", video_pipeline.native_fps),
        ("Detection FPS", video_pipeline.detection_fps),
        ("Detection Frame Skip", video_pipeline.detection_frame_skip),
        ("Classification Frame Skip", video_pipeline.classification_frame_skip),
        ("Turbo Mode", video_pipeline.turbo),
        ("CNN Batch Size", video_pipeline.batch_size),
        ("Detection Queue Maxsize", video_pipeline.queue_maxsize),
        ("Show FPS", video_pipeline.show_fps),
        ("Show Classifications", video_pipeline.show_classifications),
        ("Show Bounding Boxes", video_pipeline.show_bboxes),
        ("Display Video", video_pipeline.display_video),
        ("Record Video", video_pipeline.record_video),
        ("Output Directory", str(video_pipeline.output_dir)),
        ("Phashing Enabled", video_pipeline.phashing_enabled),
        ("Phash Hamming Threshold", video_pipeline.phash_hamming_thresh),
        ("Overlay Timer (s)", video_pipeline._detection_display_secs),
    ]
    
    # Calculate column widths
    left_width = max(len(name) for name, _ in settings) + 2
    right_width = max(len(str(val)) for _, val in settings) + 2
    border = "+" + "-" * left_width + "+" + "-" * right_width + "+"

    # Print the table
    print(border)
    print(f"| {'Setting'.ljust(left_width-1)}| {'Value'.ljust(right_width-1)}|")
    print(border)
    for name, val in settings:
        print(f"| {str(name).ljust(left_width-1)}| {str(val).ljust(right_width-1)}|")
    print(border)


def print_video_pipeline_metrics(video_pipeline, extra_metrics=None):
    """
    Print/log pipeline metrics in a readable, table-style format, with section headers.
    """
    # Build up the metrics table with sections.
    metrics = []
    metrics.append(("=== Pipeline Summary ===", ""))
    metrics.extend([
        ("Total Frames Processed", video_pipeline.total_frames_processed),
        ("Skipped Frames", video_pipeline.skipped_frame_count),
        ("Detected Frames", video_pipeline.detected_frames),
        ("YOLO Detections", video_pipeline.total_detections),
        ("Merged Detections", video_pipeline.total_merged_detection),
        ("Classifications Queued", video_pipeline.total_classifications_queued),
    ])
    metrics.append(("=== Time Breakdown (s) ===", ""))
    metrics.extend([
        ("YOLO Detection", f"{getattr(video_pipeline.yolo, 'det_time', 0.0):.2f}"),
        ("Crop Time", f"{getattr(video_pipeline, 'crop_time', 0.0):.2f}"),
        ("Perceptual Hashing Time", f"{getattr(video_pipeline, 'phash_time', 0.0):.2f}"),
        ("CNN Classification", f"{getattr(video_pipeline.classifier_worker, 'clf_time', 0.0):.2f}"),
        ("Display/Render", f"{getattr(video_pipeline, 'cumulative_render_time', 0.0):.2f}"),
    ])
    metrics.append(("=== Queue Metrics ===", ""))
    queue_waits = video_pipeline.queue_wait_times if hasattr(video_pipeline, 'queue_wait_times') else []
    avg_queue_wait = sum(queue_waits) / len(queue_waits) if queue_waits else 0.0
    max_queue_wait = max(queue_waits) if queue_waits else 0.0
    min_queue_wait = min(queue_waits) if queue_waits else 0.0
    metrics.extend([
        ("Avg Wait (s)", f"{avg_queue_wait:.3f}"),
        ("Min Wait (s)", f"{min_queue_wait:.3f}"),
        ("Max Wait (s)", f"{max_queue_wait:.3f}"),
        ("Samples", len(queue_waits)),
    ])
    metrics.append(("=== Performance ===", ""))
    fps = (
        video_pipeline.total_frames_processed / video_pipeline.total_time
        if getattr(video_pipeline, "total_time", 0.0) > 0 else 0.0
    )
    metrics.extend([
        ("Detected Videos", len(getattr(video_pipeline, 'all_video_detections', {}))),
        ("Overall Pipeline FPS", f"{fps:.2f}"),
        ("Total Time (s)", f"{getattr(video_pipeline, 'total_time', 0.0):.2f}"),
    ])

    # Extra user-provided metrics section
    if extra_metrics:
        metrics.append(("=== Additional Metrics ===", ""))
        metrics.extend(extra_metrics)

    # Calculate widths
    left_width = max(len(str(name)) for name, _ in metrics) + 2
    right_width = max(len(str(val)) for _, val in metrics) + 2
    border = "+" + "-" * left_width + "+" + "-" * right_width + "+"

    # Table output with section headers centered
    print(border)
    print(f"| {'Metric'.ljust(left_width-1)}| {'Value'.ljust(right_width-1)}|")
    print(border)
    
    first_header = True
    for name, val in metrics:
        if name.startswith("==="):
            # Section header, center it across both columns
            header = f" {name.strip('= ')} "
            padding = left_width + right_width
            if first_header:
                first_header = False
            else:
                print(border)
            
            print("|" + header.center(padding, " ") + " |")
            print(border)
        else:
            print(f"| {str(name).ljust(left_width-1)}| {str(val).ljust(right_width-1)}|")
    print(border + "\n")
