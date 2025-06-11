def print_section_header(title, width=100, sep='-'):
    side_len = (width - len(title) - 2) // 2
    line = f"\n{sep * side_len} {title} {sep * side_len}\n"
    print(line)
    
def print_video_pipeline_settings(logger, video_pipeline):
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
    def log(line):
        if logger is not None:
            logger.info(line)
        else:
            print(line)

    log(border)
    log(f"| {'Setting'.ljust(left_width-1)}| {'Value'.ljust(right_width-1)}|")
    log(border)
    for name, val in settings:
        log(f"| {str(name).ljust(left_width-1)}| {str(val).ljust(right_width-1)}|")
    log(border)
