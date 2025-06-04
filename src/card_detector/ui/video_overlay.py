import cv2

def draw_box_with_text(
    frame,
    x: int,
    y: int,
    text: str,
    color=(255, 220, 120),
    bg_color=(30, 30, 30),
    font_scale=0.7,
    thickness=2,
    margin=6,
):
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        frame,
        (x - margin, y - text_height - margin),
        (x + text_width + margin, y + baseline + margin),
        bg_color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def draw_fps_overlay(frame, fps_est: float):
    margin_x, margin_y = 20, 20
    text = f"FPS: {fps_est:.1f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    x = margin_x
    y = margin_y + text_height
    draw_box_with_text(frame, x, y, text)

def draw_live_detections_overlay(frame, live_items):
    if not live_items:
        return
    h, w = frame.shape[:2]
    margin_x, margin_y = 20, 20
    line_height = 28
    for i, (display_str, _) in enumerate(live_items):
        y = margin_y + i * line_height
        (text_width, text_height), baseline = cv2.getTextSize(
            display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        x = w - margin_x - text_width
        draw_box_with_text(frame, x, y, display_str)
