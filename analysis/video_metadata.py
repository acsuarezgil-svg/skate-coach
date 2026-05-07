import cv2

def get_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_s = frame_count / fps if fps and fps > 0 else None
    return {
        "fps": float(fps) if fps else 30.0,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "resolution": f"{width}×{height}",
        "duration_s": float(duration_s) if duration_s else 0.0,
    }
