import cv2
import numpy as np
import streamlit as st

from analysis.person_detector import detect_person_roi_mediapipe

# -------------------------
# ROI detection / tracking
# -------------------------
def auto_detect_roi_motion(video_path, sample_seconds=2.0, min_area=1500, pad=40):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    ret, first = cap.read()
    if not ret:
        cap.release()
        return None

    H, W = first.shape[:2]
    max_frames = int(fps * sample_seconds)

    bg = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(bg, (21, 21), 0)

    boxes = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        diff = cv2.absdiff(bg, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if aspect_ratio > 3.0 or aspect_ratio < 0.2:
                continue
            boxes.append((x, y, w, h))

    cap.release()

    if not boxes:
        return None

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:10]

    x1 = min(x for x, y, w, h in boxes)
    y1 = min(y for x, y, w, h in boxes)
    x2 = max(x + w for x, y, w, h in boxes)
    y2 = max(y + h for x, y, w, h in boxes)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)

    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def detect_motion_roi_in_frame(prev_gray, frame, min_area=1200, pad=40):
    H, W = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio > 3.0 or aspect_ratio < 0.2:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return None, gray

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:5]

    x1 = min(x for x, y, w, h in boxes)
    y1 = min(y for x, y, w, h in boxes)
    x2 = max(x + w for x, y, w, h in boxes)
    y2 = max(y + h for x, y, w, h in boxes)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)

    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), gray

def find_skater_like_motion_roi(prev_gray, frame, min_area=2500, pad=80):
    H, W = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if h < H * 0.18:
            continue

        if w < W * 0.05:
            continue

        aspect = w / float(h)

        if aspect < 0.25 or aspect > 2.2:
            continue

        score = area + (h * 8)
        if h > H * 0.28:
            score *= 1.4

        candidates.append((score, x, y, w, h))

    if not candidates:
        return None, gray

    _, x, y, w, h = max(candidates, key=lambda c: c[0])

    extra_top = int(h * 0.90)
    extra_side = int(w * 0.50)
    extra_bottom = int(h * 0.35)

    x1 = max(0, x - pad - extra_side)
    y1 = max(0, y - pad - extra_top)
    x2 = min(W, x + w + pad + extra_side)
    y2 = min(H, y + h + pad + extra_bottom)

    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), gray


def track_roi_csrt(
    video_path,
    fps,
    tracking_step_frames=1,
    roi_mode="Auto ROI",
    roi_pad=40,
    show_debug=False,
    dynamic_roi=True,
    preview_callback=None,
):
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    H, W = frame0.shape[:2]

    def _clamp_roi(x, y, w, h):
        x = int(max(0, min(x, W - 1)))
        y = int(max(0, min(y, H - 1)))
        w = int(max(1, min(w, W - x)))
        h = int(max(1, min(h, H - y)))
        return (x, y, w, h)

    def _pad_roi(x, y, w, h, pad):
        if pad <= 0:
            return _clamp_roi(x, y, w, h)
        return _clamp_roi(x - pad, y - pad, w + 2 * pad, h + 2 * pad)

    roi = None

    if roi_mode == "Full frame":
        roi = (0, 0, W, H)

    elif roi_mode == "Auto ROI":
        roi = detect_person_roi_mediapipe(
            frame0, 
            pad=max(60, int(roi_pad)),
        )
        
        if roi is None:
            roi = auto_detect_roi_motion(
                video_path,
                sample_seconds=2.0,
                min_area=1500,
                pad=max(40, int(roi_pad)),
            )
        if roi is None:
            st.warning("Auto ROI failed — switching to Manual ROI.")
            roi_mode = "Manual"

    elif roi_mode == "Last ROI":
        roi = st.session_state.get("last_roi", None)
        if roi is None:
            st.warning("No last ROI found — switching to Auto ROI.")
            roi = auto_detect_roi_motion(video_path, sample_seconds=2.0, min_area=1500, pad=max(40, int(roi_pad)))
            if roi is None:
                roi_mode = "Manual"

    if roi_mode == "Manual":
        st.info("Manual ROI opens a desktop window. Draw around board + lower body and press ENTER.")
        roi = cv2.selectROI("Select Rider", frame0, False)
        cv2.destroyAllWindows()

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("ROI selection cancelled or invalid.")

    x, y, w, h = _pad_roi(x, y, w, h, int(roi_pad))
    roi = (x, y, w, h)
    st.session_state["last_roi"] = roi

    if show_debug:
        dbg = frame0.copy()
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB), caption="ROI overlay", width="stretch")

    try:
        tracker = cv2.TrackerCSRT_create()
    except AttributeError:
        tracker = cv2.legacy.TrackerCSRT_create()

    tracker.init(frame0, roi)
    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    traj = []
    idx = 0

    cap.release()
    cap = cv2.VideoCapture(video_path)

    while True:
        for _ in range(int(tracking_step_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
        if not ret:
            break

        success, box = tracker.update(frame)

        if not success and dynamic_roi:
            recovered_roi, _ = find_skater_like_motion_roi(
                prev_gray,
                frame,
                min_area=2500,
                pad=max(80, int(roi_pad)),
            )
            if recovered_roi is None:
                recovered_roi, _ = detect_motion_roi_in_frame(
                    prev_gray,
                    frame,
                    min_area=1200,
                    pad=max(40, int(roi_pad)),
                )
            
            if recovered_roi is not None:
                rx, ry, rw, rh = recovered_roi

                if rw * rh > 1500:
                    try:
                        tracker = cv2.TrackerCSRT_create()
                    except AttributeError:
                        tracker = cv2.legacy.TrackerCSRT_create()

                    tracker.init(frame, recovered_roi)
                    success, box = True, recovered_roi

        if success and dynamic_roi and idx % 30 == 0:
            refreshed_roi, _ = find_skater_like_motion_roi(
                prev_gray,
                frame,
                min_area=2500,
                pad=max(100, int(roi_pad)),
            )

            if refreshed_roi is not None:
                rx, ry, rw, rh = refreshed_roi
                bx, by, bw, bh = box

                old_area = float(bw * bh)
                new_area = float(rw * rh)

                if new_area > old_area * 0.85:
                    try:
                        tracker = cv2.TrackerCSRT_create()
                    except AttributeError:
                        tracker = cv2.legacy.TrackerCSRT_create()

                    tracker.init(frame, refreshed_roi)
                    success, box = True, refreshed_roi
        if success:
            x, y, w, h = box
            cx = float(x + w / 2)
            cy = float(y + 0.85 * h)
            traj.append((idx / float(fps), cx, cy))

            if preview_callback and idx % 15 == 0:
                dbg = frame.copy()

                cv2.rectangle(
                    dbg,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    (0, 255, 0),
                    2,
                )

                preview_callback(dbg)

        if dynamic_roi:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    cap.release()
    return traj
