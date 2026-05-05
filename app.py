# app.py
import os
import streamlit as st

# MUST be the first Streamlit call
st.set_page_config(page_title="Skate Flow Coach", layout="wide")

from pathlib import Path
import shutil
import pandas as pd


# Optional: hard-pin ffmpeg path (Windows) so clip_export can find it if PATH is weird
FFMPEG_EXE = os.environ.get(
    "SKATE_FFMPEG_EXE", 
    r"C:\Users\acsua\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
)

ffmpeg_seen = FFMPEG_EXE if Path(FFMPEG_EXE).exists() else shutil.which("ffmpeg")

#push it into env so video/clip export py can read it
if ffmpeg_seen:
    os.environ["SKATE_FFMPEG_EXE"] = ffmpeg_seen

st.title("Skate Session Flow Report — Ollie Attempt Reel")
st.caption(f"ffmpeg seen by Streamlit: {ffmpeg_seen}")

import cv2
import numpy as np
import tempfile
import io
import zipfile
from datetime import datetime
import csv

from video.clip_export import export_clips_batch, ClipSpec

# -------------------------
# Presets
# -------------------------
PRESETS = {
    # Drone / wide: rider is small => small pixel motion => low thresholds
    "Drone (wide / far) — sensitive": {
        "cand_pop": 0.10,
        "v_up_thresh": -25.0,
        "v_down_thresh": 25.0,
        "cooldown": 0.7,
        "tracking_step": 1,
        "window_s": 0.9,
        "step_s": 0.05,
        "min_pop": 0.0,   # manual only
        "vy_med_floor": 2.0,  # cruising reject (drone often small)
    },
    # GoPro low angle: rider is big => stronger pixel motion => higher thresholds
    "GoPro (low-level) — strict": {
        "cand_pop": 0.30,
        "v_up_thresh": -90.0,
        "v_down_thresh": 90.0,
        "cooldown": 0.8,
        "tracking_step": 1,
        "window_s": 0.7,
        "step_s": 0.05,
        "min_pop": 0.0,
        "vy_med_floor": 4.0,
    },
    "Balanced (default)": {
        "cand_pop": 0.25,
        "v_up_thresh": -50.0,
        "v_down_thresh": 50.0,
        "cooldown": 1.0,
        "tracking_step": 1,
        "window_s": 0.7,
        "step_s": 0.08,
        "min_pop": 0.0,
        "vy_med_floor": 3.0,
    },
}


def apply_preset(preset_name: str):
    p = PRESETS[preset_name]
    st.session_state["cand_pop"] = float(p["cand_pop"])
    st.session_state["v_up_thresh"] = float(p["v_up_thresh"])
    st.session_state["v_down_thresh"] = float(p["v_down_thresh"])
    st.session_state["cooldown"] = float(p["cooldown"])
    st.session_state["tracking_step"] = int(p["tracking_step"])
    st.session_state["window_s"] = float(p["window_s"])
    st.session_state["step_s"] = float(p["step_s"])
    st.session_state["min_pop"] = float(p["min_pop"])
    st.session_state["vy_med_floor"] = float(p["vy_med_floor"])


# -------------------------
# UI
# -------------------------

video_file = st.file_uploader(
    "Upload a wide/drone/over the top skate video",
    type=["mp4", "mov", "m4v", "mpeg4"],
)

# -------------------------
# Utilities
# -------------------------
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
        "fps": float(fps) if fps else None,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "resolution": f"{width}×{height}",
        "duration_s": float(duration_s) if duration_s else None,
    }


def make_zip_bytes(file_paths, zip_name_prefix="ollie_clips"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in file_paths:
            p = str(p)
            if Path(p).exists():
                zf.write(p, arcname=Path(p).name)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return buf.getvalue(), f"{zip_name_prefix}_{ts}.zip"


def debug_traj_stats(traj, fps, clip_vy=True, vy_clip=500.0):
    if len(traj) < 5:
        st.warning("Not enough trajectory points for debug.")
        return

    t = np.array([p[0] for p in traj], dtype=float)
    y = np.array([p[2] for p in traj], dtype=float)

    win = max(5, int(float(fps) * 0.25))
    y_s = np.convolve(y, np.ones(win) / win, mode="same")

    dt = np.diff(t)
    dy = np.diff(y_s)
    dt = np.clip(dt, 1e-6, None)
    vy = dy / dt

    if clip_vy and vy_clip is not None:
        vy = np.clip(vy, -float(vy_clip), float(vy_clip))

    st.subheader("Debug: motion stats")
    st.write(
        "vy percentiles (signed):",
        {p: float(np.percentile(vy, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
    )
    st.write(
        "vy percentiles (abs):",
        {p: float(np.percentile(np.abs(vy), p)) for p in [50, 75, 90, 95, 99]},
    )


def suggest_shape_thresholds(traj, fps, vy_clip=500.0):
    t = np.array([p[0] for p in traj], dtype=float)
    y = np.array([p[2] for p in traj], dtype=float)

    win = max(5, int(float(fps) * 0.25))
    y_s = np.convolve(y, np.ones(win) / win, mode="same")

    dt = np.diff(t)
    dy = np.diff(y_s)
    dt = np.clip(dt, 1e-6, None)

    vy = dy / dt
    vy = np.clip(vy, -float(vy_clip), float(vy_clip))

    p5 = float(np.percentile(vy, 5))
    p95 = float(np.percentile(vy, 95))

    # Nudge to avoid being too tight
    v_up = min(-10.0, p5 * 0.9)      # negative
    v_down = max(10.0, p95 * 0.9)    # positive

    return float(v_up), float(v_down)

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

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Ignore very small or edge noise
            aspect_ratio = w / float(h)

            # Keep human-like shapes (taller than wide or slightly wide)
            if aspect_ratio > 3.0 or aspect_ratio < 0.2:
                continue
            
            boxes.append((x, y, w, h))
    cap.release()
    if not boxes:
        return None

    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:10]
    

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

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

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
# -------------------------
# Tracking
# -------------------------
def track_roi_csrt(video_path, fps, tracking_step_frames=1, roi_mode="Manual", roi_pad=0, show_debug=False, dynamic_roi=False):
    """
    roi_mode:
      - "Manual"    : user draws ROI
      - "Last ROI"  : reuse last ROI stored in st.session_state["last_roi"]
      - "Full frame": track whole frame
    roi_pad: optional padding (pixels) added around ROI, clamped to frame bounds
    """
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
        x2 = x - pad
        y2 = y - pad
        w2 = w + 2 * pad
        h2 = h + 2 * pad
        return _clamp_roi(x2, y2, w2, h2)

    # ---- choose ROI ----
    roi = None

    if roi_mode == "Full frame":
        roi = (0, 0, W, H)

    elif roi_mode == "Auto ROI":
        roi = auto_detect_roi_motion(
            video_path,
            sample_seconds=2.0,
            min_area=1500,
            pad=max(40, int(roi_pad))
        )

        if roi is None:
            st.warning("Auto ROI failed — switching to Manual ROI.")
            roi_mode = "Manual"

    elif roi_mode == "Last ROI":
        roi = st.session_state.get("last_roi", None)
        if roi is None:
            st.warning("No last ROI found — switching to Manual ROI.")
            roi_mode = "Manual"

    # Manual ALWAYS handled here (outside)
    if roi_mode == "Manual":
        st.info("Draw ROI around you (board + lower body), then press ENTER.")
        roi = cv2.selectROI("Select Rider", frame0, False)
        cv2.destroyAllWindows()

    # Validate ROI
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("ROI selection cancelled or invalid. Please draw a non-empty box.")

    # Optional padding + clamp
    x, y, w, h = _pad_roi(x, y, w, h, int(roi_pad))
    roi = (x, y, w, h)

    # Save for reuse
    st.session_state["last_roi"] = roi

    # -------------------------
    # Debug: Show ROI overlay
    # -------------------------
    if show_debug:
        dbg = frame0.copy()
        x, y, w, h = map(int, roi)
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        st.image(
            cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB),
            caption="ROI overlay (first frame)",
            width="stretch"
        )

    # ---- init tracker ----
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
            recovered_roi, new_gray = detect_motion_roi_in_frame(
                prev_gray,
                frame,
                min_area=1200,
                pad=max(40, int(roi_pad))
            )

            if recovered_roi is not None:
                x, y, w, h = recovered_roi

                if w * h > 1500:
                    try:
                        tracker = cv2.TrackerCSRT_create()
                    except AttributeError:
                        tracker = cv2.legacy.TrackerCSRT_create()

                    tracker.init(frame, recovered_roi)
                    success, box = True, recovered_roi

        if success:
            x, y, w, h = box

            # OPTIONAL: slight correction from motion center
            #if dynamic_roi:
                #motion_roi, _ = detect_motion_roi_in_frame(prev_gray, frame)

                #if motion_roi is not None:
                    #mx, my, mw, mh = motion_roi

                    # Only correct if motion box is similar size (avoid jumping)
                    #if 0.5 < (mw * mh) / (w * h) < 2.0:

                    # blend tracker + motion (stabilizes drift)
                        #alpha = 0.2
                        #x = int((1 - alpha) * x + alpha * mx)
                        #y = int((1 - alpha) * y + alpha * my)
                        #w = int((1 - alpha) * w + alpha * mw)
                        #h = int((1 - alpha) * h + alpha * mh)

            cx = float(x + w / 2)
            cy = float(y + 0.85 * h)  # lower body/board emphasis
            traj.append((idx / float(fps), cx, cy))

        if dynamic_roi:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    cap.release()
    return traj

def grade_ollie(score, smoothness, pop_power):
    # Letter grade
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    else:
        grade = "D"

    # Tags
    tags = []

    # Power tags
    if pop_power > 300:
        tags.append("High pop")
    elif pop_power < 150:
        tags.append("Low pop")

    # Smoothness tags
    if smoothness > 0.012:
        tags.append("Clean")
    else:
        tags.append("Sketchy")

    # Bonus tags (based on score)
    if score > 85:
        tags.append("🔥 Elite")
    elif score > 70:
        tags.append("Solid")

    return grade, ", ".join(tags)
# -------------------------
# Motion Detection (ollie shape filter)
# -------------------------
def detect_ollie_candidates_from_traj(
    traj,
    fps,
    window_s=0.7,
    step_s=0.08,
    cand_pop=0.25,
    v_up_thresh=-50.0,
    v_down_thresh=50.0,
    vy_clip=500.0,
    vy_med_floor=3.0,
):
    """
    Heuristic:
      1) motion burst (pop_score)
      2) ollie-like shape: goes UP then DOWN in the window (hit-count thresholds)
      3) reject tracker spikes (windows hitting clip ceiling)
      4) reject cruising (median abs vy too low)
    """
    if len(traj) < 20:
        return []

    t = np.array([p[0] for p in traj], dtype=float)
    y = np.array([p[2] for p in traj], dtype=float)

    # smooth y
    win = max(5, int(float(fps) * 0.25))
    y_s = np.convolve(y, np.ones(win) / win, mode="same")

    dt = np.diff(t)
    dy = np.diff(y_s)
    dt = np.clip(dt, 1e-6, None)

    vy = dy / dt
    if vy_clip is not None:
        vy = np.clip(vy, -float(vy_clip), float(vy_clip))

    ay = np.diff(vy) / np.clip(dt[1:], 1e-6, None)

    events = []
    cur = float(t[0])
    t_end = float(t[-1])

    while cur + window_s <= t_end:
        w0, w1 = cur, cur + window_s
        m = (t >= w0) & (t <= w1)
        idxs = np.where(m)[0]

        if len(idxs) >= 10:
            vy_slice0 = max(idxs[0] - 1, 0)
            vy_slice1 = max(idxs[-1] - 1, 1)
            ay_slice0 = max(idxs[0] - 2, 0)
            ay_slice1 = max(idxs[-1] - 2, 1)

            vy_signed = vy[vy_slice0:vy_slice1]
            ay_seg = np.abs(ay[ay_slice0:ay_slice1])

            if vy_clip is not None:
                if np.max(np.abs(vy_signed)) >= 0.98 * float(vy_clip):
                    cur += float(step_s)
                    continue

            vy_abs = np.abs(vy_signed)

            vy_med = float(np.median(vy_abs)) if len(vy_abs) else 0.0
            if vy_med < float(vy_med_floor):
                cur += float(step_s)
                continue

            vy_p95 = float(np.percentile(vy_abs, 95)) if len(vy_abs) else 0.0
            ay_p95 = float(np.percentile(ay_seg, 95)) if len(ay_seg) else 0.0
            pop_score = (vy_p95 / 400.0) + (ay_p95 / 2500.0)

            vmin = float(np.percentile(vy_signed, 1)) if len(vy_signed) else 0.0
            vmax = float(np.percentile(vy_signed, 99)) if len(vy_signed) else 0.0

            amp = vmax - vmin
            smoothness = 1.0 / (1.0 + float(np.std(vy_signed)))

            pop_power = abs(vmin)
            landing_power = vmax
            motion_amp = amp

            ollie_score = (
                min(pop_power / 350.0, 1.0) * 35.0 +
                min(landing_power / 250.0, 1.0) * 20.0 +
                min(motion_amp / 600.0, 1.0) * 20.0 +
                min(pop_score / 5.0, 1.0) * 10.0 +
                min(smoothness * 5.0, 1.0) * 15.0
            )

            grade, tags = grade_ollie(ollie_score, smoothness, pop_power)

            up_hits = int(np.sum(vy_signed <= float(v_up_thresh)))
            down_hits = int(np.sum(vy_signed >= float(v_down_thresh)))
            has_up_then_down = (up_hits >= 2) and (down_hits >= 2)

            if (
                (pop_score >= float(cand_pop))
                and has_up_then_down
                and amp > 220.0   #  NEW filter
                and vmin < -140
                and vmax > 140
            ):
                events.append(
                    {
                        "t0": float(w0),
                        "t1": float(w1),
                        "pop_score": float(pop_score),
                        "ollie_score": float(ollie_score),
                        "pop_power": float(pop_power),
                        "landing_power": float(landing_power),
                        "amp": float(amp),
                        "vy_p95": float(vy_p95),
                        "ay_p95": float(ay_p95),
                        "vmin": float(vmin),
                        "vmax": float(vmax),
                        "smoothness": float(smoothness),
                        "grade": grade,
                        "tags": tags,
                    }
                )

        cur += float(step_s)

    return events


def apply_cooldown(events, cooldown_s=1.0):
    events = sorted(events, key=lambda e: float(e.get("pop_score", 0.0)), reverse=True)
    selected = []
    last_end = -1e9
    for e in events:
        if float(e["t0"]) >= last_end + float(cooldown_s):
            selected.append(e)
            last_end = float(e["t1"])
    return sorted(selected, key=lambda e: float(e["t0"]))


# -------------------------
# Main App
# -------------------------
if video_file:
    st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix or ".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    meta = get_video_metadata(tmp_path)

    st.write(f"FPS: {meta['fps']:.2f}")
    st.write(f"Duration: {meta['duration_s']:.2f}s")
    st.write(f"Resolution: {meta['resolution']}")

    # Shot type & presets
    shot_type = st.selectbox("Shot type", ["Drone (wide / far)", "GoPro (low-level)", "Unknown"], index=0)
    preset_options = ["Auto (recommended)"] + list(PRESETS.keys())
    preset_choice = st.selectbox("Detection preset", preset_options, index=0)

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Apply preset"):
            if preset_choice == "Auto (recommended)":
                if shot_type.startswith("Drone"):
                    apply_preset("Drone (wide / far) — sensitive")
                elif shot_type.startswith("GoPro"):
                    apply_preset("GoPro (low-level) — strict")
                else:
                    apply_preset(
                        "GoPro (low-level) — strict"
                        if (meta["fps"] or 0) >= 50
                        else "Drone (wide / far) — sensitive"
                    )
            else:
                apply_preset(preset_choice)

    with c2:
        st.caption("Tip: Apply a preset once, then tweak sliders if needed (they keep their values).")

    mode = st.radio("Detection Mode", ["Auto", "Manual"], horizontal=True)

    sensitivity = ""
    if mode == "Auto":
        sensitivity = st.selectbox("Sensitivity", ["Low", "Medium", "High"], index=1)

    top_k = st.slider("Max clips (export/preview)", 1, 30, 10, 1)

    reel_size = st.selectbox(
        "Reel size shortcut",
        options=[3, 5, 10, 15, top_k],
        index=2,
        help="Auto will find candidates, then you choose how many to show/export.",
    )
    reel_size = int(min(int(reel_size), int(top_k)))

    cand_pop = st.slider(
        "Candidate pop (pre-filter)",
        0.0,
        3.0,
        value=float(st.session_state.get("cand_pop", 0.5)),
        step=0.05,
        key="cand_pop",
    )

    st.caption("Ollie-shape filter (reduces false positives like rolling/crouching)")
    cA, cB = st.columns(2)

    v_up_thresh = cA.slider(
        "Up velocity threshold (vy ≤ ...)",
        -800.0,
        -10.0,
        value=float(st.session_state.get("v_up_thresh", -50.0)),
        step=5.0,
        key="v_up_thresh",
    )

    v_down_thresh = cB.slider(
        "Down velocity threshold (vy ≥ ...)",
        10.0,
        800.0,
        value=float(st.session_state.get("v_down_thresh", 50.0)),
        step=5.0,
        key="v_down_thresh",
    )

    if mode == "Manual":
        min_pop = st.slider(
            "Min pop score (final)",
            0.0,
            10.0,
            value=float(st.session_state.get("min_pop", 0.0)),
            step=0.05,
            key="min_pop",
        )
    else:
        min_pop = 0.0
        st.caption("Auto mode selects the strongest attempts (no fixed threshold).")

    cooldown = st.slider(
        "Cooldown (s)",
        0.5,
        3.0,
        value=float(st.session_state.get("cooldown", 1.0)),
        step=0.1,
        key="cooldown",
    )

    tracking_step = st.selectbox(
        "Tracking step (frames)",
        [1, 2, 3],
        index=[1, 2, 3].index(int(st.session_state.get("tracking_step", 1))),
        key="tracking_step",
    )


    # -------------------------
    # ROI Controls
    # -------------------------
    st.subheader("ROI Settings")

    roi_mode = st.selectbox(
        "ROI Mode",
        ["Auto ROI", "Manual", "Last ROI", "Full frame"],
        index=1,
        help="Auto ROI tries to find the rider/board from motion. Manual lets you draw the box yourself." 
    )

    roi_pad = st.slider(
        "ROI padding (px)",
        0,
        200,
        0,
        10,
        help="Adds margin around ROI. Useful for drone/wide shots."
    )

    show_roi_debug = st.checkbox(
        "Show ROI overlay on first frame (debug)",
        value=False
    )

    dynamic_roi = st.checkbox(
        "Dynamic ROI recovery",
        value=True,
        help="If CSRT loses the rider, try to recover using motion detection."
    )

    window_s = st.slider(
        "Scan window (s)",
        0.4,
        1.2,
        value=float(st.session_state.get("window_s", 0.7)),
        step=0.05,
        key="window_s",
    )

    step_s = st.slider(
        "Scan step (s)",
        0.02,
        0.20,
        value=float(st.session_state.get("step_s", 0.08)),
        step=0.01,
        key="step_s",
    )

    vy_med_floor = st.slider(
        "Cruising reject (median |vy| floor)",
        0.0,
        10.0,
        value=float(st.session_state.get("vy_med_floor", 3.0)),
        step=0.5,
        key="vy_med_floor",
        help="Higher = fewer false positives while cruising; too high may miss small drone ollies.",
    )

    export_mode = st.radio(
        "Export mode",
        ["Normal + Slow-mo", "Slow-mo only", "Normal only"],
        horizontal=True,
    )
    slow_factor = st.selectbox("Slow-mo factor", [0.5, 0.25], index=0)

    debug = st.checkbox("Debug motion stats", value=False)

    run = st.button("Run Ollie Reel")
    test_export = st.button("Test Export 2s (debug)")

    # -------------------------
    # Quick export sanity check
    # -------------------------
    if test_export:
        out_dir = (Path("artifacts") / "clips").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        test_specs = [
            ClipSpec(t0=0.0, t1=2.0, out_path=str(out_dir / "test_x1.mp4"), speed=1.0),
            ClipSpec(t0=0.0, t1=2.0, out_path=str(out_dir / "test_x0.5.mp4"), speed=0.5),
        ]
        exported = export_clips_batch(tmp_path, test_specs)
        st.write("Exported:", exported)

        st.subheader("Preview (test exports)")
        for p in exported:
            pp = Path(p)
            if pp.exists() and pp.stat().st_size > 0:
                st.caption(pp.name)
                st.video(str(pp)) #st.video(pp.read_bytes(), format="video/mp4")
            else:
                st.warning(f"Missing/empty preview: {pp}")

    # -------------------------
    # Main run
    # -------------------------
    if run:
        # tracking
        with st.spinner("Tracking..."):
            traj = track_roi_csrt(
                tmp_path, 
                meta["fps"],
                tracking_step_frames=tracking_step,
                roi_mode=roi_mode,
                roi_pad=roi_pad,
                show_debug=show_roi_debug,
                dynamic_roi=dynamic_roi,
            )
        eff_fps = float(meta["fps"]) / float(tracking_step)

        # Auto thresholds only for this run (do NOT write to session_state)
        if mode == "Auto":
            auto_v_up, auto_v_down = suggest_shape_thresholds(traj, eff_fps, vy_clip=500.0)
            v_up_eff = auto_v_up
            v_down_eff = auto_v_down
            st.caption(
                f"Auto shape thresholds from motion (this run): "
                f"up={v_up_eff:.1f}, down={v_down_eff:.1f}"
            )
        else:
            v_up_eff = v_up_thresh
            v_down_eff = v_down_thresh

        if debug:
            debug_traj_stats(traj, eff_fps, clip_vy=True, vy_clip=500.0)

        # detection
        with st.spinner("Detecting pops..."):
            events = detect_ollie_candidates_from_traj(
                traj,
                fps=eff_fps,
                window_s=window_s,
                step_s=step_s,
                cand_pop=cand_pop,
                v_up_thresh=v_up_eff,
                v_down_thresh=v_down_eff,
                vy_clip=500.0,
                vy_med_floor=vy_med_floor,
            )

        if not events:
            st.warning(
                "No ollie-like motion bursts found (try lowering Candidate pop, loosening thresholds, or lowering cruising reject)."
            )
            st.stop()

        events_sorted = sorted(events, key=lambda e: float(e.get("ollie_score", 0.0)), reverse=True)

        # AUTO vs MANUAL selection
        if mode == "Auto":
            best_pop = float(events_sorted[0].get("pop_score", 0.0))
            pool_size = max(top_k * 5, 25)
            auto_floor = max(float(cand_pop), 0.60 * best_pop)
            candidates = [e for e in events_sorted[:pool_size] if float(e["pop_score"]) >= auto_floor]
            st.caption(f"Auto: best pop = {best_pop:.2f} | pool size = {pool_size} | auto floor = {auto_floor:.2f}")
        else:
            candidates = [e for e in events_sorted if float(e.get("pop_score", 0.0)) >= float(min_pop)]
            st.caption(f"Manual threshold = {float(min_pop):.2f}")

        # Cooldown then keep reel_size
        ollies_all = apply_cooldown(candidates, cooldown_s=cooldown)
        ollies = ollies_all[:reel_size]

        st.write(f"Ollies detected (after cooldown): {len(ollies_all)}")
        st.write(f"Exporting Top {len(ollies)} attempts")

        with st.expander("See more detected attempts (not exported)"):
            for j, e in enumerate(ollies_all[:30], start=1):
                st.write(
                    f"{j:02d} | t0={e['t0']:.2f}s | score={e.get('ollie_score',0):.1f} | pop={e['pop_score']:.2f} | "
                    f"vmin={e.get('vmin',0.0):.1f} | vmax={e.get('vmax',0.0):.1f}"
                )

        if not ollies:
            st.warning("No ollies passed selection (try lowering cruising reject, lowering min_pop, or lowering thresholds).")
            st.stop()

        # -------------------------
        # Session Log (CSV)
        # -------------------------
        log_path = (Path("artifacts") / "session_log_v2.csv").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        best_pop_final = max([float(e.get("pop_score", 0.0)) for e in ollies], default=0.0)

        fieldnames = [
            "timestamp",
            "video_name",
            "fps",
            "duration_s",
            "mode",
            "sensitivity",
            "cand_pop",
            "cooldown_s",
            "tracking_step",
            "top_k",
            "ollies_found",
            "best_pop",
            "v_up_thresh",
            "v_down_thresh",
            "window_s",
            "step_s",
            "vy_med_floor",
            "shot_type",
            "preset_choice",
            "best_score",
            "avg_score",
        ]

        write_header = not log_path.exists()
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "video_name": getattr(video_file, "name", "uploaded_video"),
                    "fps": round(float(meta["fps"] or 0.0), 2),
                    "duration_s": round(float(meta["duration_s"] or 0.0), 2),
                    "mode": mode,
                    "sensitivity": sensitivity if mode == "Auto" else "",
                    "cand_pop": float(cand_pop),
                    "cooldown_s": float(cooldown),
                    "tracking_step": int(tracking_step),
                    "top_k": int(top_k),
                    "ollies_found": int(len(ollies)),
                    "best_pop": round(float(best_pop_final), 4),
                    "best_score": round(max([float(e.get("ollie_score", 0.0)) for e in ollies], default=0.0), 2),
                    "avg_score": round(np.mean([float(e.get("ollie_score", 0.0)) for e in ollies]) if ollies else 0.0, 2),
                    "v_up_thresh": float(v_up_eff),
                    "v_down_thresh": float(v_down_eff),
                    "window_s": float(window_s),
                    "step_s": float(step_s),
                    "vy_med_floor": float(vy_med_floor),
                    "shot_type": shot_type,
                    "preset_choice": preset_choice,
                }
            )

        st.caption("Session saved → artifacts/session_log.csv ")
        st.download_button(
            "Download session log (CSV)",
            data=log_path.read_bytes(),
            file_name="session_log.csv",
            mime="text/csv",
        )

        #sliders controls
        st.subheader("Export Timing")

        pre_roll = st.slider("Pre-roll (seconds before ollie)", 0.0, 2.0, 0.5, 0.1)
        post_roll_normal = st.slider("Post-roll normal clip", 0.0, 3.0, 0.5, 0.1)    
        post_roll_slow = st.slider("Post-roll slow motion", 0.0, 4.0, 1.5, 0.1)

        st.subheader("Export Timing")
        # -------------------------
        # Export + Preview + ZIP (batch)
        # -------------------------
        out_dir = (Path("artifacts") / "clips").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        video_dur = float(meta.get("duration_s") or 0.0)

        clip_specs = []
        for i, e in enumerate(ollies, start=1):
            pop = float(e.get("pop_score", 0.0))

            out_norm = str(out_dir / f"clip_{i:02d}_pop{pop:.2f}_x1.mp4")
            out_slow = str(out_dir / f"clip_{i:02d}_pop{pop:.2f}_x{slow_factor}.mp4")
            
            # Normal window
            t0_n = max(0.0, float(e["t0"]) - float(pre_roll))
            t1_n = min(float(meta["duration_s"] or 0.0), float(e["t1"]) + float(post_roll_normal))

            # Slow-mo window (longer)
            t0_s = t0_n
            t1_s = min(float(meta["duration_s"] or 0.0), float(e["t1"]) + float(post_roll_slow))

            if export_mode in ("Normal + Slow-mo", "Normal only"):
                clip_specs.append(ClipSpec(t0=t0_n, t1=t1_n, out_path=out_norm, speed=1.0))

            if export_mode in ("Normal + Slow-mo", "Slow-mo only"):
                clip_specs.append(ClipSpec(t0=t0_s, t1=t1_s, out_path=out_slow, speed=float(slow_factor)))

        exported = []
        try:
            with st.spinner(f"Exporting {len(clip_specs)} clip file(s) in one pass..."):
                exported = export_clips_batch(tmp_path, clip_specs)

            # 🔥 Best Clip of This Session
            best_clip = max(ollies, key=lambda e: float(e.get("ollie_score", 0.0)))
            best_idx = ollies.index(best_clip) + 1
            best_score = float(best_clip.get("ollie_score", 0.0))

            st.subheader("🔥 Best Clip of This Session")
            st.write(
                f"Clip {best_idx} | Grade {best_clip.get('grade','-')} | "
                f"score={best_score:.1f} | {best_clip.get('tags','')}"
            )

            best_clip_path = None
            for p in exported:
                if f"clip_{best_idx:02d}_" in str(p):
                    best_clip_path = p
                    break
                
            if best_clip_path:
                st.video(str(best_clip_path))    
                
            for i, e in enumerate(ollies, start=1):
                pop = float(e.get("pop_score", 0.0))
                st.write(
                    f"Clip {i} | Grade {e.get('grade','-')} | score={float(e.get('ollie_score',0.0)):.1f} | "
                    f"{e.get('tags','')} | smooth={float(e.get('smoothness',0.0)):.4f} | "
                    f"pop={pop:.2f} | vmin={float(e.get('vmin',0.0)):.1f} | vmax={float(e.get('vmax',0.0)):.1f}"
                )

            st.subheader("Preview")
            for p in exported:
                pp = Path(p)
                if pp.exists() and pp.stat().st_size > 0:
                    st.caption(pp.name)
                    st.video(str(pp)) #st.video(pp.read_bytes(), format="video/mp4")
                else:
                    st.warning(f"Missing/empty preview: {pp}")

        except Exception as ex:
            st.error(f"Export failed: {ex}")
            exported = []

        st.subheader("Share / Download")
        
        
        if len(exported) == 0:
            st.info("No clips exported yet.")
        else:
            zip_bytes, zip_name = make_zip_bytes(exported, "ollie_reel")
            st.download_button(
                f"Download exported clips (ZIP) — {len(exported)} file(s)",
                zip_bytes,
                file_name=zip_name,
                mime="application/zip",
            )
# =========================
# DASHBOARD
# =========================
st.divider()
st.header("📊 Session Dashboard")

log_path = Path("artifacts/session_log_v2.csv")

if log_path.exists():
    try:    
        df = pd.read_csv(log_path)
    except pd.errors.ParserError:
        st.warning("Session log format changed. Delete artifacts/session_log_v2.csv and run again.")
        st.stop()
    if not df.empty:
        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Sessions", len(df))

        if "avg_score" in df.columns:
            col2.metric("Avg Score", round(df["avg_score"].mean(), 1))
        else:
            col2.metric("Avg Ollies", round(df["ollies_found"].mean(), 1))

        if "best_score" in df.columns:
            col3.metric("Best Score Ever", round(df["best_score"].max(), 1))
        else:
            col3.metric("Best Pop Ever", round(df["best_pop"].max(), 1))
        
        df_sorted = df.sort_values("timestamp")   
        
        if "best_score" in df.columns:
            best_row = df.loc[df["best_score"].idxmax()]

            st.success(
                f"🔥 Best Session Ever | "
                f"{best_row['timestamp']} | "
                f"Score={best_row['best_score']} | "
                f"Ollies={best_row['ollies_found']}"
            ) 
                # Compare latest session vs previous session
        if "best_score" in df.columns and len(df_sorted) >= 2:
            latest = df_sorted.iloc[-1]
            previous = df_sorted.iloc[-2]

            score_change = latest["best_score"] - previous["best_score"]
            ollie_change = latest["ollies_found"] - previous["ollies_found"]

            st.subheader("📊 Latest Session vs Previous")

            c1, c2 = st.columns(2)

            c1.metric(
                "Best Score Change",
                f"{score_change:+.1f}",
                help="Latest session best score minus previous session best score"
            )

            c2.metric(
                "Ollies Found Change",
                f"{ollie_change:+.0f}",
                help="Latest session ollies found minus previous session ollies found"
            )

        st.subheader("📈 Progress Over Time")


        if "best_score" in df.columns:
            st.line_chart(
                df_sorted.set_index("timestamp")[["best_score", "avg_score"]]
            )
        else:
            st.warning("No score data yet — run a new session.")

        st.subheader("📋 Raw Data")
        st.dataframe(df_sorted.tail(10), width="stretch")

    else:
        st.info("No session data yet.")
else:
    st.info("Run a session to generate dashboard data.")

