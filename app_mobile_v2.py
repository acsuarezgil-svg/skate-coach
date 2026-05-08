# app_mobile.py
# Mobile-first Skate Flow Coach
# Keep your original app.py for desktop/tuning.
# Run this with: streamlit run app_mobile.py

from __future__ import annotations

from analysis.video_metadata import get_video_metadata
from analysis.roi import (
    auto_detect_roi_motion,
    detect_motion_roi_in_frame,
    track_roi_csrt,
)
from analysis.scoring import (
    suggest_shape_thresholds,
    detect_ollie_candidates_from_traj,
    apply_cooldown,
)



import csv
import io
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from video.clip_export import ClipSpec, export_clips_batch

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Skate Flow Coach Mobile", layout="centered")

# Optional: hard-pin ffmpeg path for Windows
FFMPEG_EXE = os.environ.get(
    "SKATE_FFMPEG_EXE",
    r"C:\Users\acsua\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
)

ffmpeg_seen = FFMPEG_EXE if Path(FFMPEG_EXE).exists() else shutil.which("ffmpeg")
if ffmpeg_seen:
    os.environ["SKATE_FFMPEG_EXE"] = ffmpeg_seen

st.title("📱 Skate Flow Coach")
st.caption("GoPro-first ollie analyzer — upload, analyze, review your best clip.")

# -------------------------
# Constants / defaults
# -------------------------
LOG_PATH = Path("artifacts/session_log_mobile.csv")
CLIP_DIR = Path("artifacts/mobile_clips").resolve()

GOPRO_DEFAULTS = {
    "cand_pop": 0.30,
    "v_up_thresh": -90.0,
    "v_down_thresh": 90.0,
    "cooldown": 0.8,
    "tracking_step": 1,
    "window_s": 0.7,
    "step_s": 0.05,
    "vy_med_floor": 4.0,
    "roi_pad": 40,
}

ADVANCED_DEFAULTS = {
    "cand_pop": 0.25,
    "v_up_thresh": -50.0,
    "v_down_thresh": 50.0,
    "cooldown": 1.0,
    "tracking_step": 1,
    "window_s": 0.7,
    "step_s": 0.08,
    "vy_med_floor": 3.0,
    "roi_pad": 60,
}


# -------------------------
# Utilities
# -------------------------
#def get_video_metadata(video_path: str) -> dict:
    #cap = cv2.VideoCapture(video_path)
    #if not cap.isOpened():
        #raise RuntimeError("Could not open the video file.")

    #fps = cap.get(cv2.CAP_PROP_FPS)
    #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #cap.release()

    #duration_s = frame_count / fps if fps and fps > 0 else None
    #return {
        #"fps": float(fps) if fps else 30.0,
        #"frame_count": frame_count,
        #"width": width,
        #"height": height,
        #"resolution": f"{width}×{height}",
        #"duration_s": float(duration_s) if duration_s else 0.0,
    #}


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


# def debug_traj_stats(traj, fps, clip_vy=True, vy_clip=500.0):
#     if len(traj) < 5:
#         st.warning("Not enough trajectory points for debug.")
#         return

#     t = np.array([p[0] for p in traj], dtype=float)
#     y = np.array([p[2] for p in traj], dtype=float)

#     win = max(5, int(float(fps) * 0.25))
#     y_s = np.convolve(y, np.ones(win) / win, mode="same")

#     dt = np.diff(t)
#     dy = np.diff(y_s)
#     dt = np.clip(dt, 1e-6, None)
#     vy = dy / dt

#     if clip_vy and vy_clip is not None:
#         vy = np.clip(vy, -float(vy_clip), float(vy_clip))

#     st.subheader("Debug motion stats")
#     st.write(
#         "vy percentiles signed",
#         {p: float(np.percentile(vy, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
#     )
#     st.write(
#         "vy percentiles abs",
#         {p: float(np.percentile(np.abs(vy), p)) for p in [50, 75, 90, 95, 99]},
#     )


# def suggest_shape_thresholds(traj, fps, vy_clip=500.0):
#     t = np.array([p[0] for p in traj], dtype=float)
#     y = np.array([p[2] for p in traj], dtype=float)

#     win = max(5, int(float(fps) * 0.25))
#     y_s = np.convolve(y, np.ones(win) / win, mode="same")

#     dt = np.diff(t)
#     dy = np.diff(y_s)
#     dt = np.clip(dt, 1e-6, None)

#     vy = dy / dt
#     vy = np.clip(vy, -float(vy_clip), float(vy_clip))

#     p5 = float(np.percentile(vy, 5))
#     p95 = float(np.percentile(vy, 95))

#     v_up = min(-10.0, p5 * 0.9)
#     v_down = max(10.0, p95 * 0.9)
#     return float(v_up), float(v_down)


# # -------------------------
# # ROI detection / tracking
# # -------------------------
# def auto_detect_roi_motion(video_path, sample_seconds=2.0, min_area=1500, pad=40):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30

#     ret, first = cap.read()
#     if not ret:
#         cap.release()
#         return None

#     H, W = first.shape[:2]
#     max_frames = int(fps * sample_seconds)

#     bg = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
#     bg = cv2.GaussianBlur(bg, (21, 21), 0)

#     boxes = []

#     for _ in range(max_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (21, 21), 0)

#         diff = cv2.absdiff(bg, gray)
#         _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
#         thresh = cv2.dilate(thresh, None, iterations=2)

#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area < min_area:
#                 continue

#             x, y, w, h = cv2.boundingRect(cnt)
#             aspect_ratio = w / float(h)
#             if aspect_ratio > 3.0 or aspect_ratio < 0.2:
#                 continue
#             boxes.append((x, y, w, h))

#     cap.release()

#     if not boxes:
#         return None

#     boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:10]

#     x1 = min(x for x, y, w, h in boxes)
#     y1 = min(y for x, y, w, h in boxes)
#     x2 = max(x + w for x, y, w, h in boxes)
#     y2 = max(y + h for x, y, w, h in boxes)

#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(W, x2 + pad)
#     y2 = min(H, y2 + pad)

#     return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


# def detect_motion_roi_in_frame(prev_gray, frame, min_area=1200, pad=40):
#     H, W = frame.shape[:2]

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (21, 21), 0)

#     diff = cv2.absdiff(prev_gray, gray)
#     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
#     thresh = cv2.dilate(thresh, None, iterations=2)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     boxes = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < min_area:
#             continue

#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = w / float(h)
#         if aspect_ratio > 3.0 or aspect_ratio < 0.2:
#             continue
#         boxes.append((x, y, w, h))

#     if not boxes:
#         return None, gray

#     boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:5]

#     x1 = min(x for x, y, w, h in boxes)
#     y1 = min(y for x, y, w, h in boxes)
#     x2 = max(x + w for x, y, w, h in boxes)
#     y2 = max(y + h for x, y, w, h in boxes)

#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(W, x2 + pad)
#     y2 = min(H, y2 + pad)

#     return (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), gray


# def track_roi_csrt(
#     video_path,
#     fps,
#     tracking_step_frames=1,
#     roi_mode="Auto ROI",
#     roi_pad=40,
#     show_debug=False,
#     dynamic_roi=True,
# ):
#     cap = cv2.VideoCapture(video_path)
#     ret, frame0 = cap.read()
#     if not ret:
#         cap.release()
#         raise RuntimeError("Could not read first frame.")

#     H, W = frame0.shape[:2]

#     def _clamp_roi(x, y, w, h):
#         x = int(max(0, min(x, W - 1)))
#         y = int(max(0, min(y, H - 1)))
#         w = int(max(1, min(w, W - x)))
#         h = int(max(1, min(h, H - y)))
#         return (x, y, w, h)

#     def _pad_roi(x, y, w, h, pad):
#         if pad <= 0:
#             return _clamp_roi(x, y, w, h)
#         return _clamp_roi(x - pad, y - pad, w + 2 * pad, h + 2 * pad)

#     roi = None

#     if roi_mode == "Full frame":
#         roi = (0, 0, W, H)

#     elif roi_mode == "Auto ROI":
#         roi = auto_detect_roi_motion(
#             video_path,
#             sample_seconds=2.0,
#             min_area=1500,
#             pad=max(40, int(roi_pad)),
#         )
#         if roi is None:
#             st.warning("Auto ROI failed — switching to Manual ROI.")
#             roi_mode = "Manual"

#     elif roi_mode == "Last ROI":
#         roi = st.session_state.get("last_roi", None)
#         if roi is None:
#             st.warning("No last ROI found — switching to Auto ROI.")
#             roi = auto_detect_roi_motion(video_path, sample_seconds=2.0, min_area=1500, pad=max(40, int(roi_pad)))
#             if roi is None:
#                 roi_mode = "Manual"

#     if roi_mode == "Manual":
#         st.info("Manual ROI opens a desktop window. Draw around board + lower body and press ENTER.")
#         roi = cv2.selectROI("Select Rider", frame0, False)
#         cv2.destroyAllWindows()

#     x, y, w, h = [int(v) for v in roi]
#     if w <= 0 or h <= 0:
#         cap.release()
#         raise RuntimeError("ROI selection cancelled or invalid.")

#     x, y, w, h = _pad_roi(x, y, w, h, int(roi_pad))
#     roi = (x, y, w, h)
#     st.session_state["last_roi"] = roi

#     if show_debug:
#         dbg = frame0.copy()
#         cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB), caption="ROI overlay", width="stretch")

#     try:
#         tracker = cv2.TrackerCSRT_create()
#     except AttributeError:
#         tracker = cv2.legacy.TrackerCSRT_create()

#     tracker.init(frame0, roi)
#     prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
#     prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

#     traj = []
#     idx = 0

#     cap.release()
#     cap = cv2.VideoCapture(video_path)

#     while True:
#         for _ in range(int(tracking_step_frames)):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             idx += 1
#         if not ret:
#             break

#         success, box = tracker.update(frame)

#         if not success and dynamic_roi:
#             recovered_roi, _ = detect_motion_roi_in_frame(
#                 prev_gray,
#                 frame,
#                 min_area=1200,
#                 pad=max(40, int(roi_pad)),
#             )
#             if recovered_roi is not None:
#                 rx, ry, rw, rh = recovered_roi
#                 if rw * rh > 1500:
#                     try:
#                         tracker = cv2.TrackerCSRT_create()
#                     except AttributeError:
#                         tracker = cv2.legacy.TrackerCSRT_create()
#                     tracker.init(frame, recovered_roi)
#                     success, box = True, recovered_roi

#         if success:
#             x, y, w, h = box
#             cx = float(x + w / 2)
#             cy = float(y + 0.85 * h)
#             traj.append((idx / float(fps), cx, cy))

#         if dynamic_roi:
#             prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

#     cap.release()
#     return traj


# # -------------------------
# # Scoring and detection
# # -------------------------
# def grade_ollie(score, smoothness, pop_power):
#     if score >= 85:
#         grade = "A"
#     elif score >= 70:
#         grade = "B"
#     elif score >= 55:
#         grade = "C"
#     else:
#         grade = "D"

#     tags = []
#     if pop_power > 300:
#         tags.append("High pop")
#     elif pop_power < 150:
#         tags.append("Low pop")

#     if smoothness > 0.012:
#         tags.append("Clean")
#     else:
#         tags.append("Sketchy")

#     if score > 85:
#         tags.append("🔥 Elite")
#     elif score > 70:
#         tags.append("Solid")

#     return grade, ", ".join(tags)


# def detect_ollie_candidates_from_traj(
#     traj,
#     fps,
#     window_s=0.7,
#     step_s=0.05,
#     cand_pop=0.30,
#     v_up_thresh=-90.0,
#     v_down_thresh=90.0,
#     vy_clip=500.0,
#     vy_med_floor=4.0,
# ):
#     if len(traj) < 20:
#         return []

#     t = np.array([p[0] for p in traj], dtype=float)
#     y = np.array([p[2] for p in traj], dtype=float)

#     win = max(5, int(float(fps) * 0.25))
#     y_s = np.convolve(y, np.ones(win) / win, mode="same")

#     dt = np.diff(t)
#     dy = np.diff(y_s)
#     dt = np.clip(dt, 1e-6, None)

#     vy = dy / dt
#     if vy_clip is not None:
#         vy = np.clip(vy, -float(vy_clip), float(vy_clip))

#     ay = np.diff(vy) / np.clip(dt[1:], 1e-6, None)

#     events = []
#     cur = float(t[0])
#     t_end = float(t[-1])

#     while cur + window_s <= t_end:
#         w0, w1 = cur, cur + window_s
#         m = (t >= w0) & (t <= w1)
#         idxs = np.where(m)[0]

#         if len(idxs) >= 10:
#             vy_slice0 = max(idxs[0] - 1, 0)
#             vy_slice1 = max(idxs[-1] - 1, 1)
#             ay_slice0 = max(idxs[0] - 2, 0)
#             ay_slice1 = max(idxs[-1] - 2, 1)

#             vy_signed = vy[vy_slice0:vy_slice1]
#             ay_seg = np.abs(ay[ay_slice0:ay_slice1])

#             if vy_clip is not None and len(vy_signed):
#                 if np.max(np.abs(vy_signed)) >= 0.98 * float(vy_clip):
#                     cur += float(step_s)
#                     continue

#             vy_abs = np.abs(vy_signed)
#             vy_med = float(np.median(vy_abs)) if len(vy_abs) else 0.0
#             if vy_med < float(vy_med_floor):
#                 cur += float(step_s)
#                 continue

#             vy_p95 = float(np.percentile(vy_abs, 95)) if len(vy_abs) else 0.0
#             ay_p95 = float(np.percentile(ay_seg, 95)) if len(ay_seg) else 0.0
#             pop_score = (vy_p95 / 400.0) + (ay_p95 / 2500.0)

#             vmin = float(np.percentile(vy_signed, 1)) if len(vy_signed) else 0.0
#             vmax = float(np.percentile(vy_signed, 99)) if len(vy_signed) else 0.0
#             amp = vmax - vmin

#             smoothness = 1.0 / (1.0 + float(np.std(vy_signed))) if len(vy_signed) else 0.0
#             pop_power = abs(vmin)
#             landing_power = vmax
#             motion_amp = amp

#             ollie_score = (
#                 min(pop_power / 350.0, 1.0) * 35.0
#                 + min(landing_power / 250.0, 1.0) * 20.0
#                 + min(motion_amp / 600.0, 1.0) * 20.0
#                 + min(pop_score / 5.0, 1.0) * 10.0
#                 + min(smoothness * 5.0, 1.0) * 15.0
#             )

#             grade, tags = grade_ollie(ollie_score, smoothness, pop_power)

#             up_hits = int(np.sum(vy_signed <= float(v_up_thresh)))
#             down_hits = int(np.sum(vy_signed >= float(v_down_thresh)))
#             has_up_then_down = (up_hits >= 2) and (down_hits >= 2)

#             if (
#                 (pop_score >= float(cand_pop))
#                 and has_up_then_down
#                 and amp > 220.0
#                 and vmin < -140
#                 and vmax > 140
#             ):
#                 events.append(
#                     {
#                         "t0": float(w0),
#                         "t1": float(w1),
#                         "pop_score": float(pop_score),
#                         "ollie_score": float(ollie_score),
#                         "pop_power": float(pop_power),
#                         "landing_power": float(landing_power),
#                         "amp": float(amp),
#                         "vy_p95": float(vy_p95),
#                         "ay_p95": float(ay_p95),
#                         "vmin": float(vmin),
#                         "vmax": float(vmax),
#                         "smoothness": float(smoothness),
#                         "grade": grade,
#                         "tags": tags,
#                     }
#                 )

#         cur += float(step_s)

#     return events


# def apply_cooldown(events, cooldown_s=0.8):
#     events = sorted(events, key=lambda e: float(e.get("ollie_score", 0.0)), reverse=True)
#     selected = []
#     last_end = -1e9
#     for e in events:
#         if float(e["t0"]) >= last_end + float(cooldown_s):
#             selected.append(e)
#             last_end = float(e["t1"])
#     return sorted(selected, key=lambda e: float(e["t0"]))


def save_session_log(video_name, meta, mode_name, ollies, settings):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_pop = max([float(e.get("pop_score", 0.0)) for e in ollies], default=0.0)
    best_score = max([float(e.get("ollie_score", 0.0)) for e in ollies], default=0.0)
    avg_score = float(np.mean([float(e.get("ollie_score", 0.0)) for e in ollies])) if ollies else 0.0

    fieldnames = [
        "timestamp",
        "video_name",
        "fps",
        "duration_s",
        "mode",
        "ollies_found",
        "best_pop",
        "best_score",
        "avg_score",
        "cand_pop",
        "cooldown_s",
        "tracking_step",
        "window_s",
        "step_s",
        "vy_med_floor",
        "roi_mode",
    ]

    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_name": video_name,
                "fps": round(float(meta.get("fps") or 0.0), 2),
                "duration_s": round(float(meta.get("duration_s") or 0.0), 2),
                "mode": mode_name,
                "ollies_found": int(len(ollies)),
                "best_pop": round(float(best_pop), 4),
                "best_score": round(float(best_score), 2),
                "avg_score": round(float(avg_score), 2),
                "cand_pop": float(settings["cand_pop"]),
                "cooldown_s": float(settings["cooldown"]),
                "tracking_step": int(settings["tracking_step"]),
                "window_s": float(settings["window_s"]),
                "step_s": float(settings["step_s"]),
                "vy_med_floor": float(settings["vy_med_floor"]),
                "roi_mode": settings["roi_mode"],
            }
        )


# -------------------------
# Mobile UI
# -------------------------
video_file = st.file_uploader(
    "Upload GoPro video",
    type=["mp4", "mov", "m4v", "mpeg4"],
    help="Best results: low GoPro-style angle with rider and board visible.",
)

advanced_mode = st.toggle("Advanced settings", value=False)

if advanced_mode:
    st.info("Advanced is for tuning or unusual angles. GoPro auto mode is recommended for normal use.")

settings = dict(ADVANCED_DEFAULTS if advanced_mode else GOPRO_DEFAULTS)
settings["roi_mode"] = "Auto ROI"
settings["dynamic_roi"] = True
settings["show_roi_debug"] = False
settings["export_mode"] = "Normal + Slow-mo"
settings["slow_factor"] = 0.5
settings["reel_size"] = 3
settings["pre_roll"] = 0.5
settings["post_roll_normal"] = 0.5
settings["post_roll_slow"] = 1.5
settings["use_auto_shape_thresholds"] = True

if advanced_mode:
    with st.expander("Advanced analysis settings", expanded=True):
        settings["roi_mode"] = st.selectbox("ROI mode", ["Auto ROI", "Last ROI", "Manual", "Full frame"], index=0)
        settings["roi_pad"] = st.slider("ROI padding", 0, 200, int(settings["roi_pad"]), 10)
        settings["dynamic_roi"] = st.checkbox("Dynamic ROI recovery", value=True)
        settings["show_roi_debug"] = st.checkbox("Show ROI overlay", value=False)
        settings["use_auto_shape_thresholds"] = st.checkbox("Auto shape thresholds", value=True)

        settings["cand_pop"] = st.slider("Candidate pop", 0.0, 3.0, float(settings["cand_pop"]), 0.05)
        settings["v_up_thresh"] = st.slider("Up velocity threshold", -800.0, -10.0, float(settings["v_up_thresh"]), 5.0)
        settings["v_down_thresh"] = st.slider("Down velocity threshold", 10.0, 800.0, float(settings["v_down_thresh"]), 5.0)
        settings["vy_med_floor"] = st.slider("Cruising reject", 0.0, 10.0, float(settings["vy_med_floor"]), 0.5)
        settings["window_s"] = st.slider("Scan window", 0.4, 1.2, float(settings["window_s"]), 0.05)
        settings["step_s"] = st.slider("Scan step", 0.02, 0.20, float(settings["step_s"]), 0.01)
        settings["cooldown"] = st.slider("Cooldown", 0.5, 3.0, float(settings["cooldown"]), 0.1)
        settings["tracking_step"] = st.selectbox("Tracking step", [1, 2, 3], index=[1, 2, 3].index(int(settings["tracking_step"])))

    with st.expander("Advanced export settings", expanded=False):
        settings["reel_size"] = st.slider("Max clips", 1, 10, 3, 1)
        settings["pre_roll"] = st.slider("Pre-roll", 0.0, 2.0, 0.5, 0.1)
        settings["post_roll_normal"] = st.slider("Post-roll normal", 0.0, 3.0, 0.5, 0.1)
        settings["post_roll_slow"] = st.slider("Post-roll slow", 0.0, 4.0, 1.5, 0.1)
        settings["export_mode"] = st.radio("Export mode", ["Normal + Slow-mo", "Slow-mo only", "Normal only"], horizontal=False)
        settings["slow_factor"] = st.selectbox("Slow-mo factor", [0.5, 0.25], index=0)

if video_file:
    st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix or ".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    meta = get_video_metadata(tmp_path)

    st.caption(f"FPS: {meta['fps']:.2f} | Duration: {meta['duration_s']:.2f}s | {meta['resolution']}")

    run = st.button("🚀 Analyze My Ollie", type="primary", use_container_width=True)

    if run:
        st.subheader("🧠 Coach Analysis")

        status_box = st.empty()
        progress_bar = st.progress(0)

        status_box.write("📹 Reading video and preparing analysis...")
        progress_bar.progress(10)

        status_box.write("🛹 Tracking skateboard and rider motion...")
        progress_bar.progress(35)


        with st.spinner("Tracking rider..."):
            traj = track_roi_csrt(
                tmp_path,
                meta["fps"],
                tracking_step_frames=settings["tracking_step"],
                roi_mode=settings["roi_mode"],
                roi_pad=settings["roi_pad"],
                show_debug=settings["show_roi_debug"],
                dynamic_roi=settings["dynamic_roi"],
            )
        
        status_box.write("📈 Building motion trajectory...")
        progress_bar.progress(60)

        eff_fps = float(meta["fps"]) / float(settings["tracking_step"])

        if settings["use_auto_shape_thresholds"]:
            v_up_eff, v_down_eff = suggest_shape_thresholds(traj, eff_fps, vy_clip=500.0)
        else:
            v_up_eff = settings["v_up_thresh"]
            v_down_eff = settings["v_down_thresh"]

        status_box.write("🔥 Detecting ollie candidates...")
        progress_bar.progress(80)
        events = detect_ollie_candidates_from_traj(
            traj,
            fps=eff_fps,
            window_s=settings["window_s"],
            step_s=settings["step_s"],
            cand_pop=settings["cand_pop"],
            v_up_thresh=v_up_eff,
            v_down_thresh=v_down_eff,
            vy_clip=500.0,
            vy_med_floor=settings["vy_med_floor"],
        )

        status_box.write("✅ Finalizing skate session report...")
        progress_bar.progress(100)

        if not events:
            st.warning("No ollie-like movement found. Try Advanced settings → lower Candidate pop or use Manual ROI.")
            st.stop()

        events_sorted = sorted(events, key=lambda e: float(e.get("ollie_score", 0.0)), reverse=True)
        best_pop = float(events_sorted[0].get("pop_score", 0.0))
        auto_floor = max(float(settings["cand_pop"]), 0.60 * best_pop)
        candidates = [e for e in events_sorted[:25] if float(e.get("pop_score", 0.0)) >= auto_floor]

        ollies_all = apply_cooldown(candidates, cooldown_s=settings["cooldown"])
        ollies = ollies_all[: int(settings["reel_size"])]

        if not ollies:
            st.warning("Ollies were detected, but none passed final selection. Try Advanced settings.")
            st.stop()

        save_session_log(video_file.name, meta, "GoPro Mobile" if not advanced_mode else "Advanced Mobile", ollies, settings)

        # Build export specs
        CLIP_DIR.mkdir(parents=True, exist_ok=True)
        clip_specs = []

        for i, e in enumerate(ollies, start=1):
            pop = float(e.get("pop_score", 0.0))
            score = float(e.get("ollie_score", 0.0))

            out_norm = str(CLIP_DIR / f"mobile_clip_{i:02d}_score{score:.1f}_pop{pop:.2f}_x1.mp4")
            out_slow = str(CLIP_DIR / f"mobile_clip_{i:02d}_score{score:.1f}_pop{pop:.2f}_x{settings['slow_factor']}.mp4")

            t0_n = max(0.0, float(e["t0"]) - float(settings["pre_roll"]))
            t1_n = min(float(meta["duration_s"]), float(e["t1"]) + float(settings["post_roll_normal"]))

            t0_s = t0_n
            t1_s = min(float(meta["duration_s"]), float(e["t1"]) + float(settings["post_roll_slow"]))

            if settings["export_mode"] in ("Normal + Slow-mo", "Normal only"):
                clip_specs.append(ClipSpec(t0=t0_n, t1=t1_n, out_path=out_norm, speed=1.0))

            if settings["export_mode"] in ("Normal + Slow-mo", "Slow-mo only"):
                clip_specs.append(ClipSpec(t0=t0_s, t1=t1_s, out_path=out_slow, speed=float(settings["slow_factor"])))

        try:
            with st.spinner("Exporting best clips..."):
                exported = export_clips_batch(tmp_path, clip_specs)
        except Exception as ex:
            st.error(f"Export failed: {ex}")
            exported = []

        best_clip = max(ollies, key=lambda e: float(e.get("ollie_score", 0.0)))
        best_idx = ollies.index(best_clip) + 1
        best_score = float(best_clip.get("ollie_score", 0.0))
        best_grade = best_clip.get("grade", "-")
        best_tags = best_clip.get("tags", "")

        st.success(
            f"🔥 Best clip | Grade {best_grade} | Score {best_score:.1f} | {best_tags}"
        )

        best_clip_path = None
        for p in exported:
            if f"mobile_clip_{best_idx:02d}_" in str(p):
                best_clip_path = p
                break

        if best_clip_path:
            st.video(str(best_clip_path))

        with st.expander("All detected clips", expanded=False):
            for i, e in enumerate(ollies, start=1):
                st.write(
                    f"Clip {i} | Grade {e.get('grade','-')} | "
                    f"score={float(e.get('ollie_score',0.0)):.1f} | "
                    f"{e.get('tags','')} | "
                    f"smooth={float(e.get('smoothness',0.0)):.4f} | "
                    f"pop={float(e.get('pop_score',0.0)):.2f}"
                )

            for p in exported:
                pp = Path(p)
                if pp.exists() and pp.stat().st_size > 0:
                    st.caption(pp.name)
                    st.video(str(pp))

        if exported:
            zip_bytes, zip_name = make_zip_bytes(exported, "mobile_ollie_reel")
            st.download_button(
                f"Download clips ZIP — {len(exported)} file(s)",
                zip_bytes,
                file_name=zip_name,
                mime="application/zip",
                use_container_width=True,
            )

# -------------------------
# Dashboard
# -------------------------
st.divider()
st.header("📊 Progress")

if LOG_PATH.exists():
    try:
        df = pd.read_csv(LOG_PATH)
    except pd.errors.ParserError:
        st.warning("Mobile session log format changed. Delete artifacts/session_log_mobile.csv and run again.")
        st.stop()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_sorted = df.sort_values("timestamp")

        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", len(df))
        c2.metric("Avg Score", round(df["avg_score"].mean(), 1))
        c3.metric("Best Score", round(df["best_score"].max(), 1))

        best_row = df.loc[df["best_score"].idxmax()]
        st.success(
            f"🔥 Best session ever | {best_row['timestamp']} | "
            f"Score={best_row['best_score']} | Ollies={best_row['ollies_found']}"
        )

        if len(df_sorted) >= 2:
            latest = df_sorted.iloc[-1]
            previous = df_sorted.iloc[-2]
            score_change = latest["best_score"] - previous["best_score"]
            ollie_change = latest["ollies_found"] - previous["ollies_found"]

            st.subheader("Latest vs Previous")
            c1, c2 = st.columns(2)
            c1.metric("Best Score Change", f"{score_change:+.1f}")
            c2.metric("Ollies Change", f"{ollie_change:+.0f}")

        st.line_chart(df_sorted.set_index("timestamp")[["best_score", "avg_score"]])

        with st.expander("Session log", expanded=False):
            st.dataframe(df_sorted.tail(10), width="stretch")
    else:
        st.info("No mobile session data yet.")
else:
    st.info("Run one mobile analysis to create progress data.")
