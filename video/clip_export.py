# video/clip_export.py
from __future__ import annotations

import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


# If ffmpeg isn't on PATH for Streamlit, hardcode it here:
FFMPEG_EXE = os.environ.get(
    "SKATE_FFMPEG_EXE",
    r"C:\Users\acsua\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
)

@dataclass
class ClipSpec:
    t0: float
    t1: float
    out_path: str
    speed: float = 1.0  # 1.0 normal, 0.5 half-speed, 0.25 quarter-speed


def _ffmpeg_path() -> str:
    """Return a valid ffmpeg executable path."""
    if FFMPEG_EXE and Path(FFMPEG_EXE).exists():
        return str(Path(FFMPEG_EXE).resolve())

    which = shutil.which("ffmpeg")
    if which:
        return which

    raise RuntimeError(
        "ffmpeg not found. Install it or set FFMPEG_EXE in video/clip_export.py."
    )


def _has_ffmpeg() -> bool:
    try:
        _ = _ffmpeg_path()
        return True
    except Exception:
        return False


def _export_clip_ffmpeg(src_video_path: str, c: ClipSpec) -> str:
    """
    Export a clip using ffmpeg in a way that:
      - produces a browser/Streamlit-friendly MP4
      - avoids 0:00 duration / broken seeking issues
    """
    ffmpeg = _ffmpeg_path()

    src = str(Path(src_video_path).resolve())
    out = str(Path(c.out_path).resolve())
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    t0 = max(0.0, float(c.t0))
    t1 = max(t0, float(c.t1))
    dur = max(0.01, t1 - t0)

    speed = float(c.speed) if float(c.speed) > 0 else 1.0
    pts_mul = 1.0 / speed  # 0.5 => 2.0, 0.25 => 4.0
    vf = f"setpts={pts_mul}*PTS" if abs(speed - 1.0) > 1e-6 else "null"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",

        # Fast seek
        "-ss", f"{t0:.3f}",
        "-i", src,

        # Duration
        "-t", f"{dur:.3f}",

        # Timestamp sanity
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",

        # Slow-mo filter
        "-vf", vf,

        # Encode as compatible MP4
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",

        # No audio (simplifies timestamp weirdness)
        "-an",

        # Helps web playback
        "-movflags", "+faststart",

        "-y",
        out,
    ]

    subprocess.run(cmd, check=True)
    return out


def export_clips_batch(src_video_path: str, clips: List[ClipSpec]) -> List[str]:
    if not clips:
        return []

    if _has_ffmpeg():
        out_paths: List[str] = []
        for c in clips:
            out_paths.append(_export_clip_ffmpeg(src_video_path, c))
        return out_paths

    raise RuntimeError("ffmpeg not available; cannot export clips.")