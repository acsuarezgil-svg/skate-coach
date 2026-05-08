import csv
import numpy as np
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("artifacts/session_log_mobile.csv")

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
