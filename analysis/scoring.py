import numpy as np
import streamlit as st

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

    st.subheader("Debug motion stats")
    st.write(
        "vy percentiles signed",
        {p: float(np.percentile(vy, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
    )
    st.write(
        "vy percentiles abs",
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

    v_up = min(-10.0, p5 * 0.9)
    v_down = max(10.0, p95 * 0.9)
    return float(v_up), float(v_down)

# -------------------------
# Scoring and detection
# -------------------------
def grade_ollie(score, smoothness, pop_power):
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    else:
        grade = "D"

    tags = []
    if pop_power > 300:
        tags.append("High pop")
    elif pop_power < 150:
        tags.append("Low pop")

    if smoothness > 0.012:
        tags.append("Clean")
    else:
        tags.append("Sketchy")

    if score > 85:
        tags.append("🔥 Elite")
    elif score > 70:
        tags.append("Solid")

    return grade, ", ".join(tags)


def detect_ollie_candidates_from_traj(
    traj,
    fps,
    window_s=0.7,
    step_s=0.05,
    cand_pop=0.30,
    v_up_thresh=-90.0,
    v_down_thresh=90.0,
    vy_clip=500.0,
    vy_med_floor=4.0,
):
    if len(traj) < 20:
        return []

    t = np.array([p[0] for p in traj], dtype=float)
    x = np.array([p[1] for p in traj], dtype=float)
    y = np.array([p[2] for p in traj], dtype=float)

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
            x_seg = x[idxs]
            y_seg = y_s[idxs]

            vertical_amp = float(np.max(y_seg) - np.min(y_seg))
            horizontal_amp = float(np.max(x_seg) - np.min(x_seg))

            if horizontal_amp > vertical_amp * 3.0:
                cur += float(step_s)
                continue
            
            vy_slice0 = max(idxs[0] - 1, 0)
            vy_slice1 = max(idxs[-1] - 1, 1)
            ay_slice0 = max(idxs[0] - 2, 0)
            ay_slice1 = max(idxs[-1] - 2, 1)

            vy_signed = vy[vy_slice0:vy_slice1]
            ay_seg = np.abs(ay[ay_slice0:ay_slice1])

            if vy_clip is not None and len(vy_signed):
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

            smoothness = 1.0 / (1.0 + float(np.std(vy_signed))) if len(vy_signed) else 0.0
            pop_power = abs(vmin)
            landing_power = vmax
            motion_amp = amp

            ollie_score = (
                min(pop_power / 350.0, 1.0) * 35.0
                + min(landing_power / 250.0, 1.0) * 20.0
                + min(motion_amp / 600.0, 1.0) * 20.0
                + min(pop_score / 5.0, 1.0) * 10.0
                + min(smoothness * 5.0, 1.0) * 15.0
            )

            grade, tags = grade_ollie(ollie_score, smoothness, pop_power)

            up_hits = int(np.sum(vy_signed <= float(v_up_thresh)))
            down_hits = int(np.sum(vy_signed >= float(v_down_thresh)))
            has_up_then_down = (up_hits >= 2) and (down_hits >= 2)

            if (
                (pop_score >= float(cand_pop))
                and has_up_then_down
                and amp > 220.0
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


def apply_cooldown(events, cooldown_s=0.8):
    events = sorted(events, key=lambda e: float(e.get("ollie_score", 0.0)), reverse=True)
    selected = []
    last_end = -1e9
    for e in events:
        if float(e["t0"]) >= last_end + float(cooldown_s):
            selected.append(e)
            last_end = float(e["t1"])
    return sorted(selected, key=lambda e: float(e["t0"]))