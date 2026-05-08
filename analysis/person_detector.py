import cv2


def detect_person_roi_mediapipe(frame, pad=60):
    """
    Safe placeholder for person detection.
    If MediaPipe old solutions API is unavailable, return None
    so the app falls back to motion ROI.
    """
    try:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            return None

        mp_pose = mp.solutions.pose

        h, w = frame.shape[:2]

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                return None

            xs = []
            ys = []

            for lm in result.pose_landmarks.landmark:
                if lm.visibility >= 0.4:
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

            if not xs or not ys:
                return None

            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(w, max(xs) + pad)
            y2 = min(h, max(ys) + pad)

            return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    except Exception:
        return None