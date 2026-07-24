"""
Rendering visual telemetry, HUD elements, and face landmark overlays using OpenCV.
"""
import cv2
import config

class HUDVisualizer:
    """
    Renders high-quality On-Screen Display (OSD) HUD.
    """
    @staticmethod
    def draw_telemetry(frame, fps, ear_left, ear_right, mar, screen_pos, scroll_mode, action_msg=""):
        h, w, _ = frame.shape
        
        # Transparent Overlay Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (10, 10), (320, 160), config.COLOR_ACCENT, 1)

        # Mode Indicator
        mode_str = "MODE: SCROLLING" if scroll_mode else "MODE: CURSOR"
        mode_color = config.COLOR_ACCENT if scroll_mode else config.COLOR_SUCCESS
        cv2.putText(frame, mode_str, (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, mode_color, 2)

        # Telemetry metrics
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 38), cv2.FONT_HERSHEY_DUPLEX, 0.6, config.COLOR_SUCCESS, 1)
        cv2.putText(frame, f"EAR (L/R): {ear_left:.2f} / {ear_right:.2f}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1)
        cv2.putText(frame, f"Screen: {screen_pos}", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1)

        # Triggered Action Notification
        if action_msg:
            cv2.putText(frame, action_msg, (25, 150), cv2.FONT_HERSHEY_DUPLEX, 0.6, config.COLOR_DANGER, 2)

    @staticmethod
    def draw_landmarks(frame, landmarks, eye_left_idx, eye_right_idx, lip_indices, iris_left_center, iris_right_center):
        h, w, _ = frame.shape
        
        # Eyes
        for idx in eye_left_idx + eye_right_idx:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, pt, 1, config.COLOR_ACCENT, -1)

        # Mouth
        for idx in lip_indices:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, pt, 2, config.COLOR_DANGER, -1)

        # Irises
        iris_l = (int(landmarks[iris_left_center].x * w), int(landmarks[iris_left_center].y * h))
        iris_r = (int(landmarks[iris_right_center].x * w), int(landmarks[iris_right_center].y * h))
        cv2.circle(frame, iris_l, 3, config.COLOR_IRIS, -1)
        cv2.circle(frame, iris_r, 3, config.COLOR_IRIS, -1)

    @staticmethod
    def draw_calibration_screen(frame, step_name, progress_ratio, step_idx, total_steps):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Target locations: Top-Left, Top-Right, Center, Bottom-Left, Bottom-Right
        targets = {
            "TOP-LEFT": (int(w * 0.15), int(h * 0.15)),
            "TOP-RIGHT": (int(w * 0.85), int(h * 0.15)),
            "CENTER": (int(w * 0.50), int(h * 0.50)),
            "BOTTOM-LEFT": (int(w * 0.15), int(h * 0.85)),
            "BOTTOM-RIGHT": (int(w * 0.85), int(h * 0.85)),
        }

        target_pos = targets.get(step_name, (int(w * 0.5), int(h * 0.5)))
        
        # Outer animated ring & inner solid dot
        radius = 24
        cv2.circle(frame, target_pos, radius, config.COLOR_ACCENT, 2)
        inner_radius = int(radius * progress_ratio)
        if inner_radius > 0:
            cv2.circle(frame, target_pos, inner_radius, config.COLOR_SUCCESS, -1)

        # Instructions
        cv2.putText(frame, f"CALIBRATION MODE ({step_idx + 1}/{total_steps})", (int(w * 0.32), 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, config.COLOR_ACCENT, 2)
        cv2.putText(frame, f"Look directly at the target: {step_name}", (int(w * 0.28), h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)

