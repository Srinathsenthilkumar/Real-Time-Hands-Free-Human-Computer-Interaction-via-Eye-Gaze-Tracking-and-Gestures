import json
import os
import cv2
import mediapipe as mp
import numpy as np
import config
from utils.math_utils import calculate_ear, calculate_mar, get_relative_iris_pos, OneEuroFilter2D

class GazeTrackerEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.filter = OneEuroFilter2D(
            mincutoff=config.ONE_EURO_MIN_CUTOFF,
            beta=config.ONE_EURO_BETA
        )
        
        self.last_pos = None
        self.load_calibration()

    def load_calibration(self):
        if os.path.exists(config.CALIBRATION_FILE):
            try:
                with open(config.CALIBRATION_FILE, "r") as f:
                    data = json.load(f)
                    self.gaze_x_min = data.get("gaze_x_min", config.DEFAULT_GAZE_X_MIN)
                    self.gaze_x_max = data.get("gaze_x_max", config.DEFAULT_GAZE_X_MAX)
                    self.gaze_y_min = data.get("gaze_y_min", config.DEFAULT_GAZE_Y_MIN)
                    self.gaze_y_max = data.get("gaze_y_max", config.DEFAULT_GAZE_Y_MAX)
                    print(f"[INFO] Loaded calibration: X[{self.gaze_x_min:.3f}, {self.gaze_x_max:.3f}], Y[{self.gaze_y_min:.3f}, {self.gaze_y_max:.3f}]")
                    return
            except Exception as e:
                print(f"[WARN] Error loading calibration file: {e}")
        self.gaze_x_min = config.DEFAULT_GAZE_X_MIN
        self.gaze_x_max = config.DEFAULT_GAZE_X_MAX
        self.gaze_y_min = config.DEFAULT_GAZE_Y_MIN
        self.gaze_y_max = config.DEFAULT_GAZE_Y_MAX

    def save_calibration(self, x_min, x_max, y_min, y_max):
        self.gaze_x_min = x_min
        self.gaze_x_max = x_max
        self.gaze_y_min = y_min
        self.gaze_y_max = y_max
        data = {
            "gaze_x_min": x_min,
            "gaze_x_max": x_max,
            "gaze_y_min": y_min,
            "gaze_y_max": y_max
        }
        with open(config.CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Saved new calibration to {config.CALIBRATION_FILE}")

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results

    def extract_metrics(self, landmarks, img_w, img_h):
        ear_left = calculate_ear(landmarks, config.EYE_LEFT_LANDMARKS, img_w, img_h)
        ear_right = calculate_ear(landmarks, config.EYE_RIGHT_LANDMARKS, img_w, img_h)
        mar = calculate_mar(landmarks, config.LIP_TOP, config.LIP_BOTTOM, config.LIP_LEFT, config.LIP_RIGHT, img_w, img_h)

        rel_x_left, rel_y_left = get_relative_iris_pos(
            landmarks, 133, 33, 159, 145, config.IRIS_LEFT_CENTER
        )
        rel_x_right, rel_y_right = get_relative_iris_pos(
            landmarks, 362, 263, 386, 374, config.IRIS_RIGHT_CENTER
        )

        avg_rel_x = (rel_x_left + rel_x_right) / 2.0
        avg_rel_y = (rel_y_left + rel_y_right) / 2.0

        return ear_left, ear_right, mar, avg_rel_x, avg_rel_y

    def map_to_screen(self, avg_rel_x, avg_rel_y, screen_w, screen_h, timestamp=None):
        raw_x = np.interp(avg_rel_x, [self.gaze_x_min, self.gaze_x_max], [0, screen_w])
        raw_y = np.interp(avg_rel_y, [self.gaze_y_min, self.gaze_y_max], [0, screen_h])

        smoothed_x, smoothed_y = self.filter.filter((raw_x, raw_y), timestamp=timestamp)

        # Apply deadzone to eliminate micro-shake
        if self.last_pos is not None:
            dist = np.hypot(smoothed_x - self.last_pos[0], smoothed_y - self.last_pos[1])
            if dist < config.DEADZONE_PIXELS:
                return self.last_pos[0], self.last_pos[1]

        self.last_pos = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y
