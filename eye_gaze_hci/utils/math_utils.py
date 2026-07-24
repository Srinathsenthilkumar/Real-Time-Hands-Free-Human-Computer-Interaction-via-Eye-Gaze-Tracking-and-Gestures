import math
import numpy as np

class LowPassFilter:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.y = None

    def filter(self, value):
        if self.y is None:
            self.y = value
        else:
            self.y = self.alpha * value + (1.0 - self.alpha) * self.y
        return self.y

    def reset(self):
        self.y = None

class OneEuroFilter2D:
    """
    1-Euro Filter for 2D Spatial Coordinates (x, y).
    Eliminates jitter when stationary, while removing lag during rapid glances.
    Reference: Casiez et al., CHI 2012.
    """
    def __init__(self, freq=30.0, mincutoff=0.1, beta=0.05, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        self.x_filter = LowPassFilter()
        self.y_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.dy_filter = LowPassFilter()
        
        self.last_time = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, pos, timestamp=None):
        x, y = pos
        if self.last_time is not None and timestamp is not None:
            dt = timestamp - self.last_time
            if dt > 0:
                self.freq = 1.0 / dt
        self.last_time = timestamp

        # Filter X
        prev_x = self.x_filter.y
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.filter(dx)
        cutoff_x = self.mincutoff + self.beta * abs(edx)
        self.x_filter.alpha = self._alpha(cutoff_x)
        filtered_x = self.x_filter.filter(x)

        # Filter Y
        prev_y = self.y_filter.y
        dy = 0.0 if prev_y is None else (y - prev_y) * self.freq
        edy = self.dy_filter.filter(dy)
        cutoff_y = self.mincutoff + self.beta * abs(edy)
        self.y_filter.alpha = self._alpha(cutoff_y)
        filtered_y = self.y_filter.filter(y)

        return filtered_x, filtered_y

    def reset(self):
        self.x_filter.reset()
        self.y_filter.reset()
        self.dx_filter.reset()
        self.dy_filter.reset()
        self.last_time = None

class AdaptiveEMA:
    """
    Adaptive Exponential Moving Average Filter.
    Dynamically adjusts alpha based on velocity to balance responsiveness and anti-jitter.
    """
    def __init__(self, alpha_min=0.05, alpha_max=0.5, velocity_thresh=50.0):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.velocity_thresh = velocity_thresh
        self.prev_val = None

    def update(self, current_val):
        if self.prev_val is None:
            self.prev_val = np.array(current_val, dtype=float)
            return self.prev_val

        current_val = np.array(current_val, dtype=float)
        velocity = np.linalg.norm(current_val - self.prev_val)
        
        # Scale alpha based on velocity ratio
        t = min(velocity / self.velocity_thresh, 1.0)
        alpha = self.alpha_min + t * (self.alpha_max - self.alpha_min)
        
        smoothed_val = self.prev_val + alpha * (current_val - self.prev_val)
        self.prev_val = smoothed_val
        return smoothed_val

    def reset(self):
        self.prev_val = None


class KalmanFilter2D:
    """
    2D Kalman Filter for smooth cursor trajectory tracking.
    """
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        # State vector: [x, y, vx, vy]^T
        self.x = np.zeros((4, 1))
        # State transition matrix
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        # Covariance matrix
        self.P = np.eye(4) * 1000
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise

    def update(self, z):
        z = np.array(z, dtype=float).reshape((2, 1))
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x[0, 0], self.x[1, 0]


def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Computes Eye Aspect Ratio (EAR) using Euclidean distance.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in eye_indices]
    
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def calculate_mar(landmarks, top_idx, bot_idx, left_idx, right_idx, img_w, img_h):
    """
    Computes Mouth Aspect Ratio (MAR).
    MAR = ||top - bottom|| / ||left - right||
    """
    p_top = np.array([landmarks[top_idx].x * img_w, landmarks[top_idx].y * img_h])
    p_bot = np.array([landmarks[bot_idx].x * img_w, landmarks[bot_idx].y * img_h])
    p_left = np.array([landmarks[left_idx].x * img_w, landmarks[left_idx].y * img_h])
    p_right = np.array([landmarks[right_idx].x * img_w, landmarks[right_idx].y * img_h])
    
    v = np.linalg.norm(p_top - p_bot)
    h = np.linalg.norm(p_left - p_right)
    
    return v / h if h > 0 else 0.0


def get_relative_iris_pos(landmarks, inner_idx, outer_idx, top_idx, bottom_idx, iris_idx):
    """
    Calculates normalized iris position (0.0 to 1.0) inside eye socket bounds.
    """
    inner_x = landmarks[inner_idx].x
    outer_x = landmarks[outer_idx].x
    top_y = landmarks[top_idx].y
    bottom_y = landmarks[bottom_idx].y
    iris_x = landmarks[iris_idx].x
    iris_y = landmarks[iris_idx].y

    width = abs(inner_x - outer_x) + 1e-6
    height = abs(bottom_y - top_y) + 1e-6

    min_x = min(inner_x, outer_x)
    rel_x = (iris_x - min_x) / width
    rel_y = (iris_y - top_y) / height

    return rel_x, rel_y
