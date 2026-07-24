"""
Configuration file for Eye Gaze & Facial Gesture HCI System.
"""
import os
import pyautogui

# System / PyAutoGUI Settings
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# Display Specs (16-inch 16:10 FHD+ 1920x1200)
DISPLAY_DIAGONAL_CM = 40.64
DISPLAY_ASPECT_RATIO = "16:10"
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Landmark Thresholds
EAR_THRESHOLD = 0.18            # Eye Aspect Ratio threshold for wink/blink
MAR_THRESHOLD = 0.50            # Mouth Aspect Ratio threshold for mouth open
BLINK_CONSECUTIVE_FRAMES = 3    # Frames held for wink activation

# Filtering & Anti-Jitter Parameters (1-Euro Filter)
ONE_EURO_MIN_CUTOFF = 0.1       # Lower value = ultra-smooth stationary cursor (no shake)
ONE_EURO_BETA = 0.05            # Higher value = fast response on rapid glances (no lag)
DEADZONE_PIXELS = 5.0           # Ignore micro-movements smaller than 5 pixels

# Default Gaze Mapping Bounds (Calibrated for 16:10 Aspect Ratio)
# 16:10 has a taller vertical field of view, so Y bounds are tuned for comfortable top/bottom reaching
DEFAULT_GAZE_X_MIN = 0.36
DEFAULT_GAZE_X_MAX = 0.64
DEFAULT_GAZE_Y_MIN = 0.38
DEFAULT_GAZE_Y_MAX = 0.62

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

# MediaPipe FaceMesh Landmark Indices
# Left eye on mirrored screen (User's right eye)
EYE_LEFT_LANDMARKS = [33, 160, 158, 133, 153, 144]
# Right eye on mirrored screen (User's left eye)
EYE_RIGHT_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Lips indices
LIP_TOP = 13
LIP_BOTTOM = 14
LIP_LEFT = 78
LIP_RIGHT = 308

# Irises
IRIS_LEFT_CENTER = 468
IRIS_RIGHT_CENTER = 473

# Colors for Visualization (BGR)
COLOR_TEXT = (240, 240, 240)
COLOR_ACCENT = (255, 191, 0)
COLOR_SUCCESS = (0, 230, 115)
COLOR_DANGER = (71, 99, 255)
COLOR_IRIS = (255, 0, 255)
