import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ==========================================
# CONFIGURATION & THRESHOLDS
# ==========================================
# PyAutoGUI Safety & Timing setup
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# Detection Thresholds
EAR_THRESHOLD = 0.18           # Eye Aspect Ratio threshold for winks
MAR_THRESHOLD = 0.5            # Mouth Aspect Ratio threshold for opening mouth
BLINK_CONSECUTIVE_FRAMES = 3   # Number of consecutive frames needed to trigger a click
SMOOTHING_FACTOR = 0.15        # Exponential Moving Average smoothing (0.0 to 1.0)

# Gaze mapping boundaries (Relative iris position within the eye socket)
# Adjust these variables to calibrate the screen reach.
# A smaller range means you don't have to move your eyes as far.
GAZE_X_MIN = 0.35
GAZE_X_MAX = 0.65
GAZE_Y_MIN = 0.30
GAZE_Y_MAX = 0.70

# ==========================================
# LANDMARK INDICES (MediaPipe Face Mesh)
# ==========================================
# NOTE: Because we mirror the webcam frame, the user's right eye appears 
# on the left side of the screen, and the user's left eye on the right.
# Left eye on the screen (User's actual right eye)
EYE_LEFT_IDX = [33, 160, 158, 133, 153, 144]
# Right eye on the screen (User's actual left eye)
EYE_RIGHT_IDX = [362, 385, 387, 263, 373, 380]

# Lips for Mouth Aspect Ratio (MAR)
LIP_TOP_IDX = 13
LIP_BOTTOM_IDX = 14
LIP_LEFT_IDX = 78
LIP_RIGHT_IDX = 308

# Iris centers (available when refine_landmarks=True)
IRIS_LEFT_CENTER = 468   # Left eye on screen
IRIS_RIGHT_CENTER = 473  # Right eye on screen

# ==========================================
# CORE ALGORITHMS
# ==========================================

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Calculates the Eye Aspect Ratio (EAR) using Euclidean distance math.
    EAR = (V1 + V2) / (2 * H)
    Where V1, V2 are vertical distances between the eyelids, and H is the horizontal width.
    """
    p1 = np.array([landmarks[eye_indices[0]].x * img_w, landmarks[eye_indices[0]].y * img_h])
    p2 = np.array([landmarks[eye_indices[1]].x * img_w, landmarks[eye_indices[1]].y * img_h])
    p3 = np.array([landmarks[eye_indices[2]].x * img_w, landmarks[eye_indices[2]].y * img_h])
    p4 = np.array([landmarks[eye_indices[3]].x * img_w, landmarks[eye_indices[3]].y * img_h])
    p5 = np.array([landmarks[eye_indices[4]].x * img_w, landmarks[eye_indices[4]].y * img_h])
    p6 = np.array([landmarks[eye_indices[5]].x * img_w, landmarks[eye_indices[5]].y * img_h])
    
    # Euclidean distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
    return ear

def calculate_mar(landmarks, img_w, img_h):
    """
    Calculates the Mouth Aspect Ratio (MAR).
    MAR = Vertical Height / Horizontal Width
    """
    p_top = np.array([landmarks[LIP_TOP_IDX].x * img_w, landmarks[LIP_TOP_IDX].y * img_h])
    p_bot = np.array([landmarks[LIP_BOTTOM_IDX].x * img_w, landmarks[LIP_BOTTOM_IDX].y * img_h])
    p_left = np.array([landmarks[LIP_LEFT_IDX].x * img_w, landmarks[LIP_LEFT_IDX].y * img_h])
    p_right = np.array([landmarks[LIP_RIGHT_IDX].x * img_w, landmarks[LIP_RIGHT_IDX].y * img_h])
    
    v = np.linalg.norm(p_top - p_bot)
    h = np.linalg.norm(p_left - p_right)
    
    mar = v / h if h > 0 else 0
    return mar

def get_relative_iris_pos(landmarks, inner_idx, outer_idx, top_idx, bottom_idx, iris_center_idx):
    """
    Calculates the normalized position (0.0 to 1.0) of the iris within the eye bounding box.
    """
    inner_x = landmarks[inner_idx].x
    outer_x = landmarks[outer_idx].x
    top_y = landmarks[top_idx].y
    bottom_y = landmarks[bottom_idx].y
    iris_x = landmarks[iris_center_idx].x
    iris_y = landmarks[iris_center_idx].y
    
    width = abs(inner_x - outer_x) + 1e-6
    height = abs(bottom_y - top_y) + 1e-6
    
    min_x = min(inner_x, outer_x)
    
    rel_x = (iris_x - min_x) / width
    rel_y = (iris_y - top_y) / height
    
    return rel_x, rel_y

def main():
    # 1. System Setup & Configuration
    SCREEN_W, SCREEN_H = pyautogui.size()
    print(f"Detected screen resolution: {SCREEN_W}x{SCREEN_H}")
    print("Press 'ESC' or 'q' to exit.")
    
    # 2. MediaPipe Pipeline Initialization
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # State variables
    prev_screen_x, prev_screen_y = None, None
    left_wink_frames = 0
    right_wink_frames = 0
    mouth_open_state = False
    scroll_mode = False
    
    p_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from webcam. Exiting...")
            break
            
        # Mirror frame horizontally
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find facial landmarks
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- 3. Landmark & Metric Processing ---
            ear_left = calculate_ear(landmarks, EYE_LEFT_IDX, img_w, img_h)
            ear_right = calculate_ear(landmarks, EYE_RIGHT_IDX, img_w, img_h)
            mar = calculate_mar(landmarks, img_w, img_h)
            
            # --- 5. Gesture Control Mapping (Winks) ---
            # Distinguish a wink from a blink: one eye closed, the other open.
            is_left_closed = ear_left < EAR_THRESHOLD
            is_right_closed = ear_right < EAR_THRESHOLD
            
            if is_left_closed and not is_right_closed:
                left_wink_frames += 1
            else:
                if left_wink_frames >= BLINK_CONSECUTIVE_FRAMES:
                    pyautogui.click(button='left')
                    cv2.putText(frame, "LEFT CLICK", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                left_wink_frames = 0
                
            if is_right_closed and not is_left_closed:
                right_wink_frames += 1
            else:
                if right_wink_frames >= BLINK_CONSECUTIVE_FRAMES:
                    pyautogui.click(button='right')
                    cv2.putText(frame, "RIGHT CLICK", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                right_wink_frames = 0
                
            # --- Gesture Control Mapping (Scroll Mode Toggle) ---
            if mar > MAR_THRESHOLD:
                if not mouth_open_state:
                    scroll_mode = not scroll_mode
                    mouth_open_state = True
            else:
                mouth_open_state = False
                
            # --- Gaze Tracking ---
            # Left Eye parameters: inner=133, outer=33, top=159, bottom=145
            rel_x_left, rel_y_left = get_relative_iris_pos(landmarks, 133, 33, 159, 145, IRIS_LEFT_CENTER)
            # Right Eye parameters: inner=362, outer=263, top=386, bottom=374
            rel_x_right, rel_y_right = get_relative_iris_pos(landmarks, 362, 263, 386, 374, IRIS_RIGHT_CENTER)
            
            avg_rel_x = (rel_x_left + rel_x_right) / 2.0
            avg_rel_y = (rel_y_left + rel_y_right) / 2.0
            
            # --- 4. Cursor Smoothing & Motion Calibration ---
            # Interpolate relative gaze position to full screen resolution
            screen_x = np.interp(avg_rel_x, [GAZE_X_MIN, GAZE_X_MAX], [0, SCREEN_W])
            screen_y = np.interp(avg_rel_y, [GAZE_Y_MIN, GAZE_Y_MAX], [0, SCREEN_H])
            
            # Exponential Moving Average (EMA)
            if prev_screen_x is None:
                prev_screen_x, prev_screen_y = screen_x, screen_y
            else:
                screen_x = prev_screen_x + SMOOTHING_FACTOR * (screen_x - prev_screen_x)
                screen_y = prev_screen_y + SMOOTHING_FACTOR * (screen_y - prev_screen_y)
                
            prev_screen_x, prev_screen_y = screen_x, screen_y
            
            # Execute physical action safely
            try:
                if scroll_mode:
                    # Scroll based on vertical eye gaze
                    neutral_y = (GAZE_Y_MIN + GAZE_Y_MAX) / 2
                    if avg_rel_y < neutral_y - 0.05: # Looking up
                        pyautogui.scroll(30)
                    elif avg_rel_y > neutral_y + 0.05: # Looking down
                        pyautogui.scroll(-30)
                else:
                    pyautogui.moveTo(int(screen_x), int(screen_y))
            except pyautogui.FailSafeException:
                pass
                
            # --- 6. Visualization & On-Screen Display (OSD) ---
            # Draw Irises
            iris_l_pos = (int(landmarks[IRIS_LEFT_CENTER].x * img_w), int(landmarks[IRIS_LEFT_CENTER].y * img_h))
            iris_r_pos = (int(landmarks[IRIS_RIGHT_CENTER].x * img_w), int(landmarks[IRIS_RIGHT_CENTER].y * img_h))
            cv2.circle(frame, iris_l_pos, 3, (255, 0, 255), -1)
            cv2.circle(frame, iris_r_pos, 3, (255, 0, 255), -1)
            
            # Draw Eye Contours
            for idx in EYE_LEFT_IDX + EYE_RIGHT_IDX:
                pt = (int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h))
                cv2.circle(frame, pt, 1, (0, 255, 255), -1)
                
            # Draw Lips
            for idx in [LIP_TOP_IDX, LIP_BOTTOM_IDX, LIP_LEFT_IDX, LIP_RIGHT_IDX]:
                pt = (int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h))
                cv2.circle(frame, pt, 2, (0, 0, 255), -1)
                
            # Display OSD Telemetry
            mode_text = "MODE: SCROLL" if scroll_mode else "MODE: CURSOR"
            color = (0, 255, 255) if scroll_mode else (255, 255, 0)
            
            cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"EAR L: {ear_left:.2f}  R: {ear_right:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Screen Pos: ({int(screen_x)}, {int(screen_y)})", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # FPS Calculation & Display
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (img_w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Gaze Tracker & Gesture Control", frame)
        
        # 7. Code Quality & Standards (Clean Exit)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
