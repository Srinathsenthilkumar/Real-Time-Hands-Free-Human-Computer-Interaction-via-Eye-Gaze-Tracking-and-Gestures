"""
Main Execution Script for Real-Time Eye-Gaze & Facial Gesture HCI System.
Includes Interactive 5-Point Calibration & Anti-Shake Filtering.
"""
import cv2
import time
import argparse
import numpy as np
import config
from tracker import GazeTrackerEngine
from utils.controller import MouseController
from utils.visualization import HUDVisualizer

def run_app():
    parser = argparse.ArgumentParser(description="Real-Time Eye-Gaze HCI System")
    parser.add_argument("--camera", type=int, default=0, help="Camera Index (default: 0)")
    args = parser.parse_args()

    engine = GazeTrackerEngine()
    controller = MouseController(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with index {args.camera}.")
        return

    print("==================================================")
    print("   Advanced Eye-Gaze & Gesture HCI System")
    print("==================================================")
    print(f"Screen Resolution: {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
    print("Controls:")
    print("  - Move Eyes  -> Move Cursor")
    print("  - Left Wink  -> Left Click")
    print("  - Right Wink -> Right Click")
    print("  - Mouth Open -> Toggle Scroll Mode")
    print("  - Press 'c'  -> Launch 5-Point Calibration")
    print("  - Press 'q' or 'ESC' -> Exit System")
    print("==================================================")

    left_wink_counter = 0
    right_wink_counter = 0
    mouth_open_state_flag = False
    action_notification = ""
    action_notif_timer = 0

    # Calibration state
    is_calibrating = False
    calib_steps = ["TOP-LEFT", "TOP-RIGHT", "CENTER", "BOTTOM-LEFT", "BOTTOM-RIGHT"]
    calib_step_idx = 0
    calib_samples = []
    calib_step_start_time = 0
    calib_results = {}
    SAMPLES_PER_STEP_TIME = 1.2 # Seconds per target

    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARN] Camera frame capture failed.")
            break

        frame = cv2.flip(frame, 1)  # Mirror frame
        img_h, img_w, _ = frame.shape
        results = engine.process_frame(frame)

        ear_left, ear_right, mar = 0.0, 0.0, 0.0
        screen_pos = (0, 0)
        curr_time = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Extract metrics & gaze vectors
            ear_left, ear_right, mar, avg_rel_x, avg_rel_y = engine.extract_metrics(landmarks, img_w, img_h)

            # Check eye open/closed status
            is_left_closed = ear_left < config.EAR_THRESHOLD
            is_right_closed = ear_right < config.EAR_THRESHOLD

            # --- INTERACTIVE CALIBRATION MODE ---
            if is_calibrating:
                target_name = calib_steps[calib_step_idx]
                elapsed = curr_time - calib_step_start_time
                progress = min(1.0, elapsed / SAMPLES_PER_STEP_TIME)

                if not is_left_closed and not is_right_closed:
                    calib_samples.append((avg_rel_x, avg_rel_y))

                HUDVisualizer.draw_calibration_screen(
                    frame, target_name, progress, calib_step_idx, len(calib_steps)
                )

                if elapsed >= SAMPLES_PER_STEP_TIME:
                    if calib_samples:
                        avg_x = np.mean([s[0] for s in calib_samples])
                        avg_y = np.mean([s[1] for s in calib_samples])
                        calib_results[target_name] = (avg_x, avg_y)

                    calib_step_idx += 1
                    calib_samples = []
                    calib_step_start_time = curr_time

                    if calib_step_idx >= len(calib_steps):
                        is_calibrating = False
                        # Compute bounds
                        min_x = (calib_results["TOP-LEFT"][0] + calib_results["BOTTOM-LEFT"][0]) / 2.0
                        max_x = (calib_results["TOP-RIGHT"][0] + calib_results["BOTTOM-RIGHT"][0]) / 2.0
                        min_y = (calib_results["TOP-LEFT"][1] + calib_results["TOP-RIGHT"][1]) / 2.0
                        max_y = (calib_results["BOTTOM-LEFT"][1] + calib_results["BOTTOM-RIGHT"][1]) / 2.0
                        
                        engine.save_calibration(min_x, max_x, min_y, max_y)
                        action_notification = "CALIBRATION COMPLETE!"
                        action_notif_timer = curr_time

                cv2.imshow("Advanced Eye-Gaze HCI Engine", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q')]:
                    break
                continue

            # --- NORMAL TRACKING MODE ---
            smoothed_x, smoothed_y = engine.map_to_screen(
                avg_rel_x, avg_rel_y, config.SCREEN_WIDTH, config.SCREEN_HEIGHT, timestamp=curr_time
            )
            screen_pos = (int(smoothed_x), int(smoothed_y))

            # Left Wink Detection (Left Click)
            if is_left_closed and not is_right_closed:
                left_wink_counter += 1
            else:
                if left_wink_counter >= config.BLINK_CONSECUTIVE_FRAMES:
                    if controller.click('left'):
                        action_notification = "LEFT CLICK"
                        action_notif_timer = curr_time
                left_wink_counter = 0

            # Right Wink Detection (Right Click)
            if is_right_closed and not is_left_closed:
                right_wink_counter += 1
            else:
                if right_wink_counter >= config.BLINK_CONSECUTIVE_FRAMES:
                    if controller.click('right'):
                        action_notification = "RIGHT CLICK"
                        action_notif_timer = curr_time
                right_wink_counter = 0

            # Mouth Open (Toggle Scroll Mode)
            if mar > config.MAR_THRESHOLD:
                if not mouth_open_state_flag:
                    scroll_state = controller.toggle_scroll_mode()
                    action_notification = "SCROLL MODE TOGGLED"
                    action_notif_timer = curr_time
                    mouth_open_state_flag = True
            else:
                mouth_open_state_flag = False

            # Motion / Scroll Execution (FREEZE CURSOR WHEN EYES ARE CLOSED OR WINKING)
            if not is_left_closed and not is_right_closed:
                if controller.scroll_mode:
                    neutral_y = (engine.gaze_y_min + engine.gaze_y_max) / 2.0
                    if avg_rel_y < neutral_y - 0.05:
                        controller.perform_scroll(30)
                    elif avg_rel_y > neutral_y + 0.05:
                        controller.perform_scroll(-30)
                else:
                    controller.move_cursor(smoothed_x, smoothed_y)

            # Draw Landmarks
            HUDVisualizer.draw_landmarks(
                frame, landmarks,
                config.EYE_LEFT_LANDMARKS, config.EYE_RIGHT_LANDMARKS,
                [config.LIP_TOP, config.LIP_BOTTOM, config.LIP_LEFT, config.LIP_RIGHT],
                config.IRIS_LEFT_CENTER, config.IRIS_RIGHT_CENTER
            )

        # FPS calculation
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # Clear old action notifications after 1.5 seconds
        if curr_time - action_notif_timer > 1.5:
            action_notification = ""

        # Render HUD Telemetry
        HUDVisualizer.draw_telemetry(
            frame, fps, ear_left, ear_right, mar,
            screen_pos, controller.scroll_mode, action_notification
        )

        cv2.imshow("Advanced Eye-Gaze HCI Engine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break
        elif key == ord('c'):
            is_calibrating = True
            calib_step_idx = 0
            calib_samples = []
            calib_results = {}
            calib_step_start_time = curr_time
            print("[INFO] Starting 5-Point Dynamic Calibration Routine...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_app()
