"""
Mouse controller and gesture state manager.
"""
import pyautogui
import time

class MouseController:
    """
    Wrapper around PyAutoGUI for smooth, safe cursor movements and action execution.
    """
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.scroll_mode = False
        self.last_action_time = 0
        self.action_cooldown = 0.3  # Seconds between repeat actions

    def toggle_scroll_mode(self):
        self.scroll_mode = not self.scroll_mode
        return self.scroll_mode

    def move_cursor(self, target_x, target_y):
        if not self.scroll_mode:
            target_x = max(0, min(self.screen_w - 1, int(target_x)))
            target_y = max(0, min(self.screen_h - 1, int(target_y)))
            try:
                pyautogui.moveTo(target_x, target_y)
            except Exception:
                pass

    def perform_scroll(self, amount):
        if self.scroll_mode:
            try:
                pyautogui.scroll(amount)
            except Exception:
                pass

    def click(self, button='left'):
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            try:
                pyautogui.click(button=button)
                self.last_action_time = current_time
                return True
            except Exception:
                pass
        return False
