"""
Multi-Monitor GUI wrapper for the TradingLinesMonitor. (tkinter version)
Version: 2.1 - Added Test Screenshot Feature

- Manages multiple independent monitoring sessions from a single interface.
- Each monitor can be configured with its own unique detection parameters.
- NEW: Can save test screenshots of the full region and the analyzed ROI for any monitor.
- Each monitor runs in its own thread.
- Global Telegram configuration.
"""

import os
import threading
import time
from collections import deque
from datetime import datetime
import queue
from io import BytesIO

# --- GUI and Automation Libraries ---
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import pyautogui
import keyboard
import cv2
import numpy as np
import requests

### --- Helper and Core Logic Classes --- ###


class TelegramNotifier:
    """Minimal Telegram Bot API client for sending messages and photos."""

    # (Your full TelegramNotifier class goes here, this is a placeholder for brevity)
    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}/{method}"

    def send_photo(self, image_bytes: bytes, caption: str | None = None) -> bool:
        if not self.is_configured():
            print("Telegram not configured. Photo not sent.")
            return False
        try:
            files = {"photo": ("alert.png", image_bytes, "image/png")}
            data = {"chat_id": self.chat_id}
            if caption:
                data["caption"] = caption
            resp = requests.post(
                self._api_url("sendPhoto"), data=data, files=files, timeout=20
            )
            return resp.ok
        except Exception as e:
            print(f"❌ Telegram send_photo failed: {e}")
            return False


class TradingLinesMonitor:
    """Core engine for monitoring a single chart region."""

    def __init__(self, name="Default", logger_callback=None):
        self.name = name
        self.logger = logger_callback if logger_callback else print
        self.region = None
        self.monitoring = False
        self.alert_given = False
        self.telegram = TelegramNotifier()
        self.blue_lower = np.array([98, 180, 150])
        self.blue_upper = np.array([115, 255, 255])
        self.red_lower1 = np.array([0, 160, 120])
        self.red_upper1 = np.array([8, 255, 255])
        self.red_lower2 = np.array([170, 160, 120])
        self.red_upper2 = np.array([179, 255, 255])
        self.right_portion = 0.25
        self.right_margin = 0.05
        self.line_detection_threshold = 2

    def log(self, *args):
        msg = " ".join(str(a) for a in args)
        self.logger(msg)

    # ------------------ NEW METHOD FOR SCREENSHOTS ------------------
    def save_test_screenshots(self):
        """Takes screenshots of the full region and ROI, and saves them to disk."""
        if not self.region:
            self.log("Cannot save screenshots, region is not set.")
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log("Capturing test screenshots...")

            # --- 1. Screenshot the full region ---
            full_region_pil = pyautogui.screenshot(region=self.region)
            full_region_filename = (
                f"TEST_FULL_{self.name.replace(' ', '_')}_{timestamp}.png"
            )
            full_region_pil.save(full_region_filename)
            self.log(f"Saved full region screenshot as '{full_region_filename}'")

            # --- 2. Screenshot the ROI ---
            # Convert Pillow image to OpenCV format to use existing ROI logic
            full_region_cv2 = cv2.cvtColor(np.array(full_region_pil), cv2.COLOR_RGB2BGR)
            roi_cv2 = self.get_monitoring_roi(full_region_cv2)
            roi_filename = f"TEST_ROI_{self.name.replace(' ', '_')}_{timestamp}.png"
            cv2.imwrite(roi_filename, roi_cv2)
            self.log(f"Saved ROI screenshot as '{roi_filename}'")
            return True

        except Exception as e:
            self.log(f"Error saving test screenshots: {e}")
            return False

    # ------------------ END OF NEW METHOD ------------------

    # The rest of your TradingLinesMonitor methods are unchanged.
    def setup_region(self):
        self.log("Region Selector Active... Press SPACE for corners, ESC to cancel.")
        corners = []
        for corner_name in ["TOP-LEFT", "BOTTOM-RIGHT"]:
            self.log(f"Move mouse to {corner_name} and press SPACE...")
            while True:
                if keyboard.is_pressed("space"):
                    corners.append(pyautogui.position())
                    time.sleep(0.3)
                    break
                if keyboard.is_pressed("esc"):
                    self.log("Region selection cancelled.")
                    return
                time.sleep(0.05)
        x = min(corners[0][0], corners[1][0])
        y = min(corners[0][1], corners[1][1])
        width = abs(corners[0][0] - corners[1][0])
        height = abs(corners[0][1] - corners[1][1])
        self.region = (x, y, width, height)
        self.log(f"Region set: x={x}, y={y}, w={width}, h={height}")

    def get_monitoring_roi(self, img):
        height, width = img.shape[:2]
        right_margin_pixels = int(width * self.right_margin)
        start_x = int(width * (1 - self.right_portion))
        end_x = width - right_margin_pixels
        return img[:, start_x:end_x]

    def detect_lines_with_hough(self, mask, color_name):
        lines = cv2.HoughLinesP(
            mask, 1, np.pi / 180, 15, minLineLength=20, maxLineGap=15
        )
        return lines if lines is not None else []

    def detect_line_colors(self):
        try:
            self.log("Processing image for detection...")
            screenshot = pyautogui.screenshot(region=self.region)
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            roi = self.get_monitoring_roi(img)
            hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv, self.red_lower1, self.red_upper1),
                cv2.inRange(hsv, self.red_lower2, self.red_upper2),
            )
            kernel = np.ones((5, 5), np.uint8)
            blue_mask = cv2.morphologyEx(
                cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel),
                cv2.MORPH_CLOSE,
                kernel,
            )
            red_mask = cv2.morphologyEx(
                cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel),
                cv2.MORPH_CLOSE,
                kernel,
            )
            blue_lines = self.detect_lines_with_hough(blue_mask, "Blue")
            red_lines = self.detect_lines_with_hough(red_mask, "Red")
            self.log(
                f"  - Blue: {len(blue_lines)} segments, Red: {len(red_lines)} segments"
            )
            return len(blue_lines), len(red_lines)
        except Exception as e:
            self.log(f"Detection error: {e}")
            return 0, 0

    def send_alert(self, message):
        full_message = f"ALERT [{self.name}]: {message}"
        try:
            with open("trading_alerts.log", "a", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {full_message}\n"
                )
        except Exception as e:
            self.log(f"Failed to write to trading_alerts.log: {e}")

        def _send_telegram_task():
            try:
                pil_img = pyautogui.screenshot(region=self.region)
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                self.telegram.send_photo(buf.getvalue(), caption=full_message)
            except Exception as e:
                self.log(f"Telegram photo send failed: {e}")

        if self.telegram and self.telegram.is_configured():
            threading.Thread(target=_send_telegram_task, daemon=True).start()

    def analyze_lines_state(self, blue_count, red_count):
        is_blue = blue_count >= self.line_detection_threshold
        is_red = red_count >= self.line_detection_threshold
        state = "different"
        if is_blue and not is_red:
            state = "same_blue"
        elif is_red and not is_blue:
            state = "same_red"
        elif not is_blue and not is_red:
            state = "insufficient_lines"
        if state in ("same_blue", "same_red"):
            if not self.alert_given:
                msg = (
                    f"All lines BLUE (B:{blue_count}, R:{red_count})"
                    if state == "same_blue"
                    else f"All lines RED (B:{blue_count}, R:{red_count})"
                )
                self.send_alert(msg)
                self.alert_given = True
        elif state == "different":
            if self.alert_given:
                self.alert_given = False
        return state

    def monitor_lines(self, check_interval=3):
        if not self.region:
            self.log("No region set.")
            return
        self.monitoring = True
        self.log("Starting monitoring.")
        while self.monitoring:
            try:
                blue, red = self.detect_line_colors()
                state = self.analyze_lines_state(blue, red)
                self.log(
                    f"State={state} | Blue={blue} | Red={red} | Alert Given={self.alert_given}"
                )
                time.sleep(check_interval)
            except Exception as e:
                self.log(f"Monitoring loop error: {e}")
                time.sleep(check_interval)
        self.log("Monitoring stopped.")


### --- Main GUI Application --- ###


class GuiAppTkinter:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Chart Trading Monitor v2.1")
        self.root.geometry("1100x700")
        self.ui_queue = queue.Queue()
        self.monitors, self.monitor_threads = {}, {}
        self.next_monitor_id = 0
        self.telegram_notifier = TelegramNotifier()
        self._create_widgets()
        self.root.after(100, self.process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Monitor Management Panel
        mgmt_frame = ttk.LabelFrame(
            main_frame, text="Monitor Control Panel", padding="10"
        )
        mgmt_frame.grid(row=0, column=0, sticky="ns", pady=5, padx=5)
        self.monitor_tree = ttk.Treeview(
            mgmt_frame, columns=("name", "status"), show="headings", height=10
        )
        self.monitor_tree.heading("name", text="Chart Name")
        self.monitor_tree.heading("status", text="Status")
        self.monitor_tree.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.monitor_tree.bind("<<TreeviewSelect>>", self.on_monitor_select)

        ttk.Button(
            mgmt_frame, text="Add New Monitor", command=self.handle_add_monitor
        ).grid(row=1, column=0, sticky="ew", pady=(10, 2))
        ttk.Button(
            mgmt_frame, text="Remove Selected", command=self.handle_remove_monitor
        ).grid(row=1, column=1, sticky="ew", pady=(10, 2))
        ttk.Button(
            mgmt_frame, text="Start / Stop", command=self.handle_toggle_monitor
        ).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(
            mgmt_frame, text="Configure Region", command=self.handle_configure_region
        ).grid(row=2, column=1, sticky="ew", pady=2)

        # --------------- NEW WIDGET FOR SCREENSHOTS ---------------
        self.screenshot_button = ttk.Button(
            mgmt_frame, text="Save Test Images", command=self.handle_save_test_images
        )
        self.screenshot_button.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(10, 2)
        )
        # ------------------ END OF NEW WIDGET ------------------

        self.params_frame = ttk.LabelFrame(
            main_frame, text="Selected Monitor Parameters", padding="10"
        )
        self.params_frame.grid(row=0, column=1, sticky="ns", pady=5, padx=5)
        self.portion_var = tk.StringVar()
        self.margin_var = tk.StringVar()
        self.thresh_var = tk.StringVar()
        ttk.Label(self.params_frame, text="Right Portion (0.1-1.0):").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Entry(self.params_frame, textvariable=self.portion_var).grid(
            row=0, column=1, sticky="ew", pady=2
        )
        ttk.Label(self.params_frame, text="Right Margin (0.0-0.2):").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Entry(self.params_frame, textvariable=self.margin_var).grid(
            row=1, column=1, sticky="ew", pady=2
        )
        ttk.Label(self.params_frame, text="Segment Threshold:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Entry(self.params_frame, textvariable=self.thresh_var).grid(
            row=2, column=1, sticky="ew", pady=2
        )
        self.apply_params_button = ttk.Button(
            self.params_frame,
            text="Apply Changes to Selected",
            command=self.handle_apply_monitor_params,
        )
        self.apply_params_button.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=10
        )
        self.set_widgets_state(self.params_frame, "disabled")

        # Global & Other Frames
        global_controls_frame = ttk.LabelFrame(
            main_frame, text="Global Controls", padding=10
        )
        global_controls_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=5, padx=5
        )
        ttk.Button(
            global_controls_frame, text="Start All", command=self.handle_start_all
        ).pack(side="left", padx=5)
        ttk.Button(
            global_controls_frame, text="Stop All", command=self.handle_stop_all
        ).pack(side="left", padx=5)
        tg_frame = ttk.LabelFrame(
            main_frame, text="Telegram (Global Settings)", padding=10
        )
        tg_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        self.tg_token_var = tk.StringVar()
        self.tg_chat_var = tk.StringVar()
        ttk.Label(tg_frame, text="Bot Token:").grid(row=0, column=0, sticky="w")
        ttk.Entry(tg_frame, textvariable=self.tg_token_var, width=50).grid(
            row=0, column=1, padx=5, sticky="ew"
        )
        ttk.Label(tg_frame, text="Chat ID:").grid(row=0, column=2, sticky="w")
        ttk.Entry(tg_frame, textvariable=self.tg_chat_var, width=20).grid(
            row=0, column=3, padx=5, sticky="ew"
        )
        ttk.Button(tg_frame, text="Apply TG", command=self.handle_apply_tg).grid(
            row=0, column=4, padx=10
        )
        tg_frame.grid_columnconfigure(1, weight=1)
        tg_frame.grid_columnconfigure(3, weight=1)
        log_frame = ttk.LabelFrame(main_frame, text="Global Log", padding=10)
        log_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5, padx=5)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, state="disabled", wrap="word")
        log_scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text["yscrollcommand"] = log_scrollbar.set
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

    def set_widgets_state(self, parent_widget, state):
        for child in parent_widget.winfo_children():
            child.configure(state=state)

    def on_monitor_select(self, event):
        selected_id = self.get_selected_monitor_id(silent=True)
        if not selected_id:
            self.set_widgets_state(self.params_frame, "disabled")
            self.portion_var.set("")
            self.margin_var.set("")
            self.thresh_var.set("")
            return
        self.set_widgets_state(self.params_frame, "normal")
        monitor = self.monitors[selected_id]
        self.portion_var.set(str(monitor.right_portion))
        self.margin_var.set(str(monitor.right_margin))
        self.thresh_var.set(str(monitor.line_detection_threshold))

    # ------------------ NEW HANDLER FOR THE BUTTON ------------------
    def handle_save_test_images(self):
        selected_id = self.get_selected_monitor_id()
        if not selected_id:
            return

        monitor = self.monitors[selected_id]
        if not monitor.region:
            messagebox.showerror(
                "Error",
                f"Region for '{monitor.name}' is not set. Please configure it first.",
                parent=self.root,
            )
            return

        # Run in a thread to prevent GUI freezing
        def task():
            success = monitor.save_test_screenshots()
            if success:
                messagebox.showinfo(
                    "Success",
                    f"Test images for '{monitor.name}' saved to the application folder.",
                    parent=self.root,
                )
            else:
                messagebox.showerror(
                    "Error",
                    f"Failed to save test images for '{monitor.name}'. Check the log for details.",
                    parent=self.root,
                )

        threading.Thread(target=task, daemon=True).start()

    # ------------------ END OF NEW HANDLER ------------------

    # --- Rest of the class is unchanged ---
    def handle_apply_monitor_params(self):
        selected_id = self.get_selected_monitor_id()
        if not selected_id:
            return
        monitor = self.monitors[selected_id]
        try:
            new_p, new_m, new_t = (
                float(self.portion_var.get()),
                float(self.margin_var.get()),
                int(self.thresh_var.get()),
            )
            if not (0.1 <= new_p <= 1.0 and 0.0 <= new_m <= 0.2 and new_t > 0):
                raise ValueError("Values out of range.")
            (
                monitor.right_portion,
                monitor.right_margin,
                monitor.line_detection_threshold,
            ) = (new_p, new_m, new_t)
            messagebox.showinfo(
                "Success", f"Parameters for '{monitor.name}' updated.", parent=self.root
            )
            self.enqueue_log(
                monitor.name,
                f"Params updated: Portion={new_p}, Margin={new_m}, Thresh={new_t}",
            )
        except (ValueError, TypeError) as e:
            messagebox.showerror(
                "Invalid Input",
                f"Please check your parameter values.\nError: {e}",
                parent=self.root,
            )
            self.on_monitor_select(None)

    def get_selected_monitor_id(self, silent=False):
        selection = self.monitor_tree.selection()
        if not selection:
            if not silent:
                messagebox.showwarning(
                    "Warning",
                    "Please select a monitor from the list first.",
                    parent=self.root,
                )
            return None
        return selection[0]

    def handle_configure_region(self):
        selected_id = self.get_selected_monitor_id()
        if not selected_id:
            return
        monitor = self.monitors[selected_id]
        messagebox.showinfo(
            "Configure Region",
            f"Configuring '{monitor.name}'. The main window will now hide.",
            parent=self.root,
        )
        self.root.withdraw()
        time.sleep(0.5)
        monitor.setup_region()
        self.root.deiconify()
        status = "Ready to Start" if monitor.region else "Region selection cancelled."
        self.ui_queue.put(("status_update", (selected_id, status)))

    def enqueue_log(self, monitor_name, text):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{monitor_name}] {text}\n"
        self.ui_queue.put(("log", line))

    def process_queue(self):
        try:
            while not self.ui_queue.empty():
                msg_type, data = self.ui_queue.get_nowait()
                if msg_type == "log":
                    self.log_text.config(state="normal")
                    self.log_text.insert("end", data)
                    self.log_text.see("end")
                    self.log_text.config(state="disabled")
                elif msg_type == "status_update":
                    monitor_id, status_text = data
                    if self.monitor_tree.exists(monitor_id):
                        self.monitor_tree.item(
                            monitor_id,
                            values=(self.monitors[monitor_id].name, status_text),
                        )
        finally:
            self.root.after(100, self.process_queue)

    def handle_add_monitor(self):
        name = simpledialog.askstring(
            "New Monitor", "Enter a unique name for this chart:", parent=self.root
        )
        if not name or any(m.name == name for m in self.monitors.values()):
            if name:
                messagebox.showerror(
                    "Error", "A monitor with this name already exists."
                )
                return
        monitor_id = f"m_{self.next_monitor_id}"
        self.next_monitor_id += 1
        logger_cb = lambda text: self.enqueue_log(name, text)
        new_monitor = TradingLinesMonitor(name=name, logger_callback=logger_cb)
        new_monitor.telegram = self.telegram_notifier
        self.monitors[monitor_id] = new_monitor
        self.monitor_tree.insert(
            "", "end", iid=monitor_id, values=(name, "Idle - Needs Region")
        )

    def handle_remove_monitor(self):
        selected_id = self.get_selected_monitor_id()
        if not selected_id:
            return
        self.handle_stop_monitor(selected_id)
        del self.monitors[selected_id]
        if selected_id in self.monitor_threads:
            del self.monitor_threads[selected_id]
        self.monitor_tree.delete(selected_id)
        self.on_monitor_select(None)

    def handle_toggle_monitor(self):
        selected_id = self.get_selected_monitor_id()
        if not selected_id:
            return
        if self.monitors[selected_id].monitoring:
            self.handle_stop_monitor(selected_id)
        else:
            self.handle_start_monitor(selected_id)

    def handle_start_monitor(self, monitor_id):
        monitor = self.monitors.get(monitor_id)
        if not monitor or monitor.monitoring:
            return
        if not monitor.region:
            messagebox.showerror(
                "Error", f"'{monitor.name}' has no region set.", parent=self.root
            )
            return
        self.ui_queue.put(("status_update", (monitor_id, "Running...")))
        monitor.monitoring = True
        thread = threading.Thread(target=monitor.monitor_lines, daemon=True)
        self.monitor_threads[monitor_id] = thread
        thread.start()

    def handle_stop_monitor(self, monitor_id):
        monitor = self.monitors.get(monitor_id)
        if not monitor or not monitor.monitoring:
            return
        monitor.monitoring = False
        self.ui_queue.put(("status_update", (monitor_id, "Stopped")))

    def handle_start_all(self):
        for mid in self.monitors:
            self.handle_start_monitor(mid)

    def handle_stop_all(self):
        for mid in self.monitors:
            self.handle_stop_monitor(mid)

    def handle_apply_tg(self):
        self.telegram_notifier.bot_token = self.tg_token_var.get() or None
        self.telegram_notifier.chat_id = self.tg_chat_var.get() or None
        msg = (
            "Global Telegram settings applied."
            if self.telegram_notifier.is_configured()
            else "Telegram settings cleared."
        )
        messagebox.showinfo("Telegram", msg, parent=self.root)

    def on_closing(self):
        self.handle_stop_all()
        time.sleep(0.5)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GuiAppTkinter(root)
    root.mainloop()
