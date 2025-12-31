import tkinter as tk
import pyautogui
from datetime import datetime
import os


class SnippingTool:
    def __init__(self, master, callback):
        self.master = master  # This is your ttkbootstrap root
        self.callback = callback

        # 1. Setup Overlay
        self.overlay = tk.Toplevel(self.master)
        self.overlay.withdraw()
        self.overlay.attributes("-alpha", 0.3)
        self.overlay.attributes("-fullscreen", True)
        self.overlay.attributes("-topmost", True)
        
        # 2. Setup Drawing Surface
        self.canvas = tk.Canvas(self.overlay, cursor="cross", bg="grey11")
        self.canvas.pack(fill="both", expand=True)
        
        # 3. Bind Mouse Events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.rect = None
        self.start_x = self.start_y = 0

    def start(self):
        self.master.withdraw() # Hide main app
        self.overlay.deiconify()

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red', width=2)

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        # Calculate region (ensuring integers)
        x = int(min(self.start_x, event.x))
        y = int(min(self.start_y, event.y))
        w = int(abs(event.x - self.start_x))
        h = int(abs(event.y - self.start_y))

        # Capture and ~~Save~~ send to callback
        if w > 0 and h > 0:
            # os.makedirs("snips", exist_ok=True)
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # path = os.path.join("snips", f"snip_{timestamp}.png")
            # pyautogui.screenshot(path, region=(x, y, w, h))
            screenshot_img = pyautogui.screenshot(region=(x,y,w,h))
            self.master.after(100, lambda: self.callback(screenshot_img))
        
        self.exit_tool()

    def exit_tool(self):
        self.overlay.destroy()
        self.master.deiconify() # Show main app again
        self.master.focus_force()
        self.master.update()


