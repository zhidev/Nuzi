import pygetwindow as gw
import pyautogui

def screenshot_by_partial_title(partial_name):
    # Get all windows that contain the partial_name string
    windows = gw.getWindowsWithTitle(partial_name)
    
    if windows:
        # Use the first match found
        target_win = windows[0]
        
        # Optional: Bring to front
        target_win.activate()
        
        # Define region: (left, top, width, height)
        region = (target_win.left, target_win.top, target_win.width, target_win.height)
        
        screenshot = pyautogui.screenshot(region=region)
        screenshot.save("app_capture.png")
        print(f"Captured: {target_win.title}")
    else:
        print(f"No window found containing '{partial_name}'")

screenshot_by_partial_title("Notepad") # Replace with the constant part of your app's title
