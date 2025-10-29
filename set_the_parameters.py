# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:02:55 2025

@author: spallanzanig
"""

import cv2
import numpy as np
from init import *
from config import *

# --- Function to Parse Time String ---
def parse_time_to_seconds(time_str):
    """
    Parses a time string (HH:MM:SS.ms, MM:SS.ms, SS.ms) into total seconds.
    Returns None if the format is invalid.
    """
    parts = time_str.split(':')
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            return None
    except ValueError:
        return None
    
def hsv_average(frame):
    
    # Ensure frame is available before proceeding
    if frame is None:
        return
    h_mean = np.mean(frame[:,:,0])
    s_mean = np.mean(frame[:,:,1])
    v_mean = np.mean(frame[:,:,2])
    mean = np.array([h_mean, s_mean, v_mean]) 
    
    h_std = np.std(frame[:,:,0])
    s_std = np.std(frame[:,:,1])
    v_std = np.std(frame[:,:,2])
    std = np.array([h_std, s_std, v_std]) 
    print(f"Your average and std in hsv are {mean} - {std}")
    return mean, std

def hsv_range(frame):
    
    # Ensure frame is available before proceeding
    if frame is None:
        return
    
    average_values, stdDev = hsv_average(frame)
    max_out = [0]*3
    min_out = [0]*3

    for n in range(0,3,1):
        temp1 = np.int16(average_values[n]-stdDev[n])
        temp2 = np.int16(average_values[n]+stdDev[n])
        min_out[n]= max(temp1,1)
        max_out[n]= min(temp2,254)
        
    h_low = min_out[0]
    s_low = min_out[1]
    v_low = min_out[2]
    
    h_high = max_out[0]
    s_high = max_out[1]
    v_high = max_out[2]
    
    low_limit = np.array([h_low, s_low, v_low]) 
    high_limit = np.array([h_high, s_high, v_high])
    
    return low_limit, high_limit

def select_ROI_callback(event, x, y, flags, param):
    global drawing, ROI_start, ROI_end, ROI_selected, ROI_geometry, first_frame_copy
    
    # Ensure first_frame_copy is available before proceeding
    if first_frame_copy is None:
        return

    frame_height, frame_width = first_frame_copy.shape[:2]

    # Clamp coordinates to be within frame boundaries during interaction
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing: Reset points and flags
        drawing = True
        ROI_selected = False
        ROI_start = (x, y)
        ROI_end = (x, y)  # Initialize end point to start point
        ROI_geometry = None
        print(f"ROI Start: {ROI_start}")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update end point while dragging
            ROI_end = (x, y)
            # Draw temporary rectangle on a copy for feedback
            temp_frame = first_frame_copy.copy()
            cv2.rectangle(temp_frame, ROI_start, ROI_end, (0, 255, 0), 2)
            # Update window immediately
            cv2.imshow(ROI_window_name, temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            ROI_end = (x, y)
            ROI_selected = True
            print(f"ROI End: {ROI_end}")

            # Calculate final ROI (x, y, w, h), ensuring top-left start
            x1, y1 = ROI_start
            x2, y2 = ROI_end
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x1 - x2)
            h = abs(y1 - y2)

            # Basic validation: Ensure width and height are non-zero
            if w > 0 and h > 0:
                ROI_geometry = (x, y, w, h)
                print(f"ROI Defined (x,y,w,h): {ROI_geometry}")
                # Draw final rectangle on the copy
                temp_frame = first_frame_copy.copy()
                cv2.rectangle(temp_frame, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)  # Red for final
                cv2.putText(temp_frame, "Press 'c' to Confirm", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(ROI_window_name, temp_frame)
            else:
                print("ROI selection invalid (width or height is zero). Please redraw.")
                ROI_selected = False  # Invalidate selection
                ROI_geometry = None
                # Show original frame again if selection was invalid
                cv2.imshow(ROI_window_name, first_frame_copy)
                
def flush_key_buffer(delay_ms=1, max_loops=200):
    """
    Attempts to consume lingering key presses in the OpenCV waitKey buffer.
    """
    count = 0
    while count < max_loops:
        if cv2.waitKey(delay_ms) != -1:
            count = 0  # Reset count if a key was found
        else:
            count += 1  # Increment count if no key was found


# --- Video Capture Initialization ---
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    raise RuntimeError(f"Error: Could not open video file: {video_path}")  # Exit the script if the video can't be opened

# --- Get Video Properties ---
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video loaded: {video_path}")
print(f"Dimensions: {frame_width}x{frame_height}, Total frames: {
      total_frames}, FPS: {fps:.2f}")

# --- Calculate Start Frame from Time ---
if fps is None or fps <= 0:
    print("Warning: Could not determine video FPS. Cannot calculate start frame from time. Starting from frame 0.")
else:
    start_seconds = parse_time_to_seconds(start_time_str)
    if start_seconds is None:
        cap.release()
        raise RuntimeError(f"**ERROR**: Invalid start time format: '{start_time_str}'. Please use HH:MM:SS, MM:SS, or SS format.")
    elif start_seconds < 0:
        print("Warning: Start time is negative. Starting from frame 0.")
        start_seconds = 0.0

    # Calculate the target frame number
    calculated_start_frame = round(start_seconds * fps)

if calculated_start_frame:
    start_frame = calculated_start_frame

print(f"Selected frame is #{start_frame}. If you want to input a different value press x")
new_frame = input() 
if new_frame == 'x':
    start_frame = int(input("Input the new frame for hsv parameter selection: ")) 
    print(f"New frame is #{start_frame}.")
    
# --- Set the starting frame ---
# cv2.CAP_PROP_POS_FRAMES is the property for the 0-based index of the frame to be decoded/captured next.
success = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

if not success:
    print(f"Warning: Could not seek to frame {
          start_frame}. Starting from the beginning or nearest keyframe.")
    # Reset start_frame if seeking failed, though cap.read() might still start near the desired frame
    # Get the actual position it landed on
    start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))


# --- Read the first frame for ROI selection ---
ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError(f"**ERROR**: Could not read the selected frame ({start_frame}).")

first_frame_copy = first_frame.copy()  # Keep an unmodified copy

# --- ROI Selection Phase ---
cv2.namedWindow(ROI_window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(ROI_window_name, select_ROI_callback)
cv2.imshow(ROI_window_name, first_frame_copy)  # Initial display


print("\n--- ROI Selection ---")
print("Draw a rectangle on an area where LCE is.")
print("Press 'c' to confirm the selection and start processing.")
print("Press 'r' to reset the selection and redraw.")
print("Press 'q' to quit without processing.")

# --- Flush keyboard buffer before entering selection loop ---
flush_key_buffer()

while True:

    key = cv2.waitKey(20) & 0xFF  # Wait for a short time

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError("Quit signal received during ROI selection.")
    elif key == ord('r'):
        print("Resetting ROI selection.")
        drawing = False
        ROI_selected = False
        ROI_start = (-1, -1)
        ROI_end = (-1, -1)
        ROI_geometry = None
        first_frame_copy = first_frame.copy()  # Restore original view
        # Show the restored frame
        cv2.imshow(ROI_window_name, first_frame_copy)
        flush_key_buffer()
    elif key == ord('c'):
        if ROI_selected and ROI_geometry:
            print("ROI Confirmed.")
            ROI = True
            # Extract final ROI coordinates for the main loop
            ROI_x, ROI_y, ROI_width, ROI_height = ROI_geometry
            break  # Exit ROI selection loop
        else:
            print("No valid ROI selected to confirm. Please draw an ROI first.")
            flush_key_buffer()

cv2.destroyWindow(ROI_window_name)  # Close the selection window

param_frame = first_frame[ROI_y: ROI_y + ROI_height, ROI_x: ROI_x + ROI_width]

lower, upper = hsv_range(cv2.cvtColor(param_frame, cv2.COLOR_BGR2HSV))

print(f"Your new limits for color in hsv are {lower} - {upper}")
del lower, upper