# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:02:55 2025

@author: spallanzanig
"""
import cv2
import math
import init
from config import *

def select_DPI_callback(event, x, y, flags, param):
    global DPIselection_frame
    
    # Ensure DPIselection_frame is available before proceeding
    if DPIselection_frame is None:
        return

    frame_height, frame_width = DPIselection_frame.shape[:2]

    # Clamp coordinates to be within frame boundaries during interaction
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing: Reset points and flags
        init.drawing = True
        init.DPI_selected = False
        init.DPI_start = (x, y)
        init.DPI_end = (x, y)  # Initialize end point to start point
        init.DPI_geometry = None
        print(f"DPI Start: {init.DPI_start}")

    elif event == cv2.EVENT_MOUSEMOVE:
        if init.drawing:
            # Update end point while dragging
            init.DPI_end = (x, y)
            # Draw temporary rectangle on a copy for feedback
            temp_frame = DPIselection_frame.copy()
            cv2.rectangle(temp_frame, init.DPI_start, init.DPI_end, (0, 255, 0), 2)
            # Update window immediately
            cv2.imshow(init.DPI_window_name, temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        if init.drawing:
            init.drawing = False
            init.DPI_end = (x, y)
            init.DPI_selected = True
            print(f"DPI End: {init.DPI_end}")

            # Calculate final ROI (x, y, w, h), ensuring top-left start
            x1, y1 = init.DPI_start
            x2, y2 = init.DPI_end
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x1 - x2)
            h = abs(y1 - y2)

            # Basic validation: Ensure width and height are non-zero
            if w > 0 and h > 0:
                init.DPI_geometry = (x, y, w, h)
                print(f"DPI Defined (x,y,w,h): {init.DPI_geometry}")
                # Draw final rectangle on the copy
                temp_frame = DPIselection_frame.copy()
                cv2.rectangle(temp_frame, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)  # Red for final
                cv2.putText(temp_frame, "Press 'c' to Confirm", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(init.DPI_window_name, temp_frame)
            else:
                print("ROI selection invalid (width or height is zero). Please redraw.")
                init.DPI_selected = False  # Invalidate selection
                init.DPI_geometry = None
                # Show original frame again if selection was invalid
                cv2.imshow(init.DPI_window_name, DPIselection_frame)
                
def flush_key_buffer_DPI(delay_ms=1, max_loops=200):
    """
    Attempts to consume lingering key presses in the OpenCV waitKey buffer.
    """
    # print("Flushing keyboard buffer...") # Optional debug
    count = 0
    while count < max_loops:
        if cv2.waitKey(delay_ms) != -1:
            count = 0  # Reset count if a key was found
        else:
            count += 1  # Increment count if no key was found
    # print("Buffer flush attempt complete.") # Optional debug


def DPI_selection(cap):
    global DPIselection_frame
    
    print("\n----------------------------------------------------------------------")
    print("Do you want to input the resolution manually or through DPI ROI selection? \n(y for manual input, default is ROI selection)")
    maybe = input()
    if maybe == 'y':
        # Manual Resolution Input
        input_res = input("Enter the uniform resolution (e.g., 0.15) in mm/pixel: ")
        resolution = float(input_res)
        if 1 > resolution > 0 :
            print(f"\nManual Resolution Set: {resolution:} mm/pixel")
            return resolution
        else:
            print(f"\n Invalid Manual Resolution. Continuing with ROI selection.")
            return None
        
    # --- Set the DPI selection frame ---
    # cv2.CAP_PROP_POS_FRAMES is the property for the 0-based index of the frame to be decoded/captured next.    
    DPIsuccess = cap.set(cv2.CAP_PROP_POS_FRAMES, DPI_frame)
    
    if not DPIsuccess:
        print(f"Warning: Could not seek to frame {
              DPI_frame}. No DPI region of interest selection possible.")
        # Reset DPI_frame if seeking failed, though cap.read() might still start near the desired frame
        # Get the actual position it landed on
        return None
    
    # --- Read the first frame for ROI selection ---
    ret, DPIselection_frame = cap.read()
    key = 0
    if not ret:
        print(f"**ERROR**: Could not read the starting frame ({DPI_frame}).")
        cap.release()
        return None
    
    first_frame_copy = DPIselection_frame.copy()  # Keep an unmodified copy  
    
    # --- ROI Selection Phase ---
    cv2.namedWindow(init.DPI_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(init.DPI_window_name, select_DPI_callback)
    cv2.imshow(init.DPI_window_name, DPIselection_frame)  # Initial display
    
    
    print("\n--- DPI Selection ---")
    print("Draw a rectangle on the frame.")
    print("Press 'c' to confirm the selection and start processing.")
    print("Press 'r' to reset the selection and redraw.")
    print("Press 'q' to quit without processing.")
    
    # --- Flush keyboard buffer before entering selection loop ---
    flush_key_buffer_DPI()
    
    while True:
    
        key = cv2.waitKey(20) & 0xFF  # Wait for a short time
    
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Quit signal received during selection. Action cancelled by user.")
        elif key == ord('r'):
            print("Resetting DPI selection.")
            init.drawing = False
            init.DPI_selected = False
            init.DPI_start = (-1, -1)
            init.DPI_end = (-1, -1)
            init.DPI_geometry = None
            first_frame_copy = first_frame.copy()  # Restore original view
            # Show the restored frame
            cv2.imshow(init.DPI_window_name, first_frame_copy)
            flush_key_buffer_DPI()
        elif key == ord('c'):
            if init.DPI_selected and init.DPI_geometry:
                print("DPI Confirmed.")
                init.DPI = True
                # Extract final ROI coordinates for the main loop
                DPI_x, DPI_y, DPI_width, DPI_height = init.DPI_geometry
                break  # Exit ROI selection loop
            else:
                print("No valid DPI selected to confirm. Please draw a valid DPI ROI first.")
                flush_key_buffer_DPI()
    
    cv2.destroyWindow(init.DPI_window_name)  # Close the selection window
    
    
    print("\nInsert the length of the diagonal in mm in real life:")
    mm_diagonal = int(input())
    pixel_diagonal = math.sqrt(DPI_width**2 + DPI_height**2)
    resolution = mm_diagonal/pixel_diagonal
    
    print(f"\nfor {mm_diagonal} mm and {pixel_diagonal} pixels, the resolution is {resolution} mm per pixel.\n")
    return resolution