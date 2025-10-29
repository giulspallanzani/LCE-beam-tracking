# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:04:02 2025

@author: spallanzanig

Python script using open-cv for analysis of cantilever beam actuation.
Goes through the following steps:
        - selection of a small ROI for resolution calculations (pixel/mm)
        - selection of a ROI (and eventual selection of start and end point of the LCE part of the beam) for faster processing
        - background removal 
        - focus filtering 
        - color filtering 
        - mask opening and closing for feature refinement
        - clustering
        - edge detection
        - segmentation along transversal using kdtree
        - nearest neighbour ordering
        - correction for the start point of the beam
        - correction on the length of the beam
        - calculation of relevant metrics (curvature, bounding rectangle, tip angle)
        - export of data and plots
        
NB. Modify config.py file before running
    Install requirements.txt before running
"""

# Beam Tracking

import cv2
import sys
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import init 
from config import *
from DPI_calculation import DPI_selection
from functions import *

results = []
temp = []

sys.path.append("D:\1. LCE Litho Matrix")

cap = initialize_video(video_path)

resolution = DPI_selection(cap)
if resolution is None:
    print("resolution was not found, setting it to 1")
    resolution = 1
    
# --- Reset Video Capture to Start Frame ---
print(f"Resetting video capture to start frame: {init.start_frame}\n")
cap.set(cv2.CAP_PROP_POS_FRAMES, init.start_frame)

ROI_selection(cap)

ENDPOINT_selection(cap)

STARTPOINT_selection(cap)

blackout_selection(cap)

frame_idx = init.start_frame

print(f"Starting processing from frame {
      frame_idx},processing every {frame_skip} frames....\n")
try:
    # Read the *next* frame (which will be start_frame on the first iteration)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # --- Check if this frame should be processed ---
        if init.last_frame % frame_skip == 0:
            init.count += 1
            # --- Crop Frame to ROI if enabled ---
            if init.ROI:
                # Slicing: [startY:endY, startX:endX]
                analysis_frame = frame[init.ROI_y: init.ROI_y +
                                       init.ROI_height, init.ROI_x: init.ROI_x + init.ROI_width]
            else:
                analysis_frame = frame  # Use the full frame if ROI is disabled
            print(f"\nProcessing Frame {frame_idx} corresponding to {frame_idx/init.fps} seconds...")
            curvature, v_proj, h_proj, angle, spline_pts, beam_0 = process_frame(analysis_frame, frame_idx, resolution)

            if frame_idx == init.start_frame and init.point_selected!=-1:
                init.point_selected = -1

            if curvature is not None:
                print(f"Frame {frame_idx}: Curvature={
                      curvature:.4f}, VerticalProj={v_proj:.2f}, Angle={angle:.2f}")
                results.append([frame_idx, frame_idx/init.fps, curvature, v_proj, h_proj, angle])

                temp.append([beam_0[0], beam_0[1]])
                b_0 = np.column_stack(temp)
                ## Optional debug plot for checking that beam_0 doesn't move too much compared to b_start
                # plt.imshow(cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB))
                # if frame_idx == init.start_frame and init.b_end[0] != -1:
                #    plt.plot(init.b_end[0], init.b_end[1], 'gx', linewidth=1)
                #    if (init.b_start[0]!=-1): plt.plot(init.b_start[0], init.b_start[1], 'gx', linewidth=1)
                # else: 
                #    bf_x = spline_pts[0]
                #    bf_y = spline_pts[1]
                #    plt.plot(bf_x[-1], bf_y[-1], 'rx', linewidth=1)
                # plt.plot(b_0[0], b_0[1], 'bx', linewidth=1)
                # plt.title(f"Start and End points at Frame {frame_idx}")
                # plt.show()
            else:
                print(f"**ERROR**: Could not read the current frame ({frame_idx}): trying with the following one.")
                init.last_frame -= 1 

        frame_idx += 1
        init.last_frame += 1

    df = pd.DataFrame(results, columns=[
                      "Frame", "Time (s)", "Curvature (1/mm)", "VerticalProjection (mm)","HorizontalProjection (mm)", "CumulativeAngle"])
    print("Processing complete.")

except KeyboardInterrupt:  # <--- Catch KeyboardInterrupt
    df = pd.DataFrame(results, columns=["Frame", "Time (s)", "Curvature (1/mm)", "VerticalProjection (mm)","HorizontalProjection (mm)", "CumulativeAngle"])
    print("\n\nProcessing interrupted by user (Ctrl+C).")

except Exception as e:  # <--- Catch other unexpected errors
    print(f"\nAn unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()  # Print the full traceback for debugging

finally:  # block for cleanup
    df = pd.DataFrame(results, columns=["Frame", "Time (s)", "Curvature (1/mm)", "VerticalProjection (mm)","HorizontalProjection (mm)", "CumulativeAngle"])
    cdt = datetime.today()
    print("\n--- Cleaning up ---")
    if 'cap' in locals() and cap.isOpened():  # Check if cap exists and is open
        cap.release()
        print("Video capture released.")

    cv2.destroyAllWindows()  # Close any OpenCV windows
    print("OpenCV windows closed.")

    print("Do you want to close all figures? y for yes")
    DEBUG_SHOW_PLOTS = str(input())
    if DEBUG_SHOW_PLOTS == 'y' and plt.get_fignums():  # Check if any matplotlib figures are open
        plt.close('all')  # Close all matplotlib figures
        print("Matplotlib figures closed.")

    # --- Save Results (even if interrupted, if results exist) ---
    print("\nDo you want to save the results? y for yes")
    save = str(input())
    if results and save == 'y':
        print("\nDo you want to plot the results? y for yes")
        plot = str(input())
        # Plotting results:
        if plot == 'y':
            # --- Plot 1: Beam Curvature over Frames ---
            beam_curv = df[['Time (s)', 'Curvature (1/mm)']]
            plt.figure(figsize=(10, 6))
            plt.plot(beam_curv['Time (s)'],
                     beam_curv['Curvature (1/mm)'], 'b-', marker='o')
            plt.xlabel("Time (s)")
            plt.ylabel("Beam Curvature (1/mm)")
            plt.title("Beam Curvature over Time")
            plt.grid(True)
            plt.tight_layout()
            if save == 'y':
                # Save the plot as a PNG file with 500 dpi resolution
                fig_name = os.path.splitext(
                    video_path)[0] + '_' + str(cdt.date()) + '_K.png'
                if init.count < (init.total_frames - init.start_frame) // frame_skip:  # Check if it was interrupted
                    fig_name = os.path.splitext(
                        fig_name)[0] + f'_until_{frame_idx}.png'
                n=1  
                while os.path.exists(fig_name):
                    if os.path.splitext(fig_name)[0].endswith(f'_{n-1}'):
                        fig_name = os.path.splitext(fig_name)[0][:-2]+ f'_{n}.png'
                    else: fig_name= os.path.splitext(fig_name)[0] + f'_{n}.png'
                    n = n+1
                
                plt.savefig(fig_name, dpi=500, bbox_inches="tight")
            plt.show()

            # --- Plot 2: Tip Angle over Frames ---
            angleoftip = df[['Time (s)', 'CumulativeAngle']]
            plt.figure(figsize=(10, 6))
            plt.plot(angleoftip['Time (s)'],
                     angleoftip['CumulativeAngle'], 'r-', marker='o')
            plt.xlabel("Time (s)")
            plt.ylabel("Tip Angle (deg)")
            plt.title("Tip Angle over Time")
            plt.grid(True)
            plt.tight_layout()
            if save == 'y':
                # Save the plot as a PNG file with 500 dpi resolution
                fig_name = os.path.splitext(
                    video_path)[0] + '_' + str(cdt.date()) + '_Angle.png'
                if init.count < (init.total_frames - init.start_frame) // frame_skip:  # Check if it was interrupted
                    fig_name = os.path.splitext(
                        fig_name)[0] + f'_until_{frame_idx}.png'
                n=1  
                while os.path.exists(fig_name):
                    if os.path.splitext(fig_name)[0].endswith(f'_{n-1}'):
                        fig_name = os.path.splitext(fig_name)[0][:-2]+ f'_{n}.png'
                    else: fig_name= os.path.splitext(fig_name)[0] + f'_{n}.png'
                    n = n+1
                
                plt.savefig(fig_name, dpi=500, bbox_inches="tight")
            plt.show()

            # --- Plot 3: Maximum Y Distance over Frames ---
            max_y = df[['Time (s)', 'VerticalProjection (mm)']]
            plt.figure(figsize=(10, 6))
            plt.plot(max_y['Time (s)'],
                     max_y['VerticalProjection (mm)'], 'g-', marker='o')
            plt.xlabel("Time (s)")
            plt.ylabel("Max Y Distance (mm)")
            plt.title("Max Y Distance over Time")
            plt.grid(True)
            plt.tight_layout()
            if save == 'y':
                # Save the plot as a PNG file with 500 dpi resolution
                fig_name = os.path.splitext(
                    video_path)[0] + '_' + str(cdt.date()) + '_Yproj.png'
                if init.count < (init.total_frames - init.start_frame) // frame_skip:  # Check if it was interrupted
                    fig_name = os.path.splitext(
                        fig_name)[0] + f'_until_{frame_idx}.png'
                n=1  
                while os.path.exists(fig_name):
                    if os.path.splitext(fig_name)[0].endswith(f'_{n-1}'):
                        fig_name = os.path.splitext(fig_name)[0][:-2]+ f'_{n}.png'
                    else: fig_name= os.path.splitext(fig_name)[0] + f'_{n}.png'
                    n = n+1
                plt.savefig(fig_name, dpi=500, bbox_inches="tight")
            plt.show()

            # --- Plot 4: Maximum X Distance over Frames ---
            max_x = df[['Time (s)', 'HorizontalProjection (mm)']]
            plt.figure(figsize=(10, 6))
            plt.plot(max_x['Time (s)'],
                     max_x['HorizontalProjection (mm)'], 'm-', marker='o')
            plt.xlabel("Time (s)")
            plt.ylabel("Max X Distance (mm)")
            plt.title("Max X Distance over Time")
            plt.grid(True)
            plt.tight_layout()
            if save == 'y':
                # Save the plot as a PNG file with 500 dpi resolution
                fig_name = os.path.splitext(
                    video_path)[0] + '_' + str(cdt.date()) + '_Xproj.png'
                if init.count < (init.total_frames - init.start_frame) // frame_skip:  # Check if it was interrupted
                    fig_name = os.path.splitext(
                        fig_name)[0] + f'_until_{frame_idx}.png'
                n=1  
                while os.path.exists(fig_name):
                    if os.path.splitext(fig_name)[0].endswith(f'_{n-1}'):
                        fig_name = os.path.splitext(fig_name)[0][:-2]+ f'_{n}.png'
                    else: fig_name= os.path.splitext(fig_name)[0] + f'_{n}.png'
                    n = n+1
                
                plt.savefig(fig_name, dpi=500, bbox_inches="tight")
            plt.show()

            print("Plots displayed")
        try:
            output_csv = os.path.splitext(
                video_path)[0] + '_' + str(cdt.date()) + '_beam_tracking.csv'
            if init.count < (init.total_frames - init.start_frame) // frame_skip:  # Check if it was interrupted
                output_csv = os.path.splitext(
                    output_csv)[0] + f'_until_{frame_idx}.csv'
            n=1  
            while os.path.exists(output_csv):
                if os.path.splitext(output_csv)[0].endswith(f'_{n-1}'):
                    output_csv = os.path.splitext(output_csv)[0][:-2]+ f'_{n}.csv'
                else: output_csv= os.path.splitext(output_csv)[0] + f'_{n}.csv'
                n = n+1
                
            description = f" Analysed {video_path}\nDimensions: {init.frame_width}x{init.frame_height}, Total frames: {init.total_frames}, FPS: {init.fps:.2f}\nStarting processing from frame {
                init.start_frame},processing every {frame_skip} frames until {frame_idx}\n Filtering settings: \n  focus = {focus_threshold}, window = {local_window_size} \n  hsv: upper = {upper} lower = {lower}\n Resolution = {resolution} mm per pixel\n"
            with open(output_csv, 'w') as file:
                file.write(description)
            df.to_csv(output_csv, mode='a', index=False)
            
            print(f"Results (up to frame {frame_idx}) saved to {output_csv}")
        except Exception as e:
            print(f"\nError saving results to CSV during cleanup: {e}")
    else:
        print("No results were generated to save.")

    print("Cleanup complete. Exiting.")
