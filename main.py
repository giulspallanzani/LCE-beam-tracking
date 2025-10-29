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
import os
import traceback
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import init 
from config import *
from DPI_calculation import DPI_selection
from functions import *

sys.path.append("D:\\1. LCE Litho Matrix")

## Settate le colonne del dataframe
DATAFRAME_COLUMNS = [
    "Frame", "Time (s)", "Curvature (1/mm)", "VerticalProjection (mm)",
    "HorizontalProjection (mm)", "CumulativeAngle"
]

## Funzione per generare un file name univoco
def gen_filename(base_path, suffix, extension, is_interrupted=False, last_frame=0):
    base_name = f"{os.path.splitext(base_path)[0]}_{datetime.today().date()}"
    
    if is_interrupted:
        base_name += f'_until_{last_frame}'
    
    final_path = f"{base_name}{suffix}{extension}"
    n = 1
    while os.path.exists(final_path):
        final_path = f"{base_name}_{n}{suffix}{extension}"
        n += 1
    return final_path

## Ho notato dei pezzi duplicati per generare i grafici, ti ho creato una funzione che prende in input il data frame e altri parametri per snellire il codice
def plot_and_save(df, video_path, is_interrupted=False, last_frame=0):
    plot_configs = [
        {'col': 'Curvature (1/mm)', 'title': 'Beam Curvature over Time', 'ylabel': 'Beam Curvature (1/mm)', 'color': 'b-', 'suffix': '_K'},
        {'col': 'CumulativeAngle', 'title': 'Tip Angle over Time', 'ylabel': 'Tip Angle (deg)', 'color': 'r-', 'suffix': '_Angle'},
        {'col': 'VerticalProjection (mm)', 'title': 'Max Y Distance over Time', 'ylabel': 'Max Y Distance (mm)', 'color': 'g-', 'suffix': '_Yproj'},
        {'col': 'HorizontalProjection (mm)', 'title': 'Max X Distance over Time', 'ylabel': 'Max X Distance (mm)', 'color': 'm-', 'suffix': '_Xproj'}
    ]

    for config in plot_configs:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Time (s)'], df[config['col']], config['color'], marker='o')
        plt.xlabel("Time (s)")
        plt.ylabel(config['ylabel'])
        plt.title(config['title'])
        plt.grid(True)
        plt.tight_layout()
        
        fig_name = gen_filename(video_path, config['suffix'], ".png", is_interrupted, last_frame)
        plt.savefig(fig_name, dpi=500, bbox_inches="tight")
        print(f"Plot saved to: {fig_name}")
        plt.show()

## Funzione entry point del file, così è più pulito il flusso
def main():
    cap = initialize_video(video_path)

    resolution = DPI_selection(cap)
    if resolution is None:
        print("Resolution was not found, setting it to 1")
        resolution = 1
    
    print(f"Resetting video capture to start frame: {init.start_frame}\n")
    cap.set(cv2.CAP_PROP_POS_FRAMES, init.start_frame)

    ROI_selection(cap)
    ENDPOINT_selection(cap)
    STARTPOINT_selection(cap)
    blackout_selection(cap)

    results = []
    frame_idx = init.start_frame
    is_interrupted = False
    
    print(f"Starting processing from frame {frame_idx}, processing every {frame_skip} frames....\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if init.last_frame % frame_skip == 0:
                init.count += 1
                
                analysis_frame = frame
                if init.ROI:
                    analysis_frame = frame[init.ROI_y: init.ROI_y + init.ROI_height, 
                                           init.ROI_x: init.ROI_x + init.ROI_width]
                
                print(f"\nProcessing Frame {frame_idx} corresponding to {frame_idx/init.fps:.2f} seconds...")
                
                curvature, v_proj, h_proj, angle, _, _ = process_frame(analysis_frame, frame_idx, resolution)

                if frame_idx == init.start_frame and init.point_selected != -1:
                    init.point_selected = -1

                if curvature is not None:
                    print(f"Frame {frame_idx}: Curvature={curvature:.4f}, VerticalProj={v_proj:.2f}, Angle={angle:.2f}")
                    results.append([frame_idx, frame_idx/init.fps, curvature, v_proj, h_proj, angle])
                else:
                    print(f"**ERROR**: Could not read the current frame ({frame_idx}): trying with the following one.")
                    init.last_frame -= 1 

            frame_idx += 1
            init.last_frame += 1

        print("Processing complete.")

    except KeyboardInterrupt:
        is_interrupted = True
        print("\n\nProcessing interrupted by user (Ctrl+C).")
    except Exception as e:
        is_interrupted = True
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
        
    finally:
        print("\n--- Cleaning up ---")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("Video capture released.")

        cv2.destroyAllWindows()
        print("OpenCV windows closed.")

        if plt.get_fignums():
            user_choice = input("Do you want to close all figures? (y/n): ").lower().strip()
            if user_choice == 'y':
                plt.close('all')
                print("Matplotlib figures closed.")
        
        if not results:
            print("No results were generated to save. Exiting.")
            return

        df = pd.DataFrame(results, columns=DATAFRAME_COLUMNS)
        
        save_choice = input("\nDo you want to save the results? (y/n): ").lower().strip()
        if save_choice == 'y':
            plot_choice = input("Do you want to plot the results? (y/n): ").lower().strip()
            
            if plot_choice == 'y':
                plot_and_save(df, video_path, is_interrupted, frame_idx)

            try:
                output_csv = gen_filename(
                    video_path, '_beam_tracking', '.csv', is_interrupted, frame_idx
                )
                
                description = (
                    f"Analysed {video_path}\n"
                    f"Dimensions: {init.frame_width}x{init.frame_height}, Total frames: {init.total_frames}, FPS: {init.fps:.2f}\n"
                    f"Starting processing from frame {init.start_frame}, processing every {frame_skip} frames until {frame_idx}\n"
                    f"Filtering settings: \n  focus = {focus_threshold}, window = {local_window_size}\n"
                    f"  hsv: upper = {upper} lower = {lower}\n"
                    f"Resolution = {resolution} mm per pixel\n"
                )
                
                with open(output_csv, 'w') as f:
                    f.write(description)
                
                df.to_csv(output_csv, mode='a', index=False)
                print(f"Results saved to {output_csv}")
            except Exception as e:
                print(f"\nError saving results to CSV: {e}")
        else:
            print("No results were generated to save.")

        print("Cleanup complete. Exiting.")


## Best practice per il main in python :)
if __name__ == "__main__":
    main()