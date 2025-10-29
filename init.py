# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:14:59 2025

@author: spallanzanig
"""


# --- ROI Initialization ---
ROI = True                  # True to enable ROI cropping, False to process full frame
ROI_x = 0                   # Top-left x-coordinate of the ROI
ROI_y = 0                   # Top-left y-coordinate of the ROI
ROI_width = 0               # Width of the ROI
ROI_height = 0              # Height of the ROI
ROI_window_name = "Select ROI - Drag Mouse, Press 'c' to Confirm, 'r' to Reset, 'q' to Quit"

# --- Global variables for mouse callback ---
drawing = False         # True if mouse is pressed
ROI_selected = False    # True once the ROI is finalized by releasing mouse
ROI_start = (-1, -1)
ROI_end = (-1, -1)
ROI_geometry = None     # Will store (x, y, w, h) of the final ROI
b_end = (-1, -1)
b_start = (-1,-1)
point_selected = -1
max_length = -1

black = True
black_selected = False
black_start = (-1, -1)
black_end = (-1, -1)
black_geometry = None

# --- DPI ROI Initialization ---
DPI = False
DPI_selected = False    # True once the ROI for DPI is finalized by releasing mouse
DPI_start = (-1, -1)
DPI_end = (-1, -1)
DPI_geometry = None        
DPI_window_name = "Select DPI - Drag Mouse, Press 'c' to Confirm, 'r' to Reset, 'q' to Quit"


bf_window_name = "Select end point, Press 'c' to Confirm, 'r' to Reset, 'q' to Quit"
bi_window_name = "Select start point, Press 'c' to Confirm, 'r' to Reset, 'q' to Quit"

# --- Global variables initialization
start_frame = 0   # Default start frame number
last_frame = 0    # Counter to track frames since the last processed one
count = 0         # Counter for how many frames were actually processed

total_frames = 0
fps = 0
frame_width = 0
frame_height = 0
