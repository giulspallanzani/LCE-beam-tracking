# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:38:24 2025

@author: spallanzanig
"""
import numpy as np

# --- Global values for filtering, clustering and segmentation ---

# Focus filtering
focus_threshold = 20   # Higher value -> stricter focus requirement
local_window_size = 17 # Size of neighborhood for varianc e calculation (must be odd)

# HSV filtering
lower = np.array([10, 8, 50])
upper = np.array([170, 80, 200])

# Kernel sizes for morphological operations
OPEN_KERNEL_SIZE = (9, 9)   #removal of small objects
CLOSE_KERNEL_SIZE = (3, 3)  #filling of holes inside figure 

# Clustering
DBSCAN_EPS = 150         # Max distance between points for neighborhood  
DBSCAN_MIN_SAMPLES = 100 # Min number of points to form a dense region

# Segmentation 
segment_step = 120          # Segments dimension       
min_points_per_segment = 50 # Minimum points needed in a segment to calculate centroid
search_radius = 500         # Radius where to look for points to be considered part of the same segment

# Centerline ordering and error handling
max_jump_distance = 500         # skip points which are very far apart
AVERAGING_RADIUS = 1.2 * 18     # average points locally within this radius

min_length_centerline = 5       # minimum length of the centerline for processing
err_margin = 50                 # maximum allowed error on b_start position and on length of the beam for successive frames

# Spline fitting of the points
k=2     # order of the spline fitting
s=1000  # smoothig factor

# Analysis 
n_tip_points = 7        # number of tip points used to calculate tip angle 
one_sided_fraction = 2  # inverse of the half-fraction of points to be used for curvature calculation (eg. if 3, 2/3 of the points will be used to calculate the curvature)

# --- Configuration ---
video_path = "path/from_main/to/myvideo.MOV"  # Replace this with the path of the video to analyse
start_time_str = "00:00:00.0"  # User Input: Start time string
frame_skip = 30     # Process 1 frame, skip frame_skip-1 frames

DPI_frame = 0 # Frame number for resolution calculation

