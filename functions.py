# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:33:09 2025

@author: spallanzanig
"""

import cv2
import math
import numpy as np
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree 
import init
from config import *


# --- Function clustering pixels to clean the mask ---
def keep_dbscan_cluster(binary_mask, eps, min_samples):
    """
    Keeps only the largest cluster found by DBSCAN based on proximity.
                                mask_closed, DBSCAN_EPS, DBSCAN_MIN_SAMPLES
    Arguments:
        binary_mask (np.ndarray): Input binary image (uint8, 0s and 255s).
        eps (float): maximum distance between two samples for one to be
                     considered as in the neighborhood of the other (DBSCAN param).
        min_samples (int): The number of samples in a neighborhood for a point
                           to be considered as a core point (DBSCAN param).

    Returns:
        np.ndarray: Binary mask containing only the pixels belonging to the
                    largest cluster found by DBSCAN.
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = np.uint8(binary_mask > 0) * 255

    # 1. Extract coordinates of non-zero pixels
    # np.where returns (row_indices, col_indices)
    points_yx = np.where(binary_mask > 0)
    # Convert to (N, 2) array of (y, x) or (row, col) coordinates
    points = np.column_stack(points_yx)

    if points.shape[0] < min_samples:  # Not enough points to form a cluster
        return np.zeros_like(binary_mask)

    # 2. Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_  # Cluster labels for each point, -1 is noise

    # 3. Identify the largest cluster (excluding noise label -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    # print(f"labels {unique_labels}\ncounts {counts}")
    if len(unique_labels) == 0:  # Only noise was found
        return np.zeros_like(binary_mask)

    largest_cluster_label = unique_labels[np.argmax(counts)]

    # 4. Create the final mask
    output_mask = np.zeros_like(binary_mask)
    # Get the indices (in the 'points' array) of points belonging to the largest cluster
    cluster_indices = np.where(labels == largest_cluster_label)[0]
    # Get the actual (row, col) coordinates for these points
    cluster_points_yx = points[cluster_indices]
    # Set the corresponding pixels in the output mask
    # Need to use tuple indexing for rows and columns separately
    output_mask[cluster_points_yx[:, 0], cluster_points_yx[:, 1]] = 255

    return output_mask

# --- Least Squares Circle Fitting Function ---
def least_squares_circle(points):
    """
    Fits a circle to a set of points using the algebraic distance method (Pratt/Hyper method variant).

    Args:
        points (np.ndarray): A NumPy array of shape (N, 2) representing N points (x, y).

    Returns:
        tuple: (xc, yc, radius) representing the center and radius of the fitted circle.
               Returns (None, None, None) if fitting fails (e.g., < 3 points, collinear points).
    """
    if points.shape[0] < 3:
        return None, None, None  # Needs at least 3 points

    x = points[:, 0]
    y = points[:, 1]

    # Coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # Calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # Linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc + Suv * vc = (Suuu + Suvv) / 2
    #    Suv * uc + Svv * vc = (Suuv + Svvv) / 2
    Suv = np.sum(u * v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solve the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0

    try:
        # Use pseudo-inverse for stability if A is singular (collinear points)
        center_reduced, resid, rank, s = np.linalg.lstsq(A, B, rcond=None)
        # Check if the system was solvable (rank deficient -> likely collinear)
        if rank < 2:
            print("Warning: Points might be collinear, circle fit may be unreliable.")
            return None, None, None

    except np.linalg.LinAlgError:
        return None, None, None

    uc, vc = center_reduced
    xc = x_m + uc
    yc = y_m + vc

    # Calculate the radius
    radius = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(x))

    return xc, yc, radius

# --- Function calculating the path length ---
def calculate_path_length_loop(x, y):
    """
    Calculates the total length of a path defined by an ordered list of 2D points.

    Args:
        ordered_points (list): A list of tuples, where each tuple is (x, y),
                               representing the ordered points along the path.

    Returns:
        float: The total length of the path. Returns 0.0 if fewer than 2 points.
    """
    if not x.any() or len(x) < 2:
        return 0.0
    if not y.any() or len(y) < 2:
        return 0.0

    total_length = 0.0
    segment_length = 0.0

    for i in range((len(x) - 1)):
        x1 = x[i]
        x2 = x[i+1]
        y1 = y[i]
        y2 = y[i+1]

        dx = x2 - x1
        dy = y2 - y1
        # Calculate Euclidean distance between p1 and p2
        segment_length = math.hypot(dx, dy)
        total_length += segment_length

    return total_length

# --- Function calculating curvature ---
def calculate_curvature_ls(points, resolution):
    """
    Calculates curvature based on least-squares circle fitting.
    Curvature = 1 / radius. Returns 0 if circle cannot be fit.
    """
    
    if len(points) < 15:
        return 0.0, None  # Need at least 15 points

    nr = int(len(points)/2)
    margin = (len(points)-1)/one_sided_fraction

    low = int(nr-margin)
    high = int(nr+margin)
    new_points = []
    for i in range(low, high):
        new_points.append(points[i])

    try:
        # Ensure points are in the correct format (Nx2 float)
        np_points = np.array(new_points, dtype=np.float64).reshape(-1, 2)
        
        xc, yc, radius = least_squares_circle(np_points)
        radius_mm = radius*resolution
        
        # Check for valid radius (avoid division by zero)
        if radius is not None and radius > 1e-6:
            curvature = 1.0 / radius_mm
            # Return curvature and circle parameters for drawing
            return curvature, ((int(round(xc)), int(round(yc))), int(round(radius)))
        else:
            print("Fitting failed or radius too small/invalid")
            return 0.0, None  
    except Exception as e:
        print(f"Warning: calculate_curvature_ls failed: {e}")
        return 0.0, None

# --- Focus mask Function ---
def create_focus_mask(frame, threshold, window_size):
    """
    Creates a binary mask identifying in-focus regions based on Laplacian variance.

    Calculates the variance of the Laplacian operator within a local window
    around each pixel. Regions with high variance (indicating sharp edges)
    are considered in focus.

    Args:
        frame (np.ndarray): Input image (BGR or Grayscale).
        threshold (float): Variance threshold. Regions with variance above this
                           value are considered in focus.
        window_size (int, optional): Size of the square window for local variance
                                     calculation. Must be odd. Default is 15.

    Returns:
        tuple: (focus_mask_uint8, variance_map)
            - focus_mask_uint8 (np.ndarray): Binary mask (uint8) where 255 indicates
                                             in-focus regions and 0 indicates out-of-focus.
            - variance_map (np.ndarray): The calculated variance map (float64). Useful
                                         for tuning the threshold. Returns None if input
                                         frame is invalid.
    """
    if not isinstance(frame, np.ndarray):
        print("Error: Input frame is not a valid NumPy array.")
        return None, None

    if window_size % 2 == 0:
        window_size += 1
        print(f"Warning: window_size must be odd, adjusted to {window_size}")

    # Convert to Grayscale if necessary
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(frame.shape) == 2:
        gray = frame  # Already grayscale
    else:
        print(f"Error: Input frame has unexpected shape: {frame.shape}")
        return None, None

    # Calculate Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate local variance using mean filters (efficient sliding window)
    # Variance(X) = E[X^2] - (E[X])^2
    # --> calculate E[X^2]
    laplacian_sq = laplacian**2
    # Use boxFilter (mean filter) for efficiency. borderType handles edges.
    mean_lap_sq = cv2.boxFilter(laplacian_sq, -1, (window_size, window_size),
                                normalize=True, borderType=cv2.BORDER_REPLICATE)

    # --> calculate (E[X])^2
    mean_lap = cv2.boxFilter(laplacian, -1, (window_size, window_size),
                             normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_lap_pow2 = mean_lap**2

    # Variance = E[X^2] - (E[X])^2
    variance_map = mean_lap_sq - mean_lap_pow2
    # Ensure variance is non-negative (handle potential floating point inaccuracies)
    variance_map[variance_map < 0] = 0

    # Threshold the variance map
    _, focus_mask = cv2.threshold(
        variance_map, threshold, 255, cv2.THRESH_BINARY)

    # Convert final mask to uint8
    focus_mask_uint8 = np.uint8(focus_mask)

    return focus_mask_uint8, variance_map

# --- Mouse callback Functions ---
# -1- ROI callback
def select_ROI_callback(event, x, y, flags, param):
    global ROIselection_frame
    
    # Ensure first_frame_copy is available before proceeding
    if ROIselection_frame is None:
        return
    
    frame_height, frame_width = ROIselection_frame.shape[:2]

    # Clamp coordinates to be within frame boundaries during interaction
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing: Reset points and flags
        init.drawing = True
        init.ROI_selected = False
        init.ROI_start = (x, y)
        init.ROI_end = (x, y)  # Initialize end point to start point
        init.ROI_geometry = None
        print(f"ROI Start: {init.ROI_start}")

    elif event == cv2.EVENT_MOUSEMOVE:
        if init.drawing:
            # Update end point while dragging
            init.ROI_end = (x, y)
            # Draw temporary rectangle on a copy for feedback
            temp_frame = ROIselection_frame.copy()
            cv2.rectangle(temp_frame, init.ROI_start, init.ROI_end, (0, 255, 0), 2)
            # Update window immediately
            cv2.imshow(init.ROI_window_name, temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        if init.drawing:
            init.drawing = False
            init.ROI_end = (x, y)
            init.ROI_selected = True
            print(f"ROI End: {init.ROI_end}")

            # Calculate final ROI (x, y, w, h), ensuring top-left start
            x1, y1 = init.ROI_start
            x2, y2 = init.ROI_end
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x1 - x2)
            h = abs(y1 - y2)

            # Basic validation: Ensure width and height are non-zero
            if w > 0 and h > 0:
                init.ROI_geometry = (x, y, w, h)
                print(f"ROI Defined (x,y,w,h): {init.ROI_geometry}")
                # Draw final rectangle on the copy
                temp_frame = ROIselection_frame.copy()
                cv2.rectangle(temp_frame, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)  # Red for final
                cv2.putText(temp_frame, "Press 'c' to Confirm", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(init.ROI_window_name, temp_frame)
            else:
                print("ROI selection invalid (width or height is zero). Please redraw.")
                init.ROI_selected = False  # Invalidate selection
                init.ROI_geometry = None
                # Show original frame again if selection was invalid
                cv2.imshow(init.ROI_window_name, ROIselection_frame)

# -2- end or start point callback
def select_point_callback(event, x, y, flags, param):
    global Pselection_frame

    # Ensure first_frame_copy is available before proceeding
    if Pselection_frame is None:
        return

    frame_height, frame_width = Pselection_frame.shape[:2]

    # Clamp coordinates to be within frame boundaries during interaction
    x = max(0, min(x, init.frame_width - 1))
    y = max(0, min(y, init.frame_height - 1))

    if (init.point_selected == -1):  # select end point
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing: Reset points and flags
            init.drawing = True
            init.point_selected = -1
            init.b_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if init.drawing:
                # Update end point while dragging
                init.b_end = (x, y)
                # Draw temporary circle on a copy for feedback
                temp_frame = Pselection_frame.copy()
                cv2.circle(temp_frame, init.b_end, 30, (255, 0, 0), 2)
                # Update window immediately
                cv2.imshow(init.bf_window_name, temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            if init.drawing:
                init.drawing = False
                init.b_end = (x, y)
                init.point_selected = 1
                if init.ROI:
                   b_end_ROI = (x + init.ROI_start[0], y + init.ROI_start[1])
                   print(f"End point selected: {b_end_ROI}")
                else:
                    print(f"End point selected: {init.b_end}")

                # Draw final point on the copy
                temp_frame = Pselection_frame.copy()
                cv2.circle(temp_frame, init.b_end, 5,
                           (0, 255, 0), 2)  # Red for final
                cv2.putText(temp_frame, "Press 'c' to Confirm", (x, y - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(init.bf_window_name, temp_frame)
    elif (init.point_selected == 0):  # select start point
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing: Reset points and flags
            init.drawing = True
            init.point_selected = 0
            init.b_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if init.drawing:
                # Update end point while dragging
                init.b_start = (x, y)
                # Draw temporary circle on a copy for feedback
                temp_frame = Pselection_frame.copy()
                cv2.circle(temp_frame, init.b_start, 30, (255, 0, 0), 2)
                # Update window immediately
                cv2.imshow(init.bi_window_name, temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            if init.drawing:
                init.drawing = False
                init.b_start = (x, y)
                init.point_selected = 2
                if init.ROI:
                    b_start_ROI = (x + init.ROI_start[0], y + init.ROI_start[1])
                    print(f"Start point selected: {b_start_ROI}")
                else:
                    print(f"Start point selected: {init.b_start}")

                # Draw final point on the copy
                temp_frame = Pselection_frame.copy()
                cv2.circle(temp_frame, init.b_start, 5,
                           (0, 255, 0), 2)  # Red for final
                cv2.putText(temp_frame, "Press 'c' to Confirm", (x, y - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(init.bi_window_name, temp_frame)
                
# -3- Blackout ROI selection callback
def select_blackout_callback(event, x, y, flags, param):
    global blackout_frame
    
    # Ensure first_frame_copy is available before proceeding
    if blackout_frame is None:
        return
    
    frame_height, frame_width = blackout_frame.shape[:2]

    # Clamp coordinates to be within frame boundaries during interaction
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing: Reset points and flags
        init.drawing = True
        init.black_selected = False
        init.black_start = (x, y)
        init.black_end = (x, y)  # Initialize end point to start point
        init.black_geometry = None
        print(f"ROI Start: {init.black_start}")

    elif event == cv2.EVENT_MOUSEMOVE:
        if init.drawing:
            # Update end point while dragging
            init.black_end = (x, y)
            # Draw temporary rectangle on a copy for feedback
            temp_frame = blackout_frame.copy()
            cv2.rectangle(temp_frame, init.black_start, init.black_end, (0, 255, 0), 2)
            # Update window immediately
            cv2.imshow(init.ROI_window_name, temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        if init.drawing:
            init.drawing = False
            init.black_end = (x, y)
            init.black_selected = True
            print(f"ROI End: {init.black_end}")

            # Calculate final ROI (x, y, w, h), ensuring top-left start
            x1, y1 = init.black_start
            x2, y2 = init.black_end
            a = min(x1, x2)
            b = min(y1, y2)
            c = abs(x1 - x2)
            d = abs(y1 - y2)

            # Basic validation: Ensure width and height are non-zero
            if c > 0 and d > 0:
                init.black_geometry = (a, b, c, d)
                print(f"ROI for blackout Defined (x,y,w,h): {init.black_geometry}")
                # Draw final rectangle on the copy
                temp_frame = blackout_frame.copy()
                cv2.rectangle(temp_frame, (a, b), (a + c, b + d),
                              (0, 0, 255), 2)  # Red for final
                cv2.putText(temp_frame, "Press 'c' to Confirm", (a, b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(init.ROI_window_name, temp_frame)
            else:
                print("ROI selection invalid (width or height is zero). Please redraw.")
                init.black_selected = False  # Invalidate selection
                init.black_geometry = None
                # Show original frame again if selection was invalid
                cv2.imshow(init.ROI_window_name, blackout_frame)
                
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

# --- Function to Flush Keyboard Buffer ---
def flush_key_buffer(delay_ms=1, max_loops=200):
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
    #print("Buffer flush attempt complete.") # Optional debug

# --- Function to segment the beam ---
def segment_by_traversal(edge_points): 
    """
    Segments a single, continuous contour by locally traversing the points, 
    grouping them sequentially, and finding the centroid of each group. 
    This is highly stable for high-curvature lines.

    Args:
        edge_points: Nx2 array of (x, y) coordinates belonging to a single contour.
        segment_step: Number of sequential points to group into one segment.
        search_radius: Maximum pixel distance to search for the next sequential point.
        min_points_per_segment: Minimum count needed for a valid segment centroid.

    Returns:
        List of (x, y) segment centroids as NumPy arrays, ordered along the contour's path.
    """
    # Cast points to float and check size
    points_float = edge_points.astype(float)
    N = points_float.shape[0]
    if N < min_points_per_segment:
       return []
    
    # KDTree for fast search and boolean mask for tracking usage
    kdtree = KDTree(points_float)
    used_indices = np.zeros(N, dtype=bool) # Tracks which points have been added to a segment
    current_segment_points = []
    segment_centroids = []
    
    
    # Select starting point (e.g., b_start or the leftmost point)
    b_start = init.b_start
    if b_start[0] != -1:
        # If b_start is defined, use it as the starting reference
        reference_coords = np.array([b_start[0], b_start[1]])
        distances_to_start = distance.cdist(points_float, [reference_coords])
        start_index = np.argmin(distances_to_start)
    else:
        # Fallback to the min x if b_start is not defined
        start_index = np.argmin(points_float[:, 0])
    
    current_index = start_index
    
    # Iterative Traversal and Segmentation
    while True:
        
        current_point_coords = points_float[current_index]
        current_segment_points.append(tuple(current_point_coords))
        used_indices[current_index] = True
        
        if len(current_segment_points) >= segment_step:
            centroid_arr = np.mean(current_segment_points, axis=0)
            segment_centroids.append(centroid_arr.astype(int)) 
            current_segment_points = []
            
        # Find the next sequential point using KDTree
        distances, neighbor_indices = kdtree.query(current_point_coords, k=min(N, 1500)) 
        next_index = -1
        
       # Iterate through neighbors (already sorted by distance)
        for dist, idx in zip(distances, neighbor_indices):
            
            # Check: Is the point within the search radius?
            if dist > search_radius:
                print(f"point was too far ({dist}), looking for next")
                break
            
            # Check: Is the point unused and not the current point itself?
            if not used_indices[idx] and idx != current_index:
                next_index = idx
                break
        
        if next_index != -1:
            # Advance the traversal
            current_index = next_index
        else:
            # Traversal stopped (gap found or all nearby points used).
            print("Gap found: all points are further than search radius.")
            # If the segment isn't full, average the remaining points before breaking.
            if len(current_segment_points) >= min_points_per_segment:
                centroid_arr = np.mean(current_segment_points, axis=0)
                segment_centroids.append(centroid_arr.astype(int))
            break # Stop traversal for this contour
            
    return segment_centroids

# --- Function to Find the Beam in the current frame and carry out analysis ---
def process_frame(frame, frame_idx, resolution):
    
    max_length = init.max_length 
    point_selected = init.point_selected
    b_end = init.b_end 
    b_start = init.b_start
    
    # --- Background removal ---
    background = cv2.GaussianBlur(frame, (255,255), 0)
    flattened_image = cv2.subtract(frame, background, dtype=cv2.CV_8U) + 128
    
    # --- Focus filtering ---
    focus_mask, variance_map = create_focus_mask(
        flattened_image, focus_threshold, local_window_size)

    if focus_mask is not None:
        
        focused_regions = cv2.bitwise_and(frame, frame, mask=focus_mask)
        ## Plots for debug, showing the original frame, the laplacian variance,
        ## the focus mask and the masked image
        ## (comment if debugging is not needed)
        
        # plt.figure(figsize=(18, 6))
        # plt.subplot(1, 4, 1)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.title(f'Original Frame {frame_idx}')
        # plt.axis('off')

        # plt.subplot(1, 4, 2)
        # im = plt.imshow(variance_map, cmap='viridis')
        # plt.title(f'Laplacian Variance Map\n(Window={local_window_size})')
        # plt.colorbar(im, fraction=0.046, pad=0.04)  # Adjust colorbar size
        # plt.axis('off')

        # plt.subplot(1, 4, 3)
        # plt.imshow(focus_mask, cmap='gray')
        # plt.title(f'Focus Mask\n(Threshold={focus_threshold})')
        # plt.axis('off')

        # plt.subplot(1, 4, 4)
        # plt.imshow(cv2.cvtColor(focused_regions, cv2.COLOR_BGR2RGB))
        # plt.title('Focused Regions Only')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
    else:
        print("**ERROR**:Could not generate focus mask.")
    
    # --- Mask out the blackout ROI if requested ---
    if init.black:
        a,b,c,d = init.black_geometry
        focused_regions[b:b+d, a:a+c] = 0
    
    # --- HSV Filtering ---
    hsv = cv2.cvtColor(focused_regions, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    ## Image shown for debug purpuses, comment if not needed:
    # plt.imshow(mask, cmap='gray')
    # plt.title(f"Focus+color mask {frame_idx}")
    # plt.show()
    
    # --- Noise removal ---
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPEN_KERNEL_SIZE)
    kernel_closed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL_SIZE)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_closed)
    
    ## Images shown for debug purpuses, comment if not needed:
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask_opened, cmap='gray')
    # plt.title("Opened Mask")
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask_closed, cmap='gray')
    # plt.title("Closed Mask")
    # plt.show()
    
    # --- Clustering ---
    clean_mask = keep_dbscan_cluster(mask_closed, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    ## Image shown for debug purpuses, comment if not needed
    # plt.imshow(clean_mask, cmap='gray')
    # plt.title("Clean Mask")
    # plt.show()
    
    # --- Edge Detection ---
    edges = cv2.Canny(clean_mask, 50, 250)
    ## Image shown for debug purpuses, comment if not needed
    # plt.imshow(edges, cmap='gray')
    # plt.title(f"Edges Mask {frame_idx}")
    # plt.show()

    edge_points = np.argwhere(edges > 0)
    if edge_points.shape[0] == 0:
        print("**ERROR**: No edge points found. Adjust Canny thresholds.")
        return None, None, None, None, None, None

    edge_points = edge_points[:, ::-1]
    output_image = frame.copy()
    other_image = frame.copy()
    
    x, y, w, h = cv2.boundingRect(edge_points)
    
    # Segment by Transversal and List Segment Center Points (Centroids)
    segment_centroids = []
    segment_centroids = segment_by_traversal(edge_points)

    if not segment_centroids:
        print("**ERROR**: Could not calculate centroids for any segment. Check segmentation or edge points.")
        return None, None, None, None, None, None

    # Order Center Points (Nearest Neighbor)
    ordered_centerline = []
    if len(segment_centroids) > 1:
        points_np = np.array(segment_centroids)
        num_points = len(points_np)
        used_indices = np.zeros(num_points, dtype=bool)
        
        # Check if b_start is the default (-1, -1)
        if b_start[0] != -1:
            # If b_start is defined, use it as the starting reference
            reference_coords = np.array([b_start[0], b_start[1]])
        else:
            # Fallback to the bounding box corner (x, y) if b_start is not defined
            reference_coords = np.array([x, y])
            
        distances_to_start = distance.cdist(points_np, [reference_coords])
        current_index = np.argmin(distances_to_start)
        
        beam_start = points_np[current_index]
        ordered_centerline.append(tuple(beam_start))
        used_indices[current_index] = True
        num_ordered = 1
        
        while num_ordered < num_points:
            last_point = points_np[current_index]
            
            # Calculate distances from the last point to all *unused* points
            distances = distance.cdist([last_point], points_np[~used_indices])

            if distances.size > 0:
                # Find the index of the minimum distance within the subset of unused points
                unused_points = points_np[~used_indices]
                original_indices = np.where(~used_indices)[0]
                min_distance = distances[0].min()
        
                # --- Critical Check ---
                if min_distance > max_jump_distance:
                    # Found a gap or reached the end of the main structure. Stop tracing.
                    # This prevents jumping to a distant, leftover point.
                    print(f"**WARNING**: Trace interrupted. Gap found (Distance: {min_distance:.2f}).")
                    break
                # Find all unused points within the AVERAGING_RADIUS (since the closest is within it)
                close_indices_in_subset = np.where(distances[0] <= AVERAGING_RADIUS)[0]
            
                # --- LOCAL AVERAGING LOGIC ---
                if close_indices_in_subset.size > 1:
                    # Found a cluster of 2 or more close points. Average them.
                    points_to_average = unused_points[close_indices_in_subset]
                    smoothed_centroid = np.mean(points_to_average, axis=0)
                    next_point = smoothed_centroid.astype(int)
                    
                    # Mark ALL points used for averaging as 'used'
                    indices_to_remove = original_indices[close_indices_in_subset]
                    used_indices[indices_to_remove] = True
                    
                    # Update current_index for the NEXT iteration: Find the index of the one among
                    # the original points that is closest to the smoothed centroid.
                    distances_to_new_centroid = distance.cdist([next_point], points_np[indices_to_remove])
                    current_index = indices_to_remove[np.argmin(distances_to_new_centroid)]
                    print(f"Averaging Nearest Neighbor...")
                else:
                    # Only one point was found. Find the corresponding original index.
                    min_dist_idx_in_subset = np.argmin(distances[0])
                    best_next_index = original_indices[min_dist_idx_in_subset]
                    next_point = points_np[best_next_index]
                    used_indices[best_next_index] = True
                    current_index = best_next_index
                
                # Add the point to the centerline
                ordered_centerline.append(tuple(next_point))
                num_ordered += 1
            else:
                break  # No unused points left
                
   
    # Refine Centerline to start from nearby b_start if selected
    if (b_start[0] != -1):
        if len(ordered_centerline) > 0:
            start_point_coords = np.array([b_start[0], b_start[1]])
            distances_to_start = distance.cdist(ordered_centerline, [start_point_coords])
            index = np.argmin(distances_to_start)
            if index == len(ordered_centerline)-1:
                # probably the ordering is flipped and we need to revert it
                ordered_centerline.reverse()
                distances_to_start = distance.cdist(ordered_centerline, [start_point_coords])
                index = np.argmin(distances_to_start)
            if index != 0:
                for erase in range(index):
                    ordered_centerline.pop(0)
        else:
            print("***ERROR***: centerline has no length")
            return None, None, None, None, None, None

    # Centerline length check
    if len(ordered_centerline) > min_length_centerline:
        # Draw lines connecting consecutive ordered points
        for i in range(1, len(ordered_centerline)):
            pt1 = ordered_centerline[i-1]
            pt2 = ordered_centerline[i]
            # Basic check to avoid drawing huge lines if ordering fails
            dist_sq = (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2
            # if first frame, checks whether the point is closer than the selected beam end to the previous point
            if (point_selected != -1) and (b_end[0] != -1):
                max_dist_sq = (pt1[0]-b_end[0])**2 + (pt1[1]-b_end[1])**2
                if (dist_sq >= max_dist_sq):
                    pt2 = b_end
                    for j in range(i, len(ordered_centerline)):
                        outlier = ordered_centerline[i]
                        ordered_centerline.pop(i)
                    ordered_centerline.append(b_end)
                    if len(ordered_centerline) < min_length_centerline:
                        print(f"**ERROR**: Not enough points in the centerline {
                              len(ordered_centerline)} < {min_length_centerline}")
                        return None, None, None, None, None, None
                    break             
    else:
        print(f"**ERROR**: Not enough points in the centerline {
              len(ordered_centerline)} < {min_length_centerline}")
        return None, None, None, None, None, None

    x_i, y_i = tuple(map(np.array, zip(*ordered_centerline)))
    if len(x_i) < min_length_centerline:
        print(f"**ERROR**: Not enough points in the x_i array {len(x_i)}< {min_length_centerline}")
        return None, None, None, None, None, None
    try:
        # Fit ordered_centerline points (k+1 points are needed, at least, where k is the spline degree (k=2 --> 3 points))
        tck, u = splprep([x_i, y_i], k = k, s = s)
        u_new = np.linspace(0, 1, 50)
        x_smooth, y_smooth = splev(u_new, tck)
    except ValueError as e:
        # Handle the error, often due to insufficient or problematic input points
        print(f"**ERROR**: Spline fitting failed. Not enough valid points or points are coincident.")
        print(f"Details: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        # Catch any OTHER unexpected error (e.g., Runtime, System, etc.)
        print(f"**CRITICAL ERROR**: Spline fitting failed unexpectedly.")
        print(f"Details: {e}")
        return None, None, None, None, None, None
    
    # Plot points, fitting (and, on first frame, end_point if selected)
    fig, ax = plt.subplots()
    plt.title(f'uncorrected {frame_idx}')
    ax.plot(x_i, y_i, 'ro')
    ax.plot(x_i[0], y_i[0], 'go')
    ax.plot(x_i[-1], y_i[-1], 'bo')
    ax.plot(x_smooth, y_smooth, 'rx')
    if b_end[0] != -1:
        ax.plot(b_end[0], b_end[1], 'b*')
    if b_start[0] != -1:
        ax.plot(b_start[0], b_start[1], 'g*')
    plt.show()

    length = calculate_path_length_loop(x_smooth, y_smooth)

    if init.max_length == -1:
        init.max_length = length
    elif init.max_length != -1:
        flag = False
        print(f"max length is {init.max_length}")
        print(f"Path length (loop): {length}")
        # reset initial to the new length of the array of points
        initial = len(x_smooth)
        count = 0
        # checks that the length of the beam is not very different from that in the first frame
        for i in range((initial-1), 0, -1):
            if (length < (init.max_length - 4*err_margin)):
                
                #debug image
                # plt.imshow(cv2.cvtColor(other_image, cv2.COLOR_BGR2RGB))
                # plt.plot(x_smooth[-1], y_smooth[-1], 'rx', linewidth=1)
                # plt.plot(x_smooth[0], y_smooth[0], 'bx', linewidth=1)
                # plt.title(f"Start and End points of TOO SHORT LINE at Frame {frame_idx}")
                # plt.show()
                
                print("**ERROR**: detected too short of a line")
                return None, None, None, None, None, None
            elif (length > (max_length + 2*err_margin)):
                almost_length = calculate_path_length_loop(x_smooth[0:-2], y_smooth[0:-2])
                if (almost_length > (max_length - 2*err_margin)):
                    flag = True
                    count += 1
                    # remove the last value of the array
                    x_smooth = np.delete(x_smooth, i)
                    y_smooth = np.delete(y_smooth, i)
                else:
                    break
            else: 
                break
            # recalculate the length
            length = calculate_path_length_loop(x_smooth, y_smooth)
            if (count > (initial-i)):
                break  # to accomodate for the change in size of x_smooth
        if flag:
            # Plot points, fitting (and, on first frame, end_point if selected)
            fig, ax = plt.subplots()
            ax.plot(x_smooth, y_smooth, 'rx')
            plt.title(f'corrected {frame_idx}')
            if point_selected == 1:
                ax.plot(b_end[0], b_end[1], 'bx')
            elif point_selected == 2:
                if (b_end[0]!=-1):
                    ax.plot(b_end[0], b_end[1], 'bx')
                ax.plot(b_start[0], b_start[1], 'gx')
            else :
                ax.plot(x_smooth[0], y_smooth[0], 'gx')
                ax.plot(x_smooth[-1], y_smooth[-1], 'bx')
                ax.plot(b_start[0], b_start[1], 'm*')
            plt.show()

    # Update bouding rectangle to correct for length
    points = np.array(list(zip(x_smooth, y_smooth)))
    points = points.astype(np.int32)
    x, y, w, h = cv2.boundingRect(points)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    # Calculate projections based on resolution
    h_mm = h*resolution
    w_mm = w*resolution

    # Calculate curvature + draw circle
    vis_frame = frame.copy()
    curvature, curvature_circle = calculate_curvature_ls(
        list(zip(x_smooth, y_smooth)), resolution)
    
    # --- Display Results ---

    if curvature_circle:  # Curvature circle
        cv2.circle(
            vis_frame, curvature_circle[0], curvature_circle[1], (0, 255, 255), 1)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {frame_idx} - Analysis Frame with Overlays')
        # Add text info to plot
        text_y = 20
        plt.text(10, text_y, f"Curv: {
                 curvature:.4f}", color='white', fontsize=10, backgroundcolor='black')
        plt.axis('off')
        plt.show(block=True)  # Show blocking for the final plot per frame
    
    # Bounding Rectangle
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Bounding rectangle')
    plt.axis('off')
    plt.show()

    plt.tight_layout()
    plt.show()
    
    # Spline fit
    plt.imshow(cv2.cvtColor(other_image, cv2.COLOR_BGR2RGB))
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2)
    plt.title(f"Beam Spline Fit at Frame {frame_idx}")
    plt.show()


    # Calculate tip angle based on last n_tip_points points
    dy_tip = (y_smooth[-1] - y_smooth[-n_tip_points]) * resolution
    dx_tip = (x_smooth[-1] - x_smooth[-n_tip_points]) * resolution
    angle_rad = np.arctan2(dx_tip, -dy_tip)
    angle_deg = np.rad2deg(angle_rad)

    beam_start = (x_smooth[0], y_smooth[0])
    
    return curvature, h_mm, w_mm, angle_deg, (x_smooth, y_smooth), beam_start

def initialize_video(video):
    # --- Video Capture Initialization ---
    cap = cv2.VideoCapture(video)
    # Check if video opened successfully
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file: {video}")  # Exit the script if the video can't be opened
    # --- Get Video Properties ---
    init.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    init.fps = cap.get(cv2.CAP_PROP_FPS)
    init.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    init.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    print(f"Video loaded: {video}") 
    print(f"Dimensions: {init.frame_width}x{init.frame_height}, Total frames: {
          init.total_frames}, FPS: {init.fps:.2f}")
    
    # --- Calculate Start Frame from Time ---
    if init.fps is None or init.fps <= 0:
        print("Warning: Could not determine video FPS. Cannot calculate start frame from time. Starting from frame 0.")
    else:
        start_seconds = parse_time_to_seconds(start_time_str)
        if start_seconds is None:
            print(f"**ERROR**: Invalid start time format: '{
                  start_time_str}'. Please use HH:MM:SS, MM:SS, or SS format.")
            start_seconds = 0.0
        elif start_seconds < 0:
            print("Warning: Start time is negative. Starting from frame 0.") 
            start_seconds = 0.0
        # Calculate the target frame number
        calculated_start_frame = round(start_seconds * init.fps)
    
        # Validate calculated start_frame
        if calculated_start_frame >= init.total_frames:
            print(f"**ERROR**: Calculated start frame ({calculated_start_frame} from time {
                  start_time_str}) is beyond the total frames ({init.total_frames}). Starting from frame 0.")
            calculated_start_frame = 0
        elif calculated_start_frame < 0:  
            calculated_start_frame = 0
            
        init.start_frame = calculated_start_frame  # Use the calculated frame
        print(f"Parsed start time '{start_time_str}' to {
              start_seconds:.2f} seconds.")
        print(f"Calculated start frame: {init.start_frame} (based on FPS: {init.fps:.2f})") 
    
    # --- Set the starting frame ---
    # cv2.CAP_PROP_POS_FRAMES is the property for the 0-based index of the frame to be decoded/captured next.
    success = cap.set(cv2.CAP_PROP_POS_FRAMES, init.start_frame) 
    if not success:
        print(f"Warning: Could not seek to frame {
              init.start_frame}. Starting from the beginning or nearest keyframe.")
        # Reset start_frame if seeking failed, though cap.read() might still start near the desired frame
        # Get the actual position it landed on
        init.start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    return cap

def blackout_selection(cap):
    global blackout_frame
    
    print("\nDo you want to blackout parts of the frame that would disturb the analysis? y for yes")
    maybe = input()
    if not maybe == 'y': 
        init.black = False
        return
    
    # --- Read the first frame for ROI selection ---
    ret, frame = cap.read()
    key = 0
    if not ret:
        cap.release()
        raise RuntimeError("**ERROR**: Could not read the starting frame.")
        
    if init.ROI:
        # Slicing: [startY:endY, startX:endX]
        blackout_frame = frame[init.ROI_y: init.ROI_y +
                               init.ROI_height, init.ROI_x: init.ROI_x + init.ROI_width]
    else:
        blackout_frame = frame  # Use the full frame if ROI is disabled
    
    first_frame_copy = blackout_frame.copy()  # Keep an unmodified copy
    # --- ROI Selection Phase ---
    cv2.namedWindow(init.ROI_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(init.ROI_window_name, select_blackout_callback)
    cv2.imshow(init.ROI_window_name, blackout_frame)  # Initial display
    
    print("\n--- blackout ROI Selection ---")
    print("Draw a rectangle on the frame.")
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
            raise RuntimeError("Quit signal received during ROI selection. ROI selection cancelled by user.")
        elif key == ord('r'):
            print("Resetting ROI selection.")
            init.drawing = False
            init.black_selected = False
            init.black_start = (-1, -1)
            init.black_end = (-1, -1)
            init.black_geometry = None
            first_frame_copy = blackout_frame.copy()  # Restore original view
            # Show the restored frame
            cv2.imshow(init.ROI_window_name, first_frame_copy)
            flush_key_buffer()
        elif key == ord('c'):
            if init.black_selected and init.black_geometry:
                print("Blackout ROI Confirmed.")
                init.black = True
                break  # Exit ROI selection loop
            else:
                print("No valid ROI selected to confirm. Please draw an ROI first.")
                flush_key_buffer()
    
    cv2.destroyWindow(init.ROI_window_name)  # Close the selection window
    return

def ROI_selection(cap):
    global ROIselection_frame
    
    # --- Read the first frame for ROI selection ---
    ret, ROIselection_frame = cap.read()
    key = 0
    if not ret:
        cap.release()
        raise RuntimeError("**ERROR**: Could not read the starting frame.")
        
    first_frame = ROIselection_frame.copy()  # Keep an unmodified copy
    
    # --- ROI Selection Phase ---
    cv2.namedWindow(init.ROI_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(init.ROI_window_name, select_ROI_callback)
    cv2.imshow(init.ROI_window_name, ROIselection_frame)  # Initial display
    
    print("\n--- ROI Selection ---")
    print("Draw a rectangle on the frame.")
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
            raise RuntimeError("Quit signal received during ROI selection. ROI selection cancelled by user.")
        elif key == ord('r'):
            print("Resetting ROI selection.")
            init.drawing = False
            init.ROI_selected = False
            init.ROI_start = (-1, -1)
            init.ROI_end = (-1, -1)
            init.ROI_geometry = None
            first_frame_copy = first_frame.copy()  # Restore original view
            # Show the restored frame
            cv2.imshow(init.ROI_window_name, first_frame_copy)
            flush_key_buffer()
        elif key == ord('c'):
            if init.ROI_selected and init.ROI_geometry:
                print("ROI Confirmed.")
                init.ROI = True
                # Extract final ROI coordinates for the main loop
                init.ROI_x, init.ROI_y, init.ROI_width, init.ROI_height = init.ROI_geometry
                break  # Exit ROI selection loop
            else:
                print("No valid ROI selected to confirm. Please draw an ROI first.")
                flush_key_buffer()
    
    cv2.destroyWindow(init.ROI_window_name)  # Close the selection window
    return

def ENDPOINT_selection(cap):
    global ROIselection_frame, Pselection_frame
    # --- End Point Selection Phase ---
    print("Do you also want to select the end point of the LCE? y for yes.")
    end_selection = str(input())
    # If the user wants to select the end point open another window
    if end_selection == 'y':
        Pselection_frame = ROIselection_frame.copy()  # Restore original view
        if init.ROI:
            # Slicing: [startY:endY, startX:endX]
            Pselection_frame = Pselection_frame[init.ROI_y: init.ROI_y +
                                                init.ROI_height, init.ROI_x: init.ROI_x + init.ROI_width]
        
        first_frame = Pselection_frame.copy()  # Keep an unmodified copy
        
        # --- Beam Endpoint Selection Phase ---
        cv2.namedWindow(init.bf_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(init.bf_window_name, select_point_callback)
        cv2.imshow(init.bf_window_name, Pselection_frame)  # Initial display
    
        print("\n--- End point Selection ---")
        print("Pick a point on the frame.")
        print("Press 'c' to confirm the selection and start processing.")
        print("Press 'r' to reset the selection.")
        print("Press 'q' to quit without processing.")
    
        # --- Flush keyboard buffer before entering selection loop ---
        flush_key_buffer()
        key = 0
        while True:
    
            key = cv2.waitKey(20) & 0xFF  # Wait for a short time
    
            if key == ord('q'):
                cap.release()
                init.drawing = False
                init.point_selected = -1
                cv2.destroyAllWindows()
                raise RuntimeError("Quit signal received during end point selection. Selection cancelled by user.")
            elif key == ord('r'):
                print("Resetting end point selection.")
                init.drawing = False
                init.point_selected = -1
                first_frame_copy = first_frame.copy()  # Restore original view
                # Show the restored frame
                cv2.imshow(init.bf_window_name, first_frame_copy)
                flush_key_buffer()
            elif key == ord('c'):
                if init.point_selected==1:
                    print("End point Confirmed.")
                    break  # Exit point selection loop
                else:
                    print("No valid point selected to confirm. Please select a point first.")
                    init.drawing = False
                    init.point_selected = -1
                    flush_key_buffer()
        cv2.destroyWindow(init.bf_window_name)  # Close the selection window
    return
    
def STARTPOINT_selection(cap):
    global ROIselection_frame, Pselection_frame    
    # --- Start Point Selection Phase ---
    print("Do you also want to select the start point of the LCE? y for yes.")
    start_selection = str(input())
    # If the user wants to select the start point open another window
    if start_selection == 'y':
        init.point_selected = 0
        Pselection_frame = ROIselection_frame.copy()  # Restore original view
        if init.ROI:
            # Slicing: [startY:endY, startX:endX]
            Pselection_frame = Pselection_frame[init.ROI_y: init.ROI_y +
                                                init.ROI_height, init.ROI_x: init.ROI_x + init.ROI_width]
        
        first_frame = Pselection_frame.copy()  # Keep an unmodified copy
        
        # --- Beam Startpoint Selection Phase ---
        cv2.namedWindow(init.bi_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(init.bi_window_name, select_point_callback)
        cv2.imshow(init.bi_window_name, Pselection_frame)  # Initial display
    
        print("\n--- Start point Selection ---")
        print("Pick a point on the frame.")
        print("Press 'c' to confirm the selection and start processing.")
        print("Press 'r' to reset the selection.")
        print("Press 'q' to quit without processing.")
    
        # --- Flush keyboard buffer before entering selection loop ---
        flush_key_buffer()
        key = 0
        while True:
    
            key = cv2.waitKey(20) & 0xFF  # Wait for a short time
    
            if key == ord('q'):
                cap.release()
                init.drawing = False
                init.point_selected = 0
                cv2.destroyAllWindows()
                raise RuntimeError("Quit signal received during start point selection. Selection cancelled by user.")
            
            elif key == ord('r'):
                print("Resetting start point selection.")
                init.drawing = False
                init.point_selected = 0
                first_frame_copy = first_frame.copy()  # Restore original view
                # Show the restored frame
                cv2.imshow(init.bi_window_name, first_frame_copy)
                flush_key_buffer()
                
            elif key == ord('c'):
                if init.point_selected==2:
                    print("Start point Confirmed.")
                    break  # Exit point selection loop
                else:
                    print("No valid point selected to confirm. Please select a point first.")
                    init.drawing = False
                    init.point_selected = 0
                    flush_key_buffer()
        cv2.destroyWindow(init.bi_window_name)  # Close the selection window
    return