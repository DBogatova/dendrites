#!/usr/bin/env python
"""
Module 1: Initial Processing of Calcium Imaging Data

This script performs the initial processing of raw calcium imaging data:
1. Loads the raw stack
2. Normalizes each voxel
3. Subtracts mean per time frame
4. Applies Gaussian smoothing
5. Detects active frames based on activity threshold
6. Groups consecutive active frames into events
7. Saves processed data and visualizations
"""

import tifffile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
import gc

# Set matplotlib font
mpl.rcParams['font.family'] = 'CMU Serif'

# === CONFIGURATION ===
DATE = "2025-04-22"
MOUSE = "rAi162_15"
RUN = "run6"
CROP_RADIUS = 5  # Number of frames to include before/after each event
ACTIVITY_THRESHOLD = 0.37  # Threshold for detecting active frames
MAX_FRAME_GAP = 1  # Maximum gap between frames to group into same event
Y_CROP = 3          # Number of pixels to crop from bottom of Y dimension


# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_reslice_bin.tif"
PREPROCESSED_FOLDER = BASE / "preprocessed"
PREPROCESSED_FOLDER.mkdir(exist_ok=True)

# === OUTPUT PATHS ===
NORM_STACK_PATH = PREPROCESSED_FOLDER / "stack_voxel_norm_mean_sub.tif"
SMOOTHED_STACK_PATH = PREPROCESSED_FOLDER / "stack_smoothed.tif"
ACTIVE_FRAMES_PATH = PREPROCESSED_FOLDER / "active_frames.npy"
PREVIEW_FOLDER = PREPROCESSED_FOLDER / "active_frame_previews"
PREVIEW_FOLDER.mkdir(exist_ok=True)
EVENT_CROPS_FOLDER = PREPROCESSED_FOLDER / "event_crops"
EVENT_CROPS_FOLDER.mkdir(exist_ok=True)

def group_consecutive(frames, gap=1):
    """
    Group consecutive frame numbers into events, allowing for small gaps.
    
    Args:
        frames: Array of frame numbers
        gap: Maximum allowed gap between consecutive frames to be in same group
        
    Returns:
        List of lists, where each inner list contains frame numbers for one event
    """
    if len(frames) == 0:
        return []
        
    groups = []
    group = [frames[0]]
    
    for f in frames[1:]:
        if f - group[-1] <= gap:
            group.append(f)
        else:
            groups.append(group)
            group = [f]
            
    groups.append(group)
    return groups

def main():
    # === LOAD RAW STACK ===
    
    print("Loading stack...")
    stack = tifffile.imread(RAW_STACK_PATH).astype(np.float32)
    stack = stack[:, :, :-Y_CROP, :]  # Crop Y dimension
    print(f"Shape: {stack.shape} (T, Z, Y, X)")

    # === NORMALIZE EACH VOXEL ===
    print("Normalizing voxels...")
    vmin = stack.min(axis=0, keepdims=True)
    vmax = stack.max(axis=0, keepdims=True)
    stack_norm = (stack - vmin) / (vmax - vmin + 1e-6)
    del stack, vmin, vmax
    gc.collect()

    # === SUBTRACT MEAN PER TIME FRAME ===
    print("Subtracting mean per time frame...")
    frame_mean = stack_norm.mean(axis=(1, 2, 3), keepdims=True)
    stack_norm -= frame_mean
    del frame_mean
    gc.collect()

    # === GAUSSIAN SMOOTHING ===
    print("Applying Gaussian smoothing...")
    stack_smooth = gaussian_filter(stack_norm, sigma=1)
    ZS = np.arange(stack_smooth.shape[1])  # Use all Z planes

    # === DETECT ACTIVE FRAMES ===
    print("Detecting active frames...")
    frame_scores = stack_smooth.max(axis=(1, 2, 3))
    active_frames = np.where(frame_scores > ACTIVITY_THRESHOLD)[0]
    np.save(ACTIVE_FRAMES_PATH, active_frames)
    print(f"Detected {len(active_frames)} active frames.")

    # === ACTIVITY TIMELINE PLOT ===
    print("Creating activity timeline plot...")
    plt.figure(figsize=(12, 3))
    plt.plot(frame_scores)
    plt.scatter(active_frames, frame_scores[active_frames], color='red')
    plt.axhline(ACTIVITY_THRESHOLD, color='gray', linestyle='--')
    plt.title("Detected Events")
    plt.xlabel("Frame")
    plt.ylabel("Max activity")
    plt.tight_layout()
    plt.savefig(PREPROCESSED_FOLDER / "activity_timeline.pdf", format='pdf')
    plt.close()

    # === GROUP CONSECUTIVE FRAMES INTO EVENTS ===
    print("Grouping consecutive frames into events...")
    event_groups = group_consecutive(active_frames, gap=MAX_FRAME_GAP)
    print(f"Grouped into {len(event_groups)} events.")

    # === SAVE EVENT CROPS ===
    print("Saving event crops...")
    T, Z, Y, X = stack_smooth.shape
    for i, group in enumerate(event_groups):
        t_start = max(group[0] - CROP_RADIUS, 0)
        t_end = min(group[-1] + CROP_RADIUS + 1, T)
        crop = stack_smooth[t_start:t_end]
        tifffile.imwrite(EVENT_CROPS_FOLDER / f"event_group_{i:04d}.tif", crop.astype(np.float32))

    print(f"Saved {len(event_groups)} grouped event crops to:\n{EVENT_CROPS_FOLDER}")

    # === SAVE STACKS ===
    print("Saving processed stacks...")
    tifffile.imwrite(NORM_STACK_PATH, stack_norm.astype(np.float32))
    tifffile.imwrite(SMOOTHED_STACK_PATH, stack_smooth.astype(np.float32))

    print("Module 1 processing complete!")

if __name__ == "__main__":
    main()