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
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import gc

# Set matplotlib font
mpl.rcParams['font.family'] = 'CMU Serif'

# === CONFIGURATION ===
DATE = "2025-08-27"
MOUSE = "rAi162_18"
RUN = "run2"
CROP_RADIUS = 5  # Number of frames to include before/after each event
START_THRESHOLD = 1  # Z-score threshold for event start
END_THRESHOLD = 0   # Z-score threshold for event end (hysteresis)
MAX_FRAME_GAP = 2     # Maximum gap between frames to group into same event
Y_CROP = 3            # Number of pixels to crop from bottom of Y dimension
BASELINE_PERCENTILE = 10  # Percentile for rolling baseline
BASELINE_WINDOW = 600     # Window size for rolling baseline (frames) - 60s at 10Hz
NOISE_WINDOW = 100        # Window size for MAD noise estimation (frames) - 10s at 10Hz
MIN_EVENT_DURATION = 3    # Minimum event duration in frames
MIN_PROMINENCE = 1.0      # Minimum prominence in MAD units


# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green_reslice.tif"
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

    # === DETRENDING AND EVENT DETECTION ===
    print("Computing rolling baseline for multiplicative detrending...")
    frame_scores = stack_smooth.max(axis=(1, 2, 3))
    
    # Pad signal for edge handling
    padded_scores = np.pad(frame_scores, BASELINE_WINDOW//2, mode='reflect')
    
    # Compute rolling percentile baseline with padding
    baseline = np.zeros_like(frame_scores)
    half_window = BASELINE_WINDOW // 2
    
    for i in range(len(frame_scores)):
        start = i  # Already padded
        end = i + BASELINE_WINDOW
        baseline[i] = np.percentile(padded_scores[start:end], BASELINE_PERCENTILE)
    
    # Multiplicative detrending: F_corr = F/B - 1
    f_corr = (frame_scores / (baseline + 1e-6)) - 1
    
    # Compute rolling MAD for robust noise estimation
    mad_values = np.zeros_like(f_corr)
    half_noise_window = NOISE_WINDOW // 2
    
    for i in range(len(f_corr)):
        start = max(0, i - half_noise_window)
        end = min(len(f_corr), i + half_noise_window + 1)
        window_data = f_corr[start:end]
        median_val = np.median(window_data)
        mad_values[i] = np.median(np.abs(window_data - median_val))
    
    # Compute robust z-scores
    median_f_corr = np.median(f_corr)
    z_scores = (f_corr - median_f_corr) / (1.4826 * mad_values + 1e-6)
    
    # Detect events with hysteresis thresholding
    print("Detecting events with hysteresis thresholding...")
    active_frames = []
    in_event = False
    event_start = None
    
    for i, z in enumerate(z_scores):
        if not in_event and z > START_THRESHOLD:
            in_event = True
            event_start = i
        elif in_event and z < END_THRESHOLD:
            if event_start is not None and (i - event_start) >= MIN_EVENT_DURATION:
                active_frames.extend(range(event_start, i + 1))
            in_event = False
            event_start = None
    
    # Handle case where event extends to end of recording
    if in_event and event_start is not None:
        if (len(z_scores) - event_start) >= MIN_EVENT_DURATION:
            active_frames.extend(range(event_start, len(z_scores)))
    
    active_frames = np.array(active_frames)
    np.save(ACTIVE_FRAMES_PATH, active_frames)
    print(f"Detected {len(active_frames)} active frames using robust method.")

    # === ACTIVITY TIMELINE PLOT ===
    print("Creating activity timeline plot...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Top: Raw signal with baseline
    axes[0].plot(frame_scores, label='Raw signal', alpha=0.7)
    axes[0].plot(baseline, label='Rolling baseline', color='orange')
    axes[0].set_ylabel("Max activity")
    axes[0].legend()
    axes[0].set_title("Raw Signal and Rolling Baseline")
    
    # Middle: Multiplicatively corrected signal
    axes[1].plot(f_corr, label='F/B - 1', color='green')
    axes[1].set_ylabel("Corrected ΔF/F")
    axes[1].legend()
    axes[1].set_title("Multiplicatively Detrended Signal")
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Bottom: Z-scores with detections
    axes[2].plot(z_scores, label='Z-score', color='blue')
    if len(active_frames) > 0:
        axes[2].scatter(active_frames, z_scores[active_frames], color='red', s=1, label='Detected events')
    axes[2].axhline(START_THRESHOLD, color='red', linestyle='--', label=f'Start threshold ({START_THRESHOLD})')
    axes[2].axhline(END_THRESHOLD, color='orange', linestyle='--', label=f'End threshold ({END_THRESHOLD})')
    axes[2].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Robust Z-score")
    axes[2].legend()
    axes[2].set_title("Robust Z-scores with Hysteresis Detection")
    
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
    try:
        tifffile.imwrite(NORM_STACK_PATH, stack_norm.astype(np.float32))
        print(f"Saved normalized stack to: {NORM_STACK_PATH}")
    except OSError as e:
        print(f"Warning: Could not save normalized stack due to disk space/size limit: {e}")
        print("Skipping normalized stack save...")
    
    try:
        tifffile.imwrite(SMOOTHED_STACK_PATH, stack_smooth.astype(np.float32))
        print(f"Saved smoothed stack to: {SMOOTHED_STACK_PATH}")
    except OSError as e:
        print(f"Warning: Could not save smoothed stack due to disk space/size limit: {e}")
        print("Skipping smoothed stack save...")

    print("Module 1 processing complete!")

if __name__ == "__main__":
    main()