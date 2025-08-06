#!/usr/bin/env python
"""
Module X: Global ΔF/F Calculation for SCAPE Data (Mesoscope-like)

This script computes global calcium signals from 4D SCAPE recordings:
1. Loads 4D SCAPE stack (T, Z, Y, X)
2. Creates tissue mask by excluding zero voxels and thresholding brightest regions
3. Calculates global ΔF/F across masked tissue
4. Applies light Gaussian smoothing (optional)
5. Saves:
   - CSV with raw and smoothed traces
   - PNG plot with raw+smoothed overlay and 10 s scale bar
   - Debug mask visualization
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import gc

# === CONFIGURATION ===
DATE = "2025-04-22"
MOUSE = "rAi162_15"
RUN = "run6"

FRAME_RATE = 10        # Hz
MASK_PERCENTILE = 10   # Keep brightest X% of non-zero voxels
SMOOTH_SEC = 0       # Gaussian smoothing in seconds (0.2 s = light smoothing)
ADD_RAW_TRACE = True   # Overlay raw trace for QC
SHOW_MASK_PREVIEW = True

# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_reslice_bin.tif"
OUTPUT_FOLDER = BASE / "global_dff"
OUTPUT_FOLDER.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_FOLDER / f"{MOUSE}_{RUN}_global_dff.csv"
PLOT_PATH = OUTPUT_FOLDER / f"{MOUSE}_{RUN}_global_dff.png"
MASK_PREVIEW_PATH = OUTPUT_FOLDER / f"{RUN}_mask_preview.png"

# Matplotlib settings
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering


def compute_tissue_mask(stack, mask_percentile):
    """
    Compute binary tissue mask from time-averaged intensity,
    excluding zeros and applying percentile threshold.
    """
    mean_img = stack.mean(axis=0)

    # Debug stats
    zero_fraction = np.mean(mean_img == 0)
    print(f"mean_img stats: min={mean_img.min():.3f}, max={mean_img.max():.3f}")
    print(f"Fraction of zeros: {zero_fraction*100:.1f}%")

    # Exclude zeros for percentile
    nonzero_values = mean_img[mean_img > 0]
    threshold = np.percentile(nonzero_values, mask_percentile)
    mask = mean_img > threshold

    coverage = np.sum(mask) / mask.size * 100
    print(f"Mask threshold: {threshold:.3f}")
    print(f"Mask coverage: {coverage:.1f}% of volume")

    # Optional debug visualization
    if SHOW_MASK_PREVIEW:
        plt.figure(figsize=(8, 6))
        plt.hist(nonzero_values, bins=100, color='gray', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold ({mask_percentile}%)")
        plt.title("Mean Intensity Distribution (Non-Zero Voxels)")
        plt.xlabel("Intensity")
        plt.ylabel("Voxel count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MASK_PREVIEW_PATH, dpi=300)
        plt.close()

    return mask


def compute_global_dff(stack, mask, frame_rate, smooth_sec=0.0):
    """
    Compute global ΔF/F and optional smoothing.
    """
    masked_stack = stack[:, mask]
    F_t = masked_stack.mean(axis=1)

    # Baseline as 20th percentile
    F0 = np.percentile(F_t, 20)
    dff = (F_t - F0) / F0

    if smooth_sec > 0:
        sigma = smooth_sec * frame_rate
        dff_smooth = gaussian_filter1d(dff, sigma=sigma)
    else:
        dff_smooth = dff.copy()

    time = np.arange(len(dff)) / frame_rate
    return time, dff, dff_smooth


def plot_dff(time, dff, dff_smooth, title, save_path, overlay_raw=True):
    """
    Plot global ΔF/F with optional raw overlay and scale bar.
    """
    plt.figure(figsize=(10, 4))
    if overlay_raw:
        plt.plot(time, dff, alpha=0.4, color="gray", label="Raw ΔF/F")
    plt.plot(time, dff_smooth, color="black", linewidth=1.5, label="Smoothed")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F")
    plt.title(title)

    # Add 10-second scale bar
    scale_bar_x = 10
    y_pos = min(dff_smooth) - 0.05
    plt.plot([time[-1]-scale_bar_x, time[-1]], [y_pos, y_pos], color="black", lw=3)
    plt.text(time[-1]-scale_bar_x/2, y_pos-0.02, "10 s", ha="center")

    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    # === LOAD STACK ===
    print(f"Loading stack from {RAW_STACK_PATH}...")
    stack = tifffile.imread(RAW_STACK_PATH).astype(np.float32)
    print(f"Loaded stack: {stack.shape} (T, Z, Y, X)")

    # === CREATE MASK ===
    print("Computing tissue mask...")
    mask = compute_tissue_mask(stack, MASK_PERCENTILE)

    # === COMPUTE GLOBAL ΔF/F ===
    print("Calculating ΔF/F...")
    time, dff, dff_smooth = compute_global_dff(stack, mask, FRAME_RATE, SMOOTH_SEC)

    # === SAVE CSV ===
    print(f"Saving CSV: {CSV_PATH}")
    pd.DataFrame({"time_sec": time, "dff": dff, "dff_smooth": dff_smooth}).to_csv(CSV_PATH, index=False)

    # === PLOT ===
    print("Creating plot...")
    plot_dff(time, dff, dff_smooth, f"Global Calcium:{MOUSE} {RUN}", PLOT_PATH, overlay_raw=ADD_RAW_TRACE)

    del stack
    gc.collect()
    print(f"Done! Plot saved to {PLOT_PATH}")
    if SHOW_MASK_PREVIEW:
        print(f"Mask histogram saved to {MASK_PREVIEW_PATH}")


if __name__ == "__main__":
    main()
