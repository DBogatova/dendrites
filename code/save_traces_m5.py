#!/usr/bin/env python
"""
Module 4: Save Traces and Generate Visualizations

This script processes curated masks to extract calcium traces and create visualizations:
1. Loads raw data and calculates ΔF/F
2. Processes each curated mask to extract core and background traces
3. Corrects motion artifacts
4. Generates individual trace plots with MIPs
5. Creates a combined plot of selected traces
6. Saves traces to CSV and pickle formats
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.ndimage import gaussian_filter, gaussian_filter1d, binary_erosion, binary_dilation
from skimage.morphology import ball
import pickle
from pathlib import Path
import pandas as pd
import gc

# Set matplotlib parameters
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False  # Use standard minus sign instead of Unicode

# === CONFIGURATION ===
DATE = "2025-04-22"
MOUSE = "rAi162_15"
RUN = "run6"
Y_CROP = 3
FRAME_RATE = 5  # Frames per second

# === PATHS ===
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "data" / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_reslice_bin.tif"
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
TRACE_FOLDER = BASE / "traces"
TRACE_FOLDER.mkdir(exist_ok=True)
TRACE_PKL = TRACE_FOLDER / "dff_traces_curated_bgsub.pkl"
TRACE_CSV = TRACE_FOLDER / "dff_traces_curated_bgsub.csv"
PREVIEW_FOLDER = BASE / "trace_previews_curated"
PREVIEW_FOLDER.mkdir(exist_ok=True)

# === PLOTTING OPTIONS ===
PLOT_ALL_TRACES = True
SELECTED_NAMES = [
    "dend_006", "dend_014", "dend_019", "dend_035",
    "dend_023", "dend_025", "dend_027"
]



def main():
    # === LOAD RAW STACK AND CALCULATE ΔF/F ===
    print("Loading and computing ΔF/F stack...")
    raw_stack = tifffile.imread(RAW_STACK_PATH).astype(np.float32)
    if Y_CROP > 0:
        raw_stack = raw_stack[:, :, :-Y_CROP, :]
    T = raw_stack.shape[0]

    # Calculate F0 as the mean of the lowest 20% of values
    sorted_stack = np.sort(raw_stack, axis=0)
    lowest_20pct = sorted_stack[:int(T * 0.2)]
    F0 = np.mean(lowest_20pct, axis=0)
    dff_stack = (raw_stack - F0[None]) / (F0[None] + 1e-6)
    
    # Free memory
    del sorted_stack, lowest_20pct
    gc.collect()

    # === PROCESS CURATED MASKS ===
    print(f"Loading curated masks from: {MASK_FOLDER}")
    mask_paths = sorted(MASK_FOLDER.glob("dend_*.tif"))
    traces = []
    labels = []

    for path in mask_paths:
        name = path.stem.replace("_labelmap", "")
        mask = tifffile.imread(path).astype(bool)
        
        # Ensure mask dimensions match raw stack
        if mask.shape != dff_stack.shape[1:]:
            print(f"Adjusting mask {name} from {mask.shape} to {dff_stack.shape[1:]}")
            # If mask is smaller in Y, pad it; if larger, crop it
            if mask.shape[1] < dff_stack.shape[2]:
                pad_y = dff_stack.shape[2] - mask.shape[1]
                mask = np.pad(mask, ((0,0), (0,pad_y), (0,0)), mode='constant')
            elif mask.shape[1] > dff_stack.shape[2]:
                crop_y = mask.shape[1] - dff_stack.shape[2]
                mask = mask[:, :-crop_y, :]
        
        if not np.any(mask):
            continue

        # === Extract core and shell ===
        # Core is the eroded mask (inner region)
        core_mask = binary_erosion(mask, structure=ball(1))
        if not np.any(core_mask):
            core_mask = mask.copy()

        # Shell is the dilated mask minus the original mask (surrounding region)
        shell_mask = binary_dilation(mask, structure=ball(3)) & ~mask
        
        # Calculate mean trace for core and background
        core_trace = dff_stack[:, core_mask].mean(axis=1)
        bg_trace = dff_stack[:, shell_mask].mean(axis=1) if np.any(shell_mask) else 0

        # === Final ΔF/F (%) ===
        dff = core_trace - bg_trace
        
        # Replace motion artifacts (values < -0.5) with 0
        artifact_mask = dff < -0.5
        artifact_count = np.sum(artifact_mask)
        
        if artifact_count > 0:
            print(f"  {name}: Found {artifact_count} points below -0.5, replacing with 0")
            
            # Save original values for verification
            orig_values = dff[artifact_mask].copy()
            
            # Replace artifact points with 0
            dff[artifact_mask] = 0
            
            # Print before/after for verification
            print(f"    Before: min={orig_values.min():.2f}, After: min={dff.min():.2f}")
            
        # Apply Gaussian smoothing and convert to percentage
        smoothed = gaussian_filter1d(dff, sigma=0.5) * 100

        traces.append(smoothed)
        labels.append(name)

        # === Generate preview with MIP ===
        mip = np.max(mask, axis=0)

        fig, (ax_img, ax_trace) = plt.subplots(1, 2, figsize=(12, 4), 
                                              gridspec_kw={"width_ratios": [1.2, 2]})

        # Left: MIP
        rgb = np.zeros((*mip.shape, 3), dtype=np.float32)
        color = (0.2, 0.7, 0.2)  
        for c in range(3):
            rgb[..., c][mip > 0] = color[c]
        ax_img.imshow(rgb)
        cell_number = name.split('_')[1]
        ax_img.set_title(f"Cell {cell_number} — MIP")
        ax_img.axis("off")

        # Right: Trace
        t = np.arange(T) / FRAME_RATE
        ax_trace.plot(t, smoothed, color='teal', lw=1.5)
        ax_trace.set_title("ΔF/F Trace (%)")
        ax_trace.set_xlabel("Time (s)")
        ax_trace.set_ylabel("ΔF/F (%)")
        ax_trace.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(PREVIEW_FOLDER / f"{name}_trace_mip.pdf", format='pdf')
        plt.close(fig)
        print(f"[✓] Saved: {name}_trace_mip.pdf")

    # === SAVE TO PKL AND CSV ===
    print("Saving traces to files...")
    with open(TRACE_PKL, "wb") as f:
        pickle.dump(list(zip(labels, traces)), f)

    pd.DataFrame(np.array(traces).T, columns=labels).to_csv(TRACE_CSV, index_label="Frame")
    print(f"✅ Traces saved to:\n{TRACE_PKL}\n{TRACE_CSV}")

    # === STACKED COMBO PLOT ===
    print("Generating combo plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 6  # Spacing between traces
    count = 0

    for i, (trace, name) in enumerate(zip(traces, labels)):
        if not PLOT_ALL_TRACES and name not in SELECTED_NAMES:
            continue
            
        t = np.arange(T) / FRAME_RATE
        
        # Plot trace with vertical offset
        ax.plot(t, trace + count * offset, 
               color=cm.turbo(i / len(traces)), lw=1)
        
        # Add label
        cell_number = name.split('_')[1]
        cell_label = f"Cell {cell_number}"
        ax.text(t[-1] + 1, count * offset, cell_label, va='center', fontsize=8)
        count += 1

    # === Add scale bar ===
    scale_bar_x = t[-1] - 7  # Position from right edge
    scale_bar_y_start = offset * (count - 0.5)  # Near the top trace
    scale_bar_height = 2  # 2% ΔF/F

    ax.plot([scale_bar_x, scale_bar_x], 
           [scale_bar_y_start, scale_bar_y_start + scale_bar_height], 
           color='k', lw=2)
    ax.text(scale_bar_x + 1, scale_bar_y_start + scale_bar_height / 2, 
           "2%", va='center', ha='left', fontsize=10)

    # Set plot limits and labels
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-offset, count * offset + 10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (%) + offset")
    ax.set_yticks([])
    ax.set_title("Stacked ΔF/F Traces")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(PREVIEW_FOLDER / "combo_traces_curated_bgsub.pdf", format='pdf')
    plt.close()

    print(f"✅ Combo plot saved to: {PREVIEW_FOLDER / 'combo_traces_curated_bgsub.pdf'}")
    print("Module 4 processing complete!")

if __name__ == "__main__":
    main()