#!/usr/bin/env python
"""
Downsample Traces: Process saved traces with decimation and/or smoothing options

Options:
- DECIMATE: Take every 5th point (10Hz -> 2Hz)
- SMOOTH: Apply 5-point rolling mean before decimation
- Generates individual trace PDFs and combo plot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# ===== Matplotlib config =====
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False

# ===== CONFIG =====
DATE = "2025-08-06"
MOUSE = "organoid"
RUN = "run4-crop"

# Processing options
DECIMATE = False      # Take every 5th point (10Hz -> 2Hz)
SMOOTH = True       # Apply 5-point rolling mean before decimation
ORIGINAL_FRAME_RATE = 10.0  # Hz

# ===== PATHS =====
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "data" / DATE / MOUSE / RUN
TRACE_FOLDER = BASE / "traces"
INPUT_CSV = TRACE_FOLDER / "dff_traces_curated_bgsub.csv"

# Output paths with suffix
suffix = ""
if SMOOTH: suffix += "_smooth"
if DECIMATE: suffix += "_2hz"

OUTPUT_CSV = TRACE_FOLDER / f"dff_traces_curated_bgsub{suffix}.csv"
PREVIEW_FOLDER = BASE / f"trace_previews_curated{suffix}"
PREVIEW_FOLDER.mkdir(exist_ok=True)
COMBO_PDF = PREVIEW_FOLDER / f"combo_traces{suffix}.pdf"

# Plot settings
OFFSET = 10.0
LINEWIDTH = 1.0
COLORMAP = "turbo"

# Selected traces for combo plot
SELECTED_TRACES = ["dend_001","dend_003","dend_006","dend_008","dend_009","dend_010", "dend_015"]

def main():
    print(f"Loading traces from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, index_col=0)
    
    # Process traces
    processed_df = df.copy()
    
    if SMOOTH:
        print("Applying 5-point rolling mean smoothing...")
        processed_df = processed_df.rolling(window=5, center=True).mean()
    
    if DECIMATE:
        print("Decimating: taking every 2nd point...")
        processed_df = processed_df.iloc[::2]
    
    # Update frame rate
    final_frame_rate = ORIGINAL_FRAME_RATE
    if DECIMATE:
        final_frame_rate = ORIGINAL_FRAME_RATE / 2
    
    T = len(processed_df)
    t_axis = np.arange(T) / final_frame_rate
    
    print(f"Original: {len(df)} frames at {ORIGINAL_FRAME_RATE}Hz")
    print(f"Processed: {T} frames at {final_frame_rate}Hz")
    
    # Save processed CSV
    processed_df.to_csv(OUTPUT_CSV)
    print(f"Saved processed traces: {OUTPUT_CSV}")
    
    # Generate individual trace previews
    print("Generating individual trace previews...")
    for col in processed_df.columns:
        trace = processed_df[col].values
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_axis, trace, 'b-', lw=LINEWIDTH)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔF/F (%)")
        ax.set_title(f"{col} - Processed Trace")
        ax.grid(True, alpha=0.3)
        
        pdf_path = PREVIEW_FOLDER / f"{col}_trace{suffix}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
    
    # Generate combo plot with selected traces
    print("Generating combo plot...")
    selected_cols = [col for col in SELECTED_TRACES if col in processed_df.columns]
    N = len(selected_cols)
    
    fig, ax = plt.subplots(figsize=(12, max(6, N * 0.5)))
    
    for i, col in enumerate(selected_cols):
        trace = processed_df[col].values
        color = getattr(cm, COLORMAP)(i / max(1, N - 1))
        ax.plot(t_axis, trace + i * OFFSET, color=color, lw=LINEWIDTH)
        
        # Add label
        try:
            cell_number = col.split('_')[1]
            label = f"Cell {cell_number}"
        except:
            label = col
        ax.text(t_axis[-1] + 1.0, i * OFFSET, label, va='center', fontsize=8)
    
    # Scale bar
    if N > 0:
        sb_x = t_axis[-1] - 10
        sb_y = (N - 0.5) * OFFSET
        ax.plot([sb_x, sb_x], [sb_y, sb_y + 2.0], color='k', lw=2)
        ax.text(sb_x + 1.0, sb_y + 1.0, "2%", va='center', ha='left', fontsize=10)
    
    ax.set_xlim(0, t_axis[-1])
    ax.set_ylim(-OFFSET, N * OFFSET + 5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (%) + offset")
    ax.set_yticks([])
    
    title = f"Stacked ΔF/F Traces"
    if SMOOTH and DECIMATE:
        title += " (smoothed, 2Hz)"
    elif SMOOTH:
        title += " (smoothed)"
    elif DECIMATE:
        title += " (2Hz)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(COMBO_PDF, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Individual previews saved to: {PREVIEW_FOLDER}")
    print(f"✅ Combo plot saved to: {COMBO_PDF}")
    print(f"✅ Processed CSV saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()