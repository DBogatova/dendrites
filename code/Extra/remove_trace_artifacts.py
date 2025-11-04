#!/usr/bin/env python
"""
Remove frames based on trace artifacts (downward spikes)

This script:
1. Loads existing traces
2. Detects frames with sudden drops across multiple ROIs
3. Creates a new clean stack excluding those frames
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tifffile

# ===== CONFIG =====
DATE = "2025-08-06"
MOUSE = "organoid"
RUN = "run4-crop"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
TRACE_CSV = BASE / "traces" / "dff_traces_curated_bgsub.csv"
CLEAN_STACK_PATH = BASE / "preprocessed" / "raw_clean.tif"
FINAL_CLEAN_PATH = BASE / "preprocessed" / "raw_final_clean.tif"

# Detection params - conservative
MIN_ROIS_AFFECTED = 3   # Multiple ROIs must be affected
LARGE_DROP_THRESHOLD = -8.0  # Only very large drops (8% ΔF/F)

def detect_trace_artifacts(traces_df):
    """Detect frames where traces show sudden drops."""
    traces = traces_df.values  # (T, N_rois)
    T, N = traces.shape
    print(f"Analyzing {T} frames across {N} ROIs")
    
    # Compute frame-to-frame differences for each ROI
    diffs = np.diff(traces, axis=0)  # (T-1, N)
    
    artifact_frames = []
    debug_count = 0
    
    for t in range(diffs.shape[0]):
        frame_diffs = diffs[t]  # differences for this frame
        
        # Only detect severe drops affecting multiple ROIs
        severe_drops = np.sum(frame_diffs < LARGE_DROP_THRESHOLD)
        
        if severe_drops >= MIN_ROIS_AFFECTED:
            artifact_frames.append(t + 1)  # +1 because diff shifts indices
            
            # Debug first few detections
            if debug_count < 10:
                print(f"Frame {t+1}: {severe_drops} ROIs with drops < {LARGE_DROP_THRESHOLD}%")
                print(f"  Worst drops: {np.sort(frame_diffs)[:3]}")
                debug_count += 1
    
    return artifact_frames

def main():
    print(f"Loading traces from: {TRACE_CSV}")
    df = pd.read_csv(TRACE_CSV, index_col=0)
    
    # Detect artifact frames
    artifact_frames = detect_trace_artifacts(df)
    print(f"Detected {len(artifact_frames)} frames with trace artifacts")
    
    if not artifact_frames:
        print("No artifacts detected")
        return
    
    # Load current clean stack
    print(f"Loading clean stack: {CLEAN_STACK_PATH}")
    stack = tifffile.imread(CLEAN_STACK_PATH)
    T, Z, Y, X = stack.shape
    print(f"Current stack: (T={T}, Z={Z}, Y={Y}, X={X})")
    
    # Remove artifact frames
    keep_frames = [t for t in range(T) if t not in artifact_frames]
    final_stack = stack[keep_frames]
    
    print(f"Final stack: (T={final_stack.shape[0]}, Z={Z}, Y={Y}, X={X})")
    print(f"Removed {len(artifact_frames)} frames ({100*len(artifact_frames)/T:.1f}%)")
    
    # Save final clean stack
    tifffile.imwrite(FINAL_CLEAN_PATH, final_stack, bigtiff=True)
    print(f"✅ Saved final clean stack: {FINAL_CLEAN_PATH}")
    
    # Update save_traces_m4 to use this path
    print("Update save_traces_m4.py to use raw_final_clean.tif")

if __name__ == "__main__":
    main()