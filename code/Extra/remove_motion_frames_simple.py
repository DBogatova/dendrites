#!/usr/bin/env python
"""
Simple Motion Filter: Remove bad frames entirely instead of setting to NaN

This approach:
1. Detects motion frames (same as before)
2. Physically removes them from the 4D stack
3. Creates a clean, shorter stack for Module 1
4. Saves frame mapping for later reference
"""

from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc

# ===== CONFIG =====
DATE = "2025-08-06"
MOUSE = "organoid"
RUN = "run4-crop"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / "runB_run4_reslice-crop.tif"
MIP_PATH = BASE / "raw" / "runB_run4_reslice-crop_processed.tif"

PREPROCESSED = BASE / "preprocessed"
PREPROCESSED.mkdir(exist_ok=True)

CLEAN_STACK_PATH = PREPROCESSED / "raw_clean.tif"
FRAME_MAP_PATH = PREPROCESSED / "frame_mapping.npy"
QC_PDF = PREPROCESSED / "motion_filter_simple.pdf"

# Motion detection params - moderately more sensitive
TILES_YX = (4, 4)        # More tiles to catch local motion
ROLL_WIN = 350           # Slightly shorter window
K_MAD = 2.5              # Lower threshold (was 3.0)
PAD_NEIGHBOR = 2         # Keep original padding
MIN_VALID_PIX = 150      # Slightly lower minimum
EPS = 1e-12

# Downward spike detection (motion artifacts)
USE_SPIKE_DETECTION = True
SPIKE_THRESHOLD = -2.0   # Detect sudden drops (negative spikes)

def load_mip_movie(path):
    with tifffile.TiffFile(path) as tif:
        arr = tif.series[0].asarray().astype(np.float32)
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        return np.nanmax(arr, axis=1)

def norm_frame(F):
    med = np.nanmedian(F)
    return F / (med + EPS)

def rolling_median(x, w):
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.nanmedian(xp[i:i+w])
    return out

def rolling_mad(x, w):
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    out = np.empty_like(x)
    for i in range(len(x)):
        seg = xp[i:i+w]
        med = np.nanmedian(seg)
        out[i] = np.nanmedian(np.abs(seg - med))
    return out

def detect_motion_frames(mip):
    T, H, W = mip.shape
    ny, nx = TILES_YX
    ys = np.linspace(0, H, ny+1, dtype=int)
    xs = np.linspace(0, W, nx+1, dtype=int)
    scores = np.full(T, np.nan, dtype=np.float32)
    
    prev = norm_frame(mip[0])
    for t in range(1, T):
        cur = norm_frame(mip[t])
        tile_vals = []
        for i in range(ny):
            for j in range(nx):
                y0, y1 = ys[i], ys[i+1]
                x0, x1 = xs[j], xs[j+1]
                a = prev[y0:y1, x0:x1].ravel()
                b = cur[y0:y1, x0:x1].ravel()
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() >= MIN_VALID_PIX:
                    aa = a[m]; bb = b[m]
                    aa = (aa - aa.mean()) / (aa.std() + EPS)
                    bb = (bb - bb.mean()) / (bb.std() + EPS)
                    corr = np.clip(np.dot(aa, bb) / max(1, len(aa)), -1.0, 1.0)
                    tile_vals.append(1.0 - corr)
        if tile_vals:
            scores[t] = np.max(tile_vals)
        prev = cur
    
    # Threshold detection
    med_loc = rolling_median(scores, ROLL_WIN)
    mad_loc = rolling_mad(scores, ROLL_WIN) + EPS
    thr = med_loc + K_MAD * mad_loc
    hits = np.where(scores > thr)[0]
    
    # Pad neighbors
    padset = set()
    for f in hits:
        for k in range(max(0, f - PAD_NEIGHBOR), min(T, f + PAD_NEIGHBOR + 1)):
            padset.add(k)
    
    excluded = sorted(padset)
    
    # Additional: detect downward spikes (motion artifacts)
    if USE_SPIKE_DETECTION:
        # Compute frame-to-frame differences
        diffs = np.full(T, 0.0, dtype=np.float32)
        for t in range(1, T):
            prev_frame = norm_frame(mip[t-1])
            cur_frame = norm_frame(mip[t])
            # Mean difference (negative = drop in intensity)
            diffs[t] = np.nanmean(cur_frame - prev_frame)
        
        # Detect sudden drops
        spike_hits = np.where(diffs < SPIKE_THRESHOLD * np.nanstd(diffs))[0]
        excluded.extend(spike_hits)
        excluded = sorted(set(excluded))
        print(f"Added {len(spike_hits)} downward spike frames")
    
    return scores, thr, excluded

def main():
    print(f"Loading MIP from: {MIP_PATH}")
    mip = load_mip_movie(MIP_PATH)
    T_mip = mip.shape[0]
    
    with tifffile.TiffFile(RAW_STACK_PATH) as tif:
        T_raw, Z, Y, X = tif.series[0].shape
    
    print(f"MIP: {T_mip} frames, RAW: {T_raw} frames")
    
    # Detect motion frames
    scores, thr, excluded = detect_motion_frames(mip)
    print(f"Detected {len(excluded)} motion frames to remove")
    
    # Create frame mapping: new_idx -> original_idx
    keep_frames = [t for t in range(T_raw) if t not in excluded]
    frame_mapping = np.array(keep_frames, dtype=np.int32)
    T_clean = len(keep_frames)
    
    print(f"Keeping {T_clean}/{T_raw} frames ({100*T_clean/T_raw:.1f}%)")
    
    # QC plot
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label="Motion score")
    plt.plot(thr, 'r--', label=f"Threshold (win={ROLL_WIN}, k={K_MAD})")
    if excluded:
        xs = np.array(excluded)
        ys = scores[np.clip(xs, 0, len(scores)-1)]
        plt.scatter(xs, ys, s=10, color='red', label="Excluded", zorder=3)
    plt.title("Motion Detection (frames will be removed)")
    plt.xlabel("Original Frame"); plt.ylabel("Score")
    plt.legend(); plt.tight_layout(); plt.savefig(QC_PDF); plt.close()
    
    # Write clean stack preserving 4D structure
    print(f"Writing clean stack: {CLEAN_STACK_PATH}")
    
    with tifffile.TiffFile(RAW_STACK_PATH) as tif:
        full_stack = tif.series[0].asarray()
        T_orig, Z, Y, X = full_stack.shape
        print(f"Original: (T={T_orig}, Z={Z}, Y={Y}, X={X})")
        
        # Extract only good frames to preserve 4D structure
        clean_stack = full_stack[keep_frames].astype(np.float32)
        print(f"Clean: (T={clean_stack.shape[0]}, Z={Z}, Y={Y}, X={X})")
        
        # Write as single 4D array
        tifffile.imwrite(CLEAN_STACK_PATH, clean_stack, bigtiff=True)
        del clean_stack, full_stack
    
    # Save frame mapping
    np.save(FRAME_MAP_PATH, frame_mapping)
    
    print(f"✅ Clean stack: {CLEAN_STACK_PATH}")
    print(f"✅ Frame mapping: {FRAME_MAP_PATH}")
    print(f"✅ Use clean stack in Module 1 (set RAW_STACK_PATH)")

if __name__ == "__main__":
    main()