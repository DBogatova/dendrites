#!/usr/bin/env python
"""
Module 1 (robust ΔF/F + motion correction + event windows) - Memory Optimized

Memory-optimized version that processes data in chunks to avoid OOM errors.
Does exactly the same as find_events_m1_fixed.py but uses chunked processing.
"""

from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import csv, gc

# ------------------- CONFIG -------------------

DATE = "2025-10-03"
MOUSE = "rAi162_15"
RUN = "run7"

# Paths
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_v1_reslice.tif"
PREPROCESSED = BASE / "preprocessed"
PREPROCESSED.mkdir(exist_ok=True)
EVENT_CROPS = PREPROCESSED / "event_crops"; EVENT_CROPS.mkdir(exist_ok=True)
EVENT_CROPS_BG = PREPROCESSED / "event_crops_bg"; EVENT_CROPS_BG.mkdir(exist_ok=True)

# Outputs
DFF_STACK_PATH = PREPROCESSED / "dff_stack.tif"
TISSUE_MASK_PATH = PREPROCESSED / "tissue_mask_2d.npy"
EVENTS_CSV_PATH = PREPROCESSED / "events_summary.csv"
EVENTS_PEAKS_NPY = PREPROCESSED / "events_peak_frames.npy"
TIMELINE_PDF = PREPROCESSED / "activity_timeline.pdf"

# Processing knobs
RECOMPUTE_DFF = True        # Set to True to recompute ΔF/F stack
Y_CROP = 3
GAUSS_SIGMA = (0.0, 0.8, 1.0, 1.0)  # (T,Z,Y,X)
F0_PERCENTILE = 10
TOP_FRAC = 0.01
Z_HI = 1.5
Z_LO = 0.5
BASELINE_WINDOW = 600
NOISE_WINDOW = 200
MIN_EVENT_LEN = 3
MAX_FRAME_GAP = 2
CROP_RADIUS = 5
TISSUE_BLUR_SIGMA = 1.0
TISSUE_PERC = None
TISSUE_MIN_AREA = 800

def group_consecutive(frames, gap=1):
    if len(frames) == 0: return []
    frames = np.array(frames, dtype=int); frames.sort()
    groups, cur = [], [int(frames[0])]
    for f in frames[1:]:
        f = int(f)
        if f - cur[-1] <= gap: cur.append(f)
        else: groups.append(cur); cur = [f]
    groups.append(cur)
    return groups

def rolling_baseline_percentile(x, win, pct):
    pad = win // 2
    xp = np.pad(x, pad, mode='reflect')
    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        out[i] = np.percentile(xp[i:i+win], pct)
    return out

def rolling_mad(x, win):
    pad = win // 2
    xp = np.pad(x, pad, mode='reflect')
    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        w = xp[i:i+win]
        m = np.median(w)
        out[i] = np.median(np.abs(w - m))
    return out

def main():
    # ---- GET SHAPE WITHOUT LOADING ----
    print("Getting stack dimensions...")
    with tifffile.TiffFile(RAW_STACK_PATH) as tif:
        shape = tif.series[0].shape
    T, Z, Y, X = shape
    if Y_CROP > 0:
        Y = Y - Y_CROP
    print(f"Shape: T={T}, Z={Z}, Y={Y}, X={X}")
    print(f"Estimated memory: {T*Z*Y*X*4/1e9:.1f} GB")

    # ---- COMPUTE OR LOAD ΔF/F STACK ----
    if RECOMPUTE_DFF or not DFF_STACK_PATH.exists():
        print("Computing F0 baseline in chunks...")
        chunk_size = min(50, T)
        
        # Collect samples for F0 calculation
        samples = []
        n_samples = min(T, 200)
        sample_indices = np.linspace(0, T-1, n_samples, dtype=int)
        
        with tifffile.TiffFile(RAW_STACK_PATH) as tif:
            for i in sample_indices:
                frame = tif.series[0].asarray()[i].astype(np.float32)
                if Y_CROP > 0:
                    frame = frame[:, :-Y_CROP, :]
                samples.append(frame)
                if len(samples) % 20 == 0:
                    print(f"  Sampled {len(samples)}/{n_samples} frames")
        
        # Compute F0 from samples
        sample_stack = np.stack(samples, axis=0)
        F0 = np.percentile(sample_stack, F0_PERCENTILE, axis=0)
        del samples, sample_stack
        gc.collect()
        
        # ---- PROCESS ΔF/F IN CHUNKS ----
        print("Processing ΔF/F in chunks...")
        
        # Initialize outputs
        tissue_accumulator = np.zeros((Y, X), dtype=np.float32)
        
        # Process and save ΔF/F stack
        with tifffile.TiffWriter(DFF_STACK_PATH, bigtiff=True) as dff_writer:
            with tifffile.TiffFile(RAW_STACK_PATH) as tif:
                for t_start in range(0, T, chunk_size):
                    t_end = min(t_start + chunk_size, T)
                    print(f"  Processing frames {t_start}-{t_end-1}/{T}")
                    
                    # Load chunk
                    chunk = tif.series[0].asarray()[t_start:t_end].astype(np.float32)
                    if Y_CROP > 0:
                        chunk = chunk[:, :, :-Y_CROP, :]
                    
                    # Compute ΔF/F for chunk
                    dff_chunk = (chunk - F0[None]) / (F0 + 1e-6)
                    
                    # Apply smoothing
                    if any(s > 0 for s in GAUSS_SIGMA):
                        dff_chunk = gaussian_filter(dff_chunk, sigma=GAUSS_SIGMA)
                    
                    # Accumulate for tissue mask (time-mean Z-MIP)
                    chunk_mip = dff_chunk.mean(axis=0).max(axis=0)
                    tissue_accumulator += chunk_mip * (t_end - t_start) / T
                    
                    # Save ΔF/F chunk
                    for t_rel in range(dff_chunk.shape[0]):
                        dff_writer.write(dff_chunk[t_rel].astype(np.float32))
                    
                    del chunk, dff_chunk, chunk_mip
                    gc.collect()
        
        print(f"Saved ΔF/F stack to {DFF_STACK_PATH}")
        
        # ---- BUILD TISSUE MASK ----
        print("Building tissue mask...")
        img = gaussian_filter(tissue_accumulator, TISSUE_BLUR_SIGMA)
        if TISSUE_PERC is None:
            thr = threshold_otsu(img)
        else:
            thr = np.percentile(img, TISSUE_PERC)
        tissue2d = img > thr
        
        # Remove small islands
        from skimage.morphology import label
        lbl = label(tissue2d)
        keep = np.zeros_like(tissue2d, bool)
        for i in range(1, int(lbl.max()) + 1):
            rr = (lbl == i)
            if rr.sum() >= TISSUE_MIN_AREA:
                keep |= rr
        tissue2d = keep
        np.save(TISSUE_MASK_PATH, tissue2d)
    else:
        print(f"Using existing ΔF/F stack: {DFF_STACK_PATH}")
        if TISSUE_MASK_PATH.exists():
            tissue2d = np.load(TISSUE_MASK_PATH)
            print(f"Loaded existing tissue mask: {tissue2d.shape}")
        else:
            print("Tissue mask not found, creating from existing ΔF/F...")
            # Build tissue mask from existing ΔF/F
            tissue_accumulator = np.zeros((Y, X), dtype=np.float32)
            with tifffile.TiffFile(DFF_STACK_PATH) as tif:
                n_frames = len(tif.pages)
                for t in range(0, min(n_frames, 100), 10):  # Sample every 10th frame
                    frame = tif.pages[t].asarray()
                    mip = frame.max(axis=0)
                    tissue_accumulator += mip / min(n_frames//10, 10)
            
            img = gaussian_filter(tissue_accumulator, TISSUE_BLUR_SIGMA)
            if TISSUE_PERC is None:
                thr = threshold_otsu(img)
            else:
                thr = np.percentile(img, TISSUE_PERC)
            tissue2d = img > thr
            
            from skimage.morphology import label
            lbl = label(tissue2d)
            keep = np.zeros_like(tissue2d, bool)
            for i in range(1, int(lbl.max()) + 1):
                rr = (lbl == i)
                if rr.sum() >= TISSUE_MIN_AREA:
                    keep |= rr
            tissue2d = keep
            np.save(TISSUE_MASK_PATH, tissue2d)
    
    # ---- COMPUTE FRAME SCORES FROM SAVED ΔF/F ----
    print("Computing robust frame scores...")
    scores = np.zeros(T, dtype=np.float32)
    K = max(1, int(TOP_FRAC * np.count_nonzero(tissue2d)))
    
    with tifffile.TiffFile(DFF_STACK_PATH) as tif:
        # Check if we have the expected number of pages (T*Z)
        total_pages = len(tif.pages)
        expected_pages = T * Z
        print(f"Total pages: {total_pages}, Expected: {expected_pages}")
        
        # Get frame dimensions - each page should be (Y,X)
        first_frame = tif.pages[0].asarray()
        if len(first_frame.shape) == 2:
            dff_Y, dff_X = first_frame.shape
            dff_Z = Z  # Use original Z dimension
        else:
            dff_Z, dff_Y, dff_X = first_frame.shape
        
        print(f"ΔF/F page shape: {first_frame.shape}, tissue mask shape: {tissue2d.shape}")
        
        # Ensure tissue mask matches ΔF/F dimensions
        if tissue2d.shape != (dff_Y, dff_X):
            print(f"Resizing tissue mask from {tissue2d.shape} to ({dff_Y}, {dff_X})")
            from skimage.transform import resize
            tissue2d = resize(tissue2d, (dff_Y, dff_X), order=0, preserve_range=True).astype(bool)
        
        # Create 3D mask
        mask3d = np.repeat(tissue2d[None, :, :], dff_Z, axis=0)
        flat_idx = np.where(mask3d.ravel())[0]
        
        for t in range(T):
            # Reconstruct 3D volume from Z pages
            vol_3d = []
            for z in range(dff_Z):
                page_idx = t * dff_Z + z
                if page_idx < total_pages:
                    page = tif.pages[page_idx].asarray()
                    vol_3d.append(page)
                else:
                    # Fill missing pages with zeros
                    vol_3d.append(np.zeros((dff_Y, dff_X), dtype=np.float32))
            
            if vol_3d:
                vol = np.stack(vol_3d, axis=0).reshape(-1)  # (Z,Y,X) -> flat
                # Ensure indices are within bounds
                valid_idx = flat_idx[flat_idx < vol.size]
                vals = vol[valid_idx]
                
                if vals.size == 0:
                    scores[t] = 0.0
                else:
                    topk = np.partition(vals, -min(K, vals.size))[-min(K, vals.size):]
                    scores[t] = float(np.mean(topk))
            else:
                scores[t] = 0.0
            
            if t % 100 == 0:
                print(f"  Processed {t+1}/{T} frames")

    # ---- DETREND SCORES AND DETECT EVENTS ----
    print("Detrending score and detecting events...")
    base = rolling_baseline_percentile(scores, BASELINE_WINDOW, pct=F0_PERCENTILE)
    resid = (scores - base)
    mad = rolling_mad(resid, NOISE_WINDOW) + 1e-6
    z = resid / (1.4826 * mad)

    # Hysteresis on z
    active = []
    in_evt = False; st = None
    for i, zi in enumerate(z):
        if not in_evt and zi > Z_HI:
            in_evt = True; st = i
        elif in_evt and zi < Z_LO:
            if st is not None and (i - st) >= MIN_EVENT_LEN:
                active.extend(range(st, i+1))
            in_evt = False; st = None
    if in_evt and st is not None and (T - st) >= MIN_EVENT_LEN:
        active.extend(range(st, T))
    active = np.array(sorted(set(active)), dtype=int)

    # Group into events
    groups = group_consecutive(active, gap=MAX_FRAME_GAP)
    print(f"Detected {len(active)} active frames, grouped into {len(groups)} events.")

    # ---- TIMELINE PLOT ----
    print("Saving activity timeline...")
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(scores, label="Top-K mean ΔF/F", color="black")
    ax[0].plot(base, label="Rolling baseline", color="orange")
    ax[0].legend(); ax[0].set_ylabel("Score")

    ax[1].plot(resid, label="Score - baseline", color="green")
    ax[1].axhline(0, color='k', alpha=0.3)
    ax[1].legend(); ax[1].set_ylabel("Residual")

    ax[2].plot(z, label="z-score", color="blue")
    ax[2].axhline(Z_HI, color="red", linestyle="--", label=f"start {Z_HI}")
    ax[2].axhline(Z_LO, color="orange", linestyle="--", label=f"end {Z_LO}")
    if len(active) > 0:
        ax[2].scatter(active, z[active], s=4, color="red")
    ax[2].legend(); ax[2].set_xlabel("Frame"); ax[2].set_ylabel("z")
    plt.tight_layout(); plt.savefig(TIMELINE_PDF, format="pdf"); plt.close()

    # ---- SAVE EVENT CROPS FROM SAVED ΔF/F ----
    print("Summarizing and saving event crops...")
    rows = []; peak_frames = []
    
    with tifffile.TiffFile(DFF_STACK_PATH) as tif:
        for i, g in enumerate(groups):
            t_start = max(0, g[0] - CROP_RADIUS)
            t_end = min(T, g[-1] + CROP_RADIUS + 1)

            # Peak frame by z-score within this event
            gi = np.array(g, dtype=int)
            local_peak_idx = int(np.argmax(z[gi]))
            peak_frame = int(gi[local_peak_idx])

            # Load crop from saved ΔF/F - reconstruct (T,Z,Y,X) structure
            crop_frames = []
            for t in range(t_start, t_end):
                # Reconstruct 3D volume from Z pages for this timepoint
                vol_3d = []
                for z_idx in range(Z):
                    page_idx = t * Z + z_idx
                    if page_idx < total_pages:
                        page = tif.pages[page_idx].asarray()
                        vol_3d.append(page)
                    else:
                        vol_3d.append(np.zeros((dff_Y, dff_X), dtype=np.float32))
                
                if vol_3d:
                    vol = np.stack(vol_3d, axis=0)  # (Z,Y,X)
                    crop_frames.append(vol)
            
            if crop_frames:
                crop = np.stack(crop_frames, axis=0).astype(np.float32)  # (T,Z,Y,X)
            else:
                continue  # Skip if no valid frames

            # Save ΔF/F crop
            out_name = f"event_group_{i:04d}_peak{peak_frame}.tif"
            tifffile.imwrite(EVENT_CROPS / out_name, crop)

            # Save 2D background (Z-MIP of time-max)
            vol_tmax = np.max(crop, axis=0)                      # (Z,Y,X)
            bg2d = np.max(vol_tmax, axis=0).astype(np.float16)   # (Y,X)
            tifffile.imwrite(EVENT_CROPS_BG / out_name.replace(".tif", "_bg.tif"), bg2d, dtype=np.float16)

            rows.append({"event_id": i, "t_start": int(t_start), "t_end": int(t_end-1),
                         "peak_frame": int(peak_frame), "score_peak": float(scores[peak_frame])})
            peak_frames.append(peak_frame)

    # Save event tables
    with open(EVENTS_CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", "t_start", "t_end", "peak_frame", "score_peak"])
        w.writeheader(); w.writerows(rows)
    np.save(EVENTS_PEAKS_NPY, np.array(peak_frames, dtype=np.int32))

    # Quick stdout summary
    if rows:
        preview = ", ".join([f"#{r['event_id']}@{r['peak_frame']}" for r in rows[:10]])
        print(f"Events (first 10): {preview} ... total {len(rows)}.")
    else:
        print("No events detected.")

    print("Module 1 complete (ΔF/F + robust events).")

if __name__ == "__main__":
    main()