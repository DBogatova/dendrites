#!/usr/bin/env python
"""
Module 1 (robust ΔF/F + motion correction + event windows) - NaN aware, memory-optimized

- If preprocessed/raw_nanmasked.tif exists, uses it; otherwise uses raw/*.tif
- Computes F0 from valid (non-NaN) frames with np.nanpercentile
- Keeps excluded frames as NaN throughout; smoothing never bleeds across them
- Builds tissue mask ignoring NaNs
- Frame scoring and baseline estimation are NaN-safe (excluded frames never trigger events)
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

DATE = "2025-08-06"
MOUSE = "organoid"
RUN = "run4-crop"

# Paths
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_ORIG_PATH = BASE / "raw" / f"runB_run4_reslice-crop.tif"
RAW_CLEAN_PATH = BASE / "preprocessed" / "raw_clean.tif"
RAW_STACK_PATH = RAW_CLEAN_PATH if RAW_CLEAN_PATH.exists() else RAW_ORIG_PATH

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
Z_HI = 2.5
Z_LO = 1.0
BASELINE_WINDOW = 600
NOISE_WINDOW = 200
MIN_EVENT_LEN = 3
MAX_FRAME_GAP = 2
CROP_RADIUS = 5
TISSUE_BLUR_SIGMA = 1.0
TISSUE_PERC = None
TISSUE_MIN_AREA = 800

EPS = 1e-6

# ------------------- HELPERS -------------------

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

def is_bad_frame(vol):  # vol shape (Z,Y,X)
    return not np.isfinite(vol).any()

def interp_nans_1d(x):
    x = x.astype(np.float32)
    nans = ~np.isfinite(x)
    if nans.all():
        # All values are NaN - fill with zeros
        x[:] = 0.0
        return x
    if nans.any():
        xi = np.arange(len(x))
        x[nans] = np.interp(xi[nans], xi[~nans], x[~nans])
    return x

def nan_gaussian_filter(arr, sigma):
    """
    NaN-aware Gaussian smoothing:
    - Fill NaNs with 0
    - Convolve signal and mask
    - Normalize to avoid bleed-in from NaNs
    """
    mask = np.isfinite(arr).astype(np.float32)
    arr_filled = np.nan_to_num(arr, nan=0.0)
    smooth = gaussian_filter(arr_filled, sigma=sigma)
    norm = gaussian_filter(mask, sigma=sigma) + EPS
    out = smooth / norm
    # keep original NaNs (e.g., whole-frame exclusions)
    out[mask == 0] = np.nan
    return out

# ------------------- MAIN -------------------

def main():
    # ---- GET SHAPE WITHOUT LOADING ----
    print(f"Using {'clean' if RAW_STACK_PATH == RAW_CLEAN_PATH else 'original'} RAW: {RAW_STACK_PATH}")
    with tifffile.TiffFile(RAW_STACK_PATH) as tif:
        n_pages = len(tif.pages)
        Z = 28  # 🔧 set this to your known number of z-planes
        T = n_pages // Z
        Y, X = tif.pages[0].asarray().shape
        print(f"Detected {n_pages} pages → T={T}, Z={Z}, Y={Y}, X={X}")

    if Y_CROP > 0:
        Y = Y - Y_CROP
    print(f"Shape: T={T}, Z={Z}, Y={Y}, X={X}")
    print(f"Estimated memory: {T*Z*Y*X*4/1e9:.1f} GB")

    # ---- COMPUTE OR LOAD ΔF/F STACK ----
    if RECOMPUTE_DFF or not DFF_STACK_PATH.exists():
        print("Computing F0 baseline (NaN-safe)...")
        # Collect samples for F0 calculation
        samples = []
        n_samples = min(T, 200)
        sample_indices = np.linspace(0, T-1, n_samples, dtype=int)
        
        with tifffile.TiffFile(RAW_STACK_PATH) as tif:
            n_pages = len(tif.pages)
            assert n_pages == T * Z, f"Page count mismatch: {n_pages} vs {T*Z}"
            for i in sample_indices:
                z_start = i * Z
                z_end   = z_start + Z
                pages = [tif.pages[z].asarray().astype(np.float32) for z in range(z_start, z_end)]
                frame = np.stack(pages, axis=0)  # (Z,Y,X)
                if Y_CROP > 0:
                    frame = frame[:, :-Y_CROP, :]
                if not np.isfinite(frame).any():
                    continue
                samples.append(frame)
                if len(samples) % 20 == 0:
                    print(f"  Sampled {len(samples)} valid frames")

        if len(samples) == 0:
            raise RuntimeError("No valid frames to compute F0.")
        sample_stack = np.stack(samples, axis=0)  # (Ns,Z,Y,X)
        F0 = np.nanpercentile(sample_stack, F0_PERCENTILE, axis=0).astype(np.float32)
        del samples, sample_stack
        gc.collect()
        
        # ---- PROCESS ΔF/F IN CHUNKS ----
        print("Processing ΔF/F in chunks (NaN-safe)...")
        tissue_accumulator = np.zeros((Y, X), dtype=np.float32)
        chunk_size = min(50, T)
        
        with tifffile.TiffWriter(DFF_STACK_PATH, bigtiff=True) as dff_writer:
            with tifffile.TiffFile(RAW_STACK_PATH) as tif:
                for t_start in range(0, T, chunk_size):
                    t_end = min(t_start + chunk_size, T)
                    print(f"  Processing frames {t_start}-{t_end-1}/{T}")
                    
                    # Load chunk and rebuild 4D (T_chunk,Z,Y,X)
                    chunk_frames = []
                    for t in range(t_start, t_end):
                        z_start = t * Z
                        z_end = z_start + Z
                        pages = [tif.pages[z].asarray().astype(np.float32) for z in range(z_start, z_end)]
                        vol = np.stack(pages, axis=0)  # (Z,Y,X)
                        if Y_CROP > 0:
                            vol = vol[:, :-Y_CROP, :]
                        chunk_frames.append(vol)
                    chunk = np.stack(chunk_frames, axis=0)  # (T_chunk,Z,Y,X)

                    # Identify fully-NaN timepoints
                    bad_t = np.array([~np.isfinite(chunk[k]).any() for k in range(chunk.shape[0])])

                    # ΔF/F
                    dff_chunk = (chunk - F0[None]) / (F0 + EPS)

                    # Smooth with NaN-aware Gaussian (no bleed)
                    if any(s > 0 for s in GAUSS_SIGMA):
                        dff_chunk = nan_gaussian_filter(dff_chunk, GAUSS_SIGMA)

                    # Restore NaNs on bad frames (entire volume)
                    dff_chunk[bad_t] = np.nan

                    # Accumulate for tissue mask (time-mean Z-MIP), NaN-aware
                    chunk_mip = np.nanmean(dff_chunk, axis=0).max(axis=0)  # (Y,X)
                    tissue_accumulator += chunk_mip * (t_end - t_start) / T
                    
                    # Save ΔF/F chunk (as Z-pages per timepoint)
                    for t_rel in range(dff_chunk.shape[0]):
                        dff_writer.write(dff_chunk[t_rel].astype(np.float32))
                    
                    del chunk, dff_chunk, chunk_mip, chunk_frames
                    gc.collect()
        
        print(f"Saved ΔF/F stack to {DFF_STACK_PATH}")
        
        # ---- BUILD TISSUE MASK ----
        print("Building tissue mask (NaN-safe)...")
        img = gaussian_filter(np.nan_to_num(tissue_accumulator, nan=0.0, posinf=0.0, neginf=0.0),
                              TISSUE_BLUR_SIGMA)
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
            print("Tissue mask not found, creating from existing ΔF/F (NaN-safe)...")
            tissue_accumulator = np.zeros((Y, X), dtype=np.float32)
            valid_count = 0
            with tifffile.TiffFile(DFF_STACK_PATH) as tif:
                n_pages = len(tif.pages)
                # Sample more frames and only use valid ones
                for p in range(0, min(n_pages, 500), 5):
                    frame = tif.pages[p].asarray().astype(np.float32)
                    if frame.ndim == 2:
                        mip = frame
                    else:  # (Z,Y,X)
                        mip = np.nanmax(frame, axis=0)
                    # Only accumulate if frame has valid data
                    if np.isfinite(mip).any():
                        tissue_accumulator += np.nan_to_num(mip, nan=0.0)
                        valid_count += 1
            
            if valid_count == 0:
                print("ERROR: No valid frames found for tissue mask!")
                tissue2d = np.ones((Y, X), dtype=bool)  # Use entire field as fallback
            else:
                tissue_accumulator /= valid_count
                img = gaussian_filter(tissue_accumulator, TISSUE_BLUR_SIGMA)
                
                # Use lower threshold since motion removal may have reduced signal
                if TISSUE_PERC is None:
                    thr = threshold_otsu(img) * 0.5  # Lower threshold
                else:
                    thr = np.percentile(img, max(10, TISSUE_PERC - 20))  # Lower percentile
                tissue2d = img > thr
                
                # Use smaller minimum area since tissue may be fragmented
                from skimage.morphology import label
                lbl = label(tissue2d)
                keep = np.zeros_like(tissue2d, bool)
                min_area = max(100, TISSUE_MIN_AREA // 4)  # Smaller minimum area
                for i in range(1, int(lbl.max()) + 1):
                    rr = (lbl == i)
                    if rr.sum() >= min_area:
                        keep |= rr
                tissue2d = keep
                
                # Final fallback: if still empty, use top 50% of pixels
                if not tissue2d.any():
                    print("WARNING: Tissue detection failed, using top 50% of pixels")
                    thr = np.percentile(img, 50)
                    tissue2d = img > thr
            
            np.save(TISSUE_MASK_PATH, tissue2d)
            print(f"Created tissue mask with {np.count_nonzero(tissue2d)} pixels from {valid_count} valid frames")
    
    # ---- DEBUG: Check frame consistency ----
    print(f"DEBUG: Raw stack T: {T}")
    print(f"DEBUG: Tissue mask shape: {tissue2d.shape}, non-zero pixels: {np.count_nonzero(tissue2d)}")
    
    # ---- COMPUTE FRAME SCORES FROM SAVED ΔF/F ----
    print("Computing robust frame scores (NaN-safe)...")
    scores = np.full(T, np.nan, dtype=np.float32)
    K = max(1, int(TOP_FRAC * np.count_nonzero(tissue2d)))
    print(f"DEBUG: Will extract top K={K} pixels per frame")
    
    with tifffile.TiffFile(DFF_STACK_PATH) as tif:
        total_pages = len(tif.pages)
        expected_pages = T * Z
        print(f"Total pages: {total_pages}, Expected: {expected_pages}")
        
        first_frame = tif.pages[0].asarray().astype(np.float32)
        if first_frame.ndim == 2:
            dff_Y, dff_X = first_frame.shape
            dff_Z = Z
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
        
        # DEBUG: Check first 10 frames for valid data
        valid_frames = []
        for t in range(min(10, T)):
            vol_3d = []
            for z in range(dff_Z):
                page_idx = t * dff_Z + z
                if page_idx < total_pages:
                    page = tif.pages[page_idx].asarray().astype(np.float32)
                    vol_3d.append(page)
                else:
                    vol_3d.append(np.zeros((dff_Y, dff_X), dtype=np.float32))
            vol = np.stack(vol_3d, axis=0)
            if np.isfinite(vol).any():
                valid_frames.append(t)
        print(f"DEBUG: First 10 frames with valid data: {valid_frames}")
        
        for t in range(T):
            # Reconstruct 3D volume from Z pages
            vol_3d = []
            for z in range(dff_Z):
                page_idx = t * dff_Z + z
                if page_idx < total_pages:
                    page = tif.pages[page_idx].asarray().astype(np.float32)
                    vol_3d.append(page)
                else:
                    vol_3d.append(np.zeros((dff_Y, dff_X), dtype=np.float32))
            vol = np.stack(vol_3d, axis=0).reshape(-1)  # (Z,Y,X) -> flat
            
            # Extract valid (finite) mask voxels
            valid_idx = flat_idx[flat_idx < vol.size]
            vals = vol[valid_idx]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                scores[t] = np.nan
            else:
                k = min(K, vals.size)
                topk = np.partition(vals, -k)[-k:]
                scores[t] = float(np.nanmean(topk))
            
            if t % 100 == 0:
                print(f"  Processed {t+1}/{T} frames")

    # ---- DEBUG: Check scores array ----
    valid_scores = np.sum(~np.isnan(scores))
    print(f"DEBUG: Non-NaN scores: {valid_scores}/{T} ({100*valid_scores/T:.1f}%)")
    if valid_scores > 0:
        print(f"DEBUG: Score range: {np.nanmin(scores):.3f} to {np.nanmax(scores):.3f}")
    else:
        print("DEBUG: WARNING - All scores are NaN!")
    
    # ---- DETREND SCORES AND DETECT EVENTS ----
    print("Detrending score and detecting events...")
    s_for_base = interp_nans_1d(scores.copy())   # fill NaNs for baseline only
    base = rolling_baseline_percentile(s_for_base, BASELINE_WINDOW, pct=F0_PERCENTILE)
    resid = scores - base
    resid_clean = np.nan_to_num(resid, nan=0.0)  # for robust MAD estimate
    mad = rolling_mad(resid_clean, NOISE_WINDOW) + EPS
    z = resid / (1.4826 * mad)

    # Excluded/NaN frames must never trigger events
    z[~np.isfinite(scores)] = -np.inf

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
    ax[0].plot(interp_nans_1d(scores.copy()), label="Top-K mean ΔF/F (filled for viz)", color="black")
    ax[0].plot(base, label="Rolling baseline", color="orange")
    ax[0].legend(); ax[0].set_ylabel("Score")

    ax[1].plot(np.nan_to_num(resid, nan=np.nanmean(resid[np.isfinite(resid)])), 
               label="Score - baseline", color="green")
    ax[1].axhline(0, color='k', alpha=0.3)
    ax[1].legend(); ax[1].set_ylabel("Residual")

    ax[2].plot(np.where(np.isfinite(z), z, np.nan), label="z-score", color="blue")
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
        total_pages = len(tif.pages)
        # reuse dff_Z, dff_Y, dff_X from above block
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
                vol_3d = []
                for z_idx in range(dff_Z):
                    page_idx = t * dff_Z + z_idx
                    if page_idx < total_pages:
                        page = tif.pages[page_idx].asarray().astype(np.float32)
                        vol_3d.append(page)
                    else:
                        vol_3d.append(np.zeros((dff_Y, dff_X), dtype=np.float32))
                vol = np.stack(vol_3d, axis=0)  # (Z,Y,X)
                crop_frames.append(vol)
            
            crop = np.stack(crop_frames, axis=0).astype(np.float32)  # (T,Z,Y,X)

            # Save ΔF/F crop
            out_name = f"event_group_{i:04d}_peak{peak_frame}.tif"
            tifffile.imwrite(EVENT_CROPS / out_name, crop)

            # Save 2D background (Z-MIP of time-max)
            vol_tmax = np.nanmax(crop, axis=0)                  # (Z,Y,X)
            bg2d = np.nanmax(vol_tmax, axis=0).astype(np.float16)   # (Y,X)
            tifffile.imwrite(EVENT_CROPS_BG / out_name.replace(".tif", "_bg.tif"),
                             bg2d, dtype=np.float16)

            rows.append({"event_id": i, "t_start": int(t_start), "t_end": int(t_end-1),
                         "peak_frame": int(peak_frame), "score_peak": float(interp_nans_1d(scores.copy())[peak_frame])})
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

    print("Module 1 complete (ΔF/F + robust events, NaN-aware).")

if __name__ == "__main__":
    main()
