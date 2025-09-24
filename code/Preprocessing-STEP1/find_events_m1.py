#!/usr/bin/env python
"""
Module 1 (robust ΔF/F + motion correction + event windows)

What's different (to help Module 2):
- No min-max normalization; no frame-mean subtraction (keeps real amplitudes).
- Optional rigid XY motion correction (fast).
- Compute per-voxel baseline F0 (10th percentile) and ΔF/F.
- Robust event score per frame: mean of top K% ΔF/F voxels (not spatial max).
- Save ΔF/F stack, a 2D tissue mask, per-event ΔF/F crops, and per-event 2D Z-MIP tmax backgrounds.
- events_summary.csv (event_id, t_start, t_end, peak_frame, score_peak).

Outputs (all under .../<DATE>/<MOUSE>/<RUN>/preprocessed):
- dff_stack.tif                      (float32, T,Z,Y,X)
- tissue_mask_2d.npy                 (bool, Y,X)
- events_summary.csv / events_peak_frames.npy
- event_crops/event_group_XXXX_peakNNNN.tif        (ΔF/F crops, float32)
- event_crops_bg/event_group_XXXX_peakNNNN_bg.tif  (2D background, float16)
- activity_timeline.pdf
"""

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
import csv, gc

# ------------------- CONFIG -------------------
#mpl.rcParams['font.family'] = 'CMU Serif'

DATE = "2025-08-18"
MOUSE = "rAi162_15"
RUN = "run9"

# Paths
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green_reslice.tif"
PREPROCESSED = BASE / "preprocessed"
PREPROCESSED.mkdir(exist_ok=True)
EVENT_CROPS = PREPROCESSED / "event_crops"; EVENT_CROPS.mkdir(exist_ok=True)
EVENT_CROPS_BG = PREPROCESSED / "event_crops_bg"; EVENT_CROPS_BG.mkdir(exist_ok=True)
PREVIEWS = PREPROCESSED / "active_frame_previews"; PREVIEWS.mkdir(exist_ok=True)

# Outputs
DFF_STACK_PATH = PREPROCESSED / "dff_stack.tif"
TISSUE_MASK_PATH = PREPROCESSED / "tissue_mask_2d.npy"
EVENTS_CSV_PATH = PREPROCESSED / "events_summary.csv"
EVENTS_PEAKS_NPY = PREPROCESSED / "events_peak_frames.npy"
TIMELINE_PDF = PREPROCESSED / "activity_timeline.pdf"

# Processing knobs
Y_CROP = 3                   # crop bottom Y rows if needed
MOTION_CORRECT = False        # rigid XY
MC_REF_STRIDE = 25           # build reference from median of every Nth frame
GAUSS_SIGMA = (0.0, 0.8, 1.0, 1.0)  # (T,Z,Y,X) — no temporal blur

# ΔF/F baseline
F0_PERCENTILE = 10

# Event scoring & detection
TOP_FRAC = 0.01              # top K fraction of voxels per frame (e.g., 1%)
Z_HI = 1.5                   # start threshold in z-scored event score
Z_LO = 0.5                   # end threshold (hysteresis)
BASELINE_WINDOW = 600        # frames for rolling baseline of score
NOISE_WINDOW = 200           # frames for rolling MAD of score
MIN_EVENT_LEN = 3
MAX_FRAME_GAP = 2
CROP_RADIUS = 5              # frames before/after group when saving crops

# Tissue mask
TISSUE_BLUR_SIGMA = 1.0
TISSUE_PERC = None           # if None → Otsu; else integer percentile (e.g., 70)
TISSUE_MIN_AREA = 800        # pixels in 2D
# ---------------------------------------------

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

def build_tissue_mask(stack_TZYX):
    """2D mask from time-mean Z-MIP, with simple closing + island filter."""
    mean_2d = stack_TZYX.mean(axis=0).max(axis=0).astype(np.float32)
    img = gaussian_filter(mean_2d, TISSUE_BLUR_SIGMA)
    if TISSUE_PERC is None:
        thr = threshold_otsu(img)
    else:
        thr = np.percentile(img, TISSUE_PERC)
    m = img > thr
    # simple close by gaussian then threshold already smooths; remove small islands
    from skimage.morphology import label
    lbl = label(m)
    keep = np.zeros_like(m, bool)
    for i in range(1, int(lbl.max()) + 1):
        rr = (lbl == i)
        if rr.sum() >= TISSUE_MIN_AREA:
            keep |= rr
    return keep

def rigid_xy_motion_correct(stack_TZYX):
    """
    Rigid XY registration per frame to a reference (median of sparse frames).
    Uses phase_cross_correlation on a Z-MIP for speed, applies same shift to all Z slices.
    Returns corrected stack and the per-frame shifts (dy, dx).
    """
    T, Z, Y, X = stack_TZYX.shape
    # Build reference from median of a subset of frames (using Z-MIPs)
    idx = np.arange(0, T, max(1, MC_REF_STRIDE))
    ref = np.median(stack_TZYX[idx].max(axis=1), axis=0).astype(np.float32)  # (Y,X)

    corrected = np.empty_like(stack_TZYX, dtype=np.float32)
    shifts = np.zeros((T, 2), dtype=np.float32)

    for t in range(T):
        mov = stack_TZYX[t].max(axis=0).astype(np.float32)  # (Y,X)
        shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
        dy, dx = -shift[0], -shift[1]  # skimage returns (y,x) shift to apply to mov -> ref
        shifts[t] = (dy, dx)
        # Apply to all Z slices with a single Affine (translation)
        tf = AffineTransform(translation=(dx, dy))
        for z in range(Z):
            corrected[t, z] = warp(stack_TZYX[t, z], tf, order=1, preserve_range=True).astype(np.float32)

    return corrected, shifts

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
    # ---- LOAD raw ----
    print("Loading stack...")
    stack = tifffile.imread(RAW_STACK_PATH).astype(np.float32)  # (T,Z,Y,X)
    if Y_CROP > 0:
        stack = stack[:, :, :-Y_CROP, :]
    T, Z, Y, X = stack.shape
    print(f"Shape: {stack.shape} (T,Z,Y,X)")

    # ---- optional motion correction ----
    if MOTION_CORRECT:
        print("Rigid XY motion correction...")
        stack, shifts = rigid_xy_motion_correct(stack)
        np.save(PREPROCESSED / "motion_shifts.npy", shifts)
        print("Motion correction done.")
    else:
        print("Skipping motion correction.")

    # ---- ΔF/F (per-voxel baseline) ----
    print("Computing ΔF/F...")
    F0 = np.percentile(stack, F0_PERCENTILE, axis=0)       # (Z,Y,X)
    dff = (stack - F0) / (F0 + 1e-6)
    del stack, F0
    gc.collect()

    # ---- gentle anisotropic smoothing (no temporal blur) ----
    print("Smoothing (no temporal blur)...")
    dff = gaussian_filter(dff, sigma=GAUSS_SIGMA)

    # ---- tissue mask (2D) ----
    print("Building tissue mask...")
    tissue2d = build_tissue_mask(dff)
    np.save(TISSUE_MASK_PATH, tissue2d)

    # ---- robust per-frame event score: mean of top K% ΔF/F voxels ----
    print("Computing robust frame scores...")
    K = max(1, int(TOP_FRAC * np.count_nonzero(tissue2d)))
    scores = np.zeros(T, dtype=np.float32)
    mask3d = np.repeat(tissue2d[None, :, :], Z, axis=0)
    flat_idx = np.where(mask3d.ravel())[0]  # index into (Z*Y*X)

    for t in range(T):
        vol = dff[t].reshape(-1)
        vals = vol[flat_idx]
        if vals.size == 0:
            scores[t] = 0.0
        else:
            # mean of top-K values (robust and motion-safe)
            topk = np.partition(vals, -K)[-K:]
            scores[t] = float(np.mean(topk))

    # ---- detrend scores with rolling baseline + MAD z-score ----
    print("Detrending score and detecting events...")
    base = rolling_baseline_percentile(scores, BASELINE_WINDOW, pct=F0_PERCENTILE)
    resid = (scores - base)
    mad = rolling_mad(resid, NOISE_WINDOW) + 1e-6
    z = resid / (1.4826 * mad)

    # hysteresis on z
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

    # group into events
    groups = group_consecutive(active, gap=MAX_FRAME_GAP)
    print(f"Detected {len(active)} active frames, grouped into {len(groups)} events.")

    # ---- timeline plot ----
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

    # ---- summarize & save events + crops (ΔF/F) + 2D backgrounds ----
    print("Summarizing and saving event crops...")
    rows = []; peak_frames = []
    T_, Z_, Y_, X_ = dff.shape

    for i, g in enumerate(groups):
        t_start = max(0, g[0] - CROP_RADIUS)
        t_end   = min(T_, g[-1] + CROP_RADIUS + 1)

        # peak frame by z-score within this event
        gi = np.array(g, dtype=int)
        local_peak_idx = int(np.argmax(z[gi]))
        peak_frame = int(gi[local_peak_idx])

        # Save ΔF/F crop for Module 2 (don’t over-normalize)
        crop = dff[t_start:t_end].astype(np.float32)
        out_name = f"event_group_{i:04d}_peak{peak_frame}.tif"
        tifffile.imwrite(EVENT_CROPS / out_name, crop)

        # Save lightweight 2D background (Z-MIP of time-max in the same window)
        vol_tmax = np.max(crop, axis=0)                      # (Z,Y,X)
        bg2d = np.max(vol_tmax, axis=0).astype(np.float16)   # (Y,X)
        tifffile.imwrite(EVENT_CROPS_BG / out_name.replace(".tif", "_bg.tif"), bg2d, dtype=np.float16)

        rows.append({"event_id": i, "t_start": int(t_start), "t_end": int(t_end-1),
                     "peak_frame": int(peak_frame), "score_peak": float(scores[peak_frame])})
        peak_frames.append(peak_frame)

    # Save ΔF/F stack (optional but helpful for later QC)
    print("Saving ΔF/F stack (float32)...")
    tifffile.imwrite(DFF_STACK_PATH, dff.astype(np.float32))

    # Save event tables
    with open(EVENTS_CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", "t_start", "t_end", "peak_frame", "score_peak"])
        w.writeheader(); w.writerows(rows)
    np.save(EVENTS_PEAKS_NPY, np.array(peak_frames, dtype=np.int32))

    # cleanup
    del dff; gc.collect()

    # Quick stdout summary
    if rows:
        preview = ", ".join([f"#{r['event_id']}@{r['peak_frame']}" for r in rows[:10]])
        print(f"Events (first 10): {preview} ... total {len(rows)}.")
    else:
        print("No events detected.")

    print("Module 1 complete (ΔF/F + robust events).")

if __name__ == "__main__":
    main()
    