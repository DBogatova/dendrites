#!/usr/bin/env python
"""
Module 4 — Trace extraction with correct neuropil handling.

Pipeline per ROI:
  1) core_raw(t)  = mean/brightest-fraction inside eroded ROI on RAW stack
  2) np_raw(t)    = mean in a ring around ROI (other ROIs excluded)
     -> np_lp(t)  = low-pass filtered neuropil (slow/common component)
  3) residual d(t)= core_raw(t) - α * np_lp(t)          (α≈0.3–0.5 or auto-fit)
  4) baseline B(t)= rolling 20th percentile (long window, handles bleaching)
  5) dFF(t)       = d(t)/B(t) - 1
  6) light smoothing for plotting; save traces, PDFs, CSV, PKL
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d, binary_erosion, binary_dilation, percentile_filter
from skimage.morphology import ball
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import gc

# ====================== CONFIG ======================
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False

DATE = "2025-08-06"
MOUSE = "organoid"
RUN = "run4"

FRAME_RATE = 5               # Hz
Y_CROP = 3                   # crop from bottom of Y

# Neuropil & baseline
ALPHA_METHOD = "fixed"       # "fixed" or "auto"
ALPHA_FIXED   = 0.4          # used if ALPHA_METHOD == "fixed"
NEUROPIL_RING = 4            # ring dilation (vox)
NEIGHBOR_PAD  = 1            # exclude other ROIs with a 1-voxel pad
NEUROPIL_LPF_SEC = 0.75      # low-pass neuropil (Gaussian sigma in seconds)

BASELINE_PERCENTILE = 20     # rolling baseline percentile
BASELINE_WIN_SEC    = 90     # window seconds (60–120 s typical for bleaching at 5 Hz)

# ROI readout (motion-robust)
ROI_CORE_ERODE  = 1          # erode ROI for core
USE_BRIGHTEST   = True       # use brightest fraction inside core (motion-robust)
BRIGHT_FRAC     = 0.20       # top 20% voxels per frame

# Plotting / output
PLOT_ALL_TRACES = False
SELECTED_NAMES = ["dend_002","dend_006","dend_007","dend_008","dend_010","dend_011","dend_024"]
SMOOTH_SIGMA = 0.5           # frames, for display only
STACK_OFFSET = 6             # ΔF/F % offset between stacked traces
MAKE_PER_CELL_PDFS = True

# Paths
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "data" / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runB_{RUN}_reslice.tif"
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
TRACE_FOLDER = BASE / "traces"; TRACE_FOLDER.mkdir(exist_ok=True)
TRACE_PKL = TRACE_FOLDER / "dff_traces_curated.pkl"
TRACE_CSV = TRACE_FOLDER / "dff_traces_curated.csv"
PREVIEW_FOLDER = BASE / "trace_previews_curated"; PREVIEW_FOLDER.mkdir(exist_ok=True)
COMBO_PDF = PREVIEW_FOLDER / "combo_traces_curated.pdf"

# ====================== HELPERS ======================
def load_raw_stack(path, y_crop):
    raw = tifffile.imread(path).astype(np.float32)
    if y_crop > 0:
        raw = raw[:, :, :-y_crop, :]
    return raw

def build_mask_list(mask_folder):
    paths = sorted(mask_folder.glob("dend_*.tif"))
    masks = []
    names = []
    for p in paths:
        m = tifffile.imread(p).astype(bool)
        if np.any(m):
            masks.append(m)
            names.append(p.stem.replace("_labelmap",""))
    return masks, names

def union_of_masks(masks, pad=0):
    if not masks:
        return None
    Z, Y, X = masks[0].shape
    uni = np.zeros((Z, Y, X), dtype=bool)
    for m in masks:
        if pad > 0:
            uni |= binary_dilation(m, structure=ball(pad))
        else:
            uni |= m
    return uni

def roi_core_mask(mask, erode=1):
    if erode > 0:
        core = binary_erosion(mask, structure=ball(erode))
        if np.any(core):
            return core
    return mask.copy()

def neuropil_shell(mask, union_all, ring=4, neighbor_pad=1):
    """Ring around ROI, excluding ROI and other ROIs (with padding)."""
    ring_mask = binary_dilation(mask, structure=ball(ring))
    exclude = binary_dilation(union_all, structure=ball(neighbor_pad))
    shell = ring_mask & ~exclude
    # Fallback: if shell empty, allow ring minus ROI itself
    if not np.any(shell):
        shell = ring_mask & ~mask
    return shell

def mean_of_brightest(chunk_2d, frac=0.2):
    """chunk_2d: (frames, voxels). Return per-frame mean of top frac."""
    if chunk_2d.shape[1] == 0:
        return np.zeros(chunk_2d.shape[0], dtype=np.float32)
    k = max(1, int(round(frac * chunk_2d.shape[1])))
    # partition to get top-k without full sort
    part = np.partition(chunk_2d, -k, axis=1)[:, -k:]
    return part.mean(axis=1)

def extract_timecourse_vectorized(raw_stack, mask, use_brightest=True, bright_frac=0.2):
    """Vectorized time course extraction - much faster."""
    if not np.any(mask):
        return np.zeros(raw_stack.shape[0], dtype=np.float32)
    
    # Direct boolean indexing - much faster than reshape
    masked_data = raw_stack[:, mask]  # Shape: (T, n_voxels)
    
    if use_brightest:
        k = max(1, int(round(bright_frac * masked_data.shape[1])))
        # Use argpartition for speed - only partial sort
        top_k_indices = np.argpartition(masked_data, -k, axis=1)[:, -k:]
        tc = np.mean(np.take_along_axis(masked_data, top_k_indices, axis=1), axis=1)
    else:
        tc = masked_data.mean(axis=1)
    
    return tc.astype(np.float32)

def rolling_percentile_fast(x, q=20, win_sec=90, fs=5):
    """Fast rolling percentile using pandas-style rolling window."""
    import pandas as pd
    size = max(3, int(round(win_sec * fs)))
    # Use pandas for fast rolling percentile
    s = pd.Series(x)
    return s.rolling(window=size, center=True, min_periods=1).quantile(q/100).values.astype(np.float32)

def auto_alpha(core, neuropil, cap=(0.0, 0.8)):
    """OLS slope on slow components to set α; robustly clipped."""
    c_lp = gaussian_filter1d(core, sigma=FRAME_RATE)     # ~1 s
    n_lp = gaussian_filter1d(neuropil, sigma=FRAME_RATE) # ~1 s
    var = np.var(n_lp)
    if var < 1e-12:
        return ALPHA_FIXED
    alpha = np.cov(c_lp, n_lp)[0,1] / var
    return float(np.clip(alpha, *cap))

def make_cell_preview(mip, t, trace_pct, outpath, title_id):
    fig, (ax_img, ax_trace) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.2, 2]}
    )
    # MIP
    rgb = np.zeros((*mip.shape, 3), dtype=np.float32)
    rgb[..., 1] = (mip > 0) * 0.7
    ax_img.imshow(rgb)
    ax_img.set_title(f"Cell {title_id} — MIP")
    ax_img.axis("off")

    # Trace
    ax_trace.plot(t, trace_pct, color='teal', lw=1.5)
    ax_trace.set_title("ΔF/F Trace (%)")
    ax_trace.set_xlabel("Time (s)"); ax_trace.set_ylabel("ΔF/F (%)")
    ax_trace.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, format='pdf')
    plt.close(fig)

# ====================== MAIN ======================
def main():
    print("Loading RAW stack…")
    raw = load_raw_stack(RAW_STACK_PATH, Y_CROP)
    T, Z, Y, X = raw.shape
    print(f"RAW shape: {raw.shape}  ({raw.nbytes/1e9:.2f} GB)")

    print("Loading masks…")
    masks, names = build_mask_list(MASK_FOLDER)
    if not masks:
        print("No masks found."); return
    union_all = union_of_masks(masks, pad=0)

    traces = []
    labels = []

    np_sigma_frames = max(1, int(round(NEUROPIL_LPF_SEC * FRAME_RATE)))
    print(f"Neuropil LPF sigma: {np_sigma_frames} frames")
    base_win = BASELINE_WIN_SEC
    print(f"Baseline: {BASELINE_PERCENTILE}th pct over {base_win}s")

    for i, (mask, name) in enumerate(tqdm(list(zip(masks, names)), desc="ROIs")):
        # Masks for ROI
        core_mask = roi_core_mask(mask, erode=ROI_CORE_ERODE)
        shell_mask = neuropil_shell(mask, union_all, ring=NEUROPIL_RING, neighbor_pad=NEIGHBOR_PAD)

        # Time courses from RAW (vectorized - much faster)
        core_raw = extract_timecourse_vectorized(raw, core_mask, USE_BRIGHTEST, BRIGHT_FRAC)
        if np.any(shell_mask):
            np_raw = extract_timecourse_vectorized(raw, shell_mask, use_brightest=False)
        else:
            np_raw = np.zeros_like(core_raw)

        # Low-pass neuropil (slow/common only)
        np_lp = gaussian_filter1d(np_raw, sigma=np_sigma_frames)

        # α (fixed or auto)
        alpha = ALPHA_FIXED if ALPHA_METHOD=="fixed" else auto_alpha(core_raw, np_lp)
        # Residual
        d = core_raw - alpha * np_lp

        # Rolling baseline (handles bleaching) - fast version
        B = rolling_percentile_fast(d, q=BASELINE_PERCENTILE, win_sec=BASELINE_WIN_SEC, fs=FRAME_RATE)
        B = np.maximum(B, 1e-6)

        # ΔF/F
        dff = d / B - 1.0
        trace_pct = gaussian_filter1d(dff, sigma=SMOOTH_SIGMA) * 100.0  # for display

        traces.append(trace_pct.astype(np.float32))
        labels.append(name)

        # Per-cell PDF
        if MAKE_PER_CELL_PDFS:
            mip = np.max(mask, axis=0)
            t_axis = np.arange(T) / FRAME_RATE
            cell_id = name.split('_')[-1]
            make_cell_preview(mip, t_axis, trace_pct, PREVIEW_FOLDER / f"{name}_trace_mip.pdf", cell_id)

    # Save PKL + CSV
    print("Saving traces…")
    with open(TRACE_PKL, "wb") as f:
        pickle.dump(list(zip(labels, traces)), f)
    df = pd.DataFrame(np.array(traces).T, columns=labels)
    df.to_csv(TRACE_CSV, index_label="Frame")
    print(f"Traces saved:\n  {TRACE_PKL}\n  {TRACE_CSV}")

    # Stacked combo plot
    print("Generating combo plot…")
    fig, ax = plt.subplots(figsize=(10, 8))
    t = np.arange(T) / FRAME_RATE
    count = 0
    for i, (tr, name) in enumerate(zip(traces, labels)):
        if not PLOT_ALL_TRACES and name not in SELECTED_NAMES:
            continue
        ax.plot(t, tr + count * STACK_OFFSET, color=cm.turbo(i / max(1,len(traces)-1)), lw=1)
        cell_number = name.split('_')[-1]
        ax.text(t[-1] + 1, count * STACK_OFFSET, f"Cell {cell_number}", va='center', fontsize=8)
        count += 1

    # scale bar
    if count > 0:
        scale_x = t[-1] - 7
        scale_y = STACK_OFFSET * (count - 0.5)
        ax.plot([scale_x, scale_x], [scale_y, scale_y + 2], color='k', lw=2)  # 2% bar
        ax.text(scale_x + 1, scale_y + 1, "2%", va='center', ha='left', fontsize=10)

    ax.set_xlim(0, t[-1]); ax.set_ylim(-STACK_OFFSET, count * STACK_OFFSET + 10)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ΔF/F (%) + offset")
    ax.set_yticks([]); ax.set_title("Stacked ΔF/F Traces")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(COMBO_PDF, format='pdf'); plt.close(fig)
    print(f"Combo plot saved to: {COMBO_PDF}")

    # cleanup
    del raw; gc.collect()
    print("Done.")

if __name__ == "__main__":
    main()
