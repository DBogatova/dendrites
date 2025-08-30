#!/usr/bin/env python
"""
Module 5C (updated): Compare Ca and ACh with/without background subtraction,
while automatically excluding cropped/black regions in the ACh channel.

Adds:
- detect_valid_region(): builds a boolean mask of ACh pixels that have non-trivial
  variance across time (not all-zero / dead tiles).
- All ACh extractions intersect masks with this validity mask.
"""

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d, binary_erosion, binary_dilation
from skimage.morphology import ball
import pandas as pd
import gc

# ---------- style ----------
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.default'] = 'regular'

# ---------- CONFIG ----------
DATE = "2025-08-27"
MOUSE = "rAi162_18"
RUN   = "run7"

FRAME_RATE   = 5.0        # Hz
Y_CROP       = 3          # crop last Y rows to match earlier modules
SMOOTH_SIGMA = 0.5        # frames, gentle smoothing for display

# Shell params (same spirit as your Module 5)
ERODE_RAD    = 1          # ball(1) for 'core' (fallback to mask if empty)
DILATE_RAD   = 3          # ball(3) for shell radius
MIN_SHELL_FRAC = 0.001    # if shell has <0.1% of voxels, treat as empty

# Dead-region detection for ACh
ACH_STD_TOL  = 1e-7       # std threshold across time (float stacks)
ACH_MIN_MAX  = 1e-12      # also demand max>min by this tiny margin (guards NaNs/const)

# ---------- PATHS ----------
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE         = PROJECT_ROOT / "data" / DATE / MOUSE / RUN

RAW_CA_PATH  = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green_reslice.tif"
RAW_ACH_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_red_reslice.tif"

MASK_FOLDER  = BASE / "labelmaps_curated_dynamic"

OUT_DIR      = BASE / "traces_compare_bg"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR      = OUT_DIR / "per_mask_plots"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_WIDE     = OUT_DIR / "compare_bg_traces_all_masks.csv"

# ---------- helpers ----------
def dff_stack_from_raw(raw):
    T = raw.shape[0]
    n0 = max(1, int(T * 0.20))
    F0 = np.mean(raw[:n0], axis=0)
    return (raw - F0[None]) / (F0[None] + 1e-6)

def ensure_mask_shape(mask, target_shape):
    """Pad/crop only Y to match (Z,Y,X)."""
    Zt, Yt, Xt = target_shape
    Zm, Ym, Xm = mask.shape
    m = mask
    if Ym < Yt:
        m = np.pad(m, ((0,0),(0, Yt - Ym),(0,0)), mode='constant')
    elif Ym > Yt:
        m = m[:, :Yt, :]
    return m

def core_shell(mask_bool):
    """Core via erosion; shell via dilation minus mask."""
    core = binary_erosion(mask_bool, structure=ball(ERODE_RAD))
    if not core.any():
        core = mask_bool.copy()
    dil  = binary_dilation(mask_bool, structure=ball(DILATE_RAD))
    shell = dil & (~mask_bool)
    if shell.sum() < MIN_SHELL_FRAC * mask_bool.size:
        shell = np.zeros_like(mask_bool, bool)
    return core, shell

def mean_over(mask_bool, stack_tzyx):
    """Mean over voxels in mask for each frame."""
    if not mask_bool.any():
        return np.zeros(stack_tzyx.shape[0], np.float32)
    return stack_tzyx[:, mask_bool].mean(axis=1)

def matched_limits(y1, y2, pad=0.05):
    lo = float(np.nanmin([y1.min(), y2.min()]))
    hi = float(np.nanmax([y1.max(), y2.max()]))
    span = hi - lo
    if span <= 0:
        span = 1.0
    lo -= pad*span
    hi += pad*span
    return lo, hi

def detect_valid_region(stack_tzyx, std_tol=ACH_STD_TOL, min_max=ACH_MIN_MAX):
    """
    Returns boolean (Z,Y,X) True = valid ACh voxels.
    Flags as invalid those with ~zero temporal variance or flat (max≈min).
    """
    # compute per-voxel std & min/max across T
    std = stack_tzyx.std(axis=0)
    mn  = stack_tzyx.min(axis=0)
    mx  = stack_tzyx.max(axis=0)
    valid = (std > std_tol) & ((mx - mn) > min_max) & np.isfinite(std) & np.isfinite(mn) & np.isfinite(mx)
    return valid

# ---------- main ----------
def main():
    # shapes from CA
    with tifffile.TiffFile(RAW_CA_PATH) as tif:
        T, Z, Y, X = tif.series[0].shape
    Y_eff = Y - Y_CROP if Y_CROP > 0 else Y
    t_sec = np.arange(T) / FRAME_RATE
    print(f"Movie shape: T={T}, Z={Z}, Y={Y_eff}, X={X}")

    # load stacks, crop Y
    print("Loading stacks...")
    ca_raw  = tifffile.imread(RAW_CA_PATH).astype(np.float32)
    ach_raw = tifffile.imread(RAW_ACH_PATH).astype(np.float32)
    if Y_CROP > 0:
        ca_raw  = ca_raw[:, :, :-Y_CROP, :]
        ach_raw = ach_raw[:, :, :-Y_CROP, :]

    assert ca_raw.shape == ach_raw.shape

    # === build ACh valid-region mask (exclude cropped black) ===
    print("Detecting dead/invalid pixels in ACh...")
    ach_valid = detect_valid_region(ach_raw)   # (Z,Y_eff,X)
    n_valid = int(ach_valid.sum())
    frac = n_valid / ach_valid.size
    print(f"ACh valid voxels: {n_valid} ({frac*100:.1f}% of volume)")

    # ΔF/F
    print("Computing ΔF/F...")
    ca_dff  = dff_stack_from_raw(ca_raw);  del ca_raw
    ach_dff = dff_stack_from_raw(ach_raw); del ach_raw
    gc.collect()

    # load curated masks
    mask_paths = sorted(MASK_FOLDER.glob("dend_*.tif"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks in {MASK_FOLDER}")

    # wide CSV buffers
    wide = {"time_s": t_sec}

    # iterate masks
    for p in mask_paths:
        name = p.stem.replace("_labelmap", "")
        m = (tifffile.imread(p).astype(np.uint16) > 0)
        if m.shape != (Z, Y_eff, X):
            m = ensure_mask_shape(m, (Z, Y_eff, X))
        if not m.any():
            print(f"Skip empty: {name}")
            continue

        core, shell = core_shell(m)

        # ---------------- Ca traces (unchanged) ----------------
        ca_core   = mean_over(core,  ca_dff)
        ca_shell  = mean_over(shell, ca_dff) if shell.any() else 0.0
        ca_core_only   = gaussian_filter1d(ca_core,   SMOOTH_SIGMA) * 100.0
        ca_core_minus  = gaussian_filter1d(ca_core - (ca_shell if np.ndim(ca_shell) else 0.0),
                                           SMOOTH_SIGMA) * 100.0

        # ---------------- ACh traces (APPLY VALID MASK) ----------------
        # Intersect core/shell with ach_valid so dead/cropped region never contributes
        core_ach  = core & ach_valid                       # [APPLY VALID ACh MASK]
        shell_ach = (shell & ach_valid) if shell.any() else shell  # [APPLY VALID ACh MASK]

        # If intersection empties the ROI, skip or fall back gracefully
        if not core_ach.any():
            print(f"Skip {name}: ACh core overlaps only dead region.")
            continue

        ach_core  = mean_over(core_ach,  ach_dff)
        ach_shell = mean_over(shell_ach, ach_dff) if shell_ach.any() else 0.0

        ach_core_only  = gaussian_filter1d(ach_core,  SMOOTH_SIGMA) * 100.0
        ach_core_minus = gaussian_filter1d(ach_core - (ach_shell if np.ndim(ach_shell) else 0.0),
                                           SMOOTH_SIGMA) * 100.0

        # per-mask CSV
        df = pd.DataFrame({
            "time_s": t_sec,
            "Ca_core_only_pct":   ca_core_only,
            "Ca_core_minus_shell_pct": ca_core_minus,
            "ACh_core_only_pct":  ach_core_only,
            "ACh_core_minus_shell_pct": ach_core_minus,
        })
        csv_path = OUT_DIR / f"{name}_compare_bg.csv"
        df.to_csv(csv_path, index=False)

        # 2×2 figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
        (ax11, ax12), (ax21, ax22) = axes

        # Ca panels (match y-lims)
        ylo, yhi = matched_limits(ca_core_only, ca_core_minus)
        ax11.plot(t_sec, ca_core_only,  color=(0.2,0.7,0.2), lw=1.5)
        ax11.set_title(f"{name} • Ca core only")
        ax11.set_ylabel("Ca ΔF/F (%)")
        ax11.set_ylim(ylo, yhi); ax11.grid(alpha=0.3)

        ax12.plot(t_sec, ca_core_minus, color=(0.1,0.55,0.1), lw=1.5)
        ax12.set_title("Ca core − shell")
        ax12.set_ylim(ylo, yhi); ax12.grid(alpha=0.3)

        # ACh panels (match y-lims)
        ylo2, yhi2 = matched_limits(ach_core_only, ach_core_minus)
        ax21.plot(t_sec, ach_core_only,  color=(0.85,0.2,0.2), lw=1.5)
        ax21.set_title("ACh core only (valid region)")
        ax21.set_xlabel("Time (s)")
        ax21.set_ylabel("ACh ΔF/F (%)")
        ax21.set_ylim(ylo2, yhi2); ax21.grid(alpha=0.3)

        ax22.plot(t_sec, ach_core_minus, color=(0.6,0.1,0.1), lw=1.5)
        ax22.set_title("ACh core − shell (valid region)")
        ax22.set_xlabel("Time (s)")
        ax22.set_ylim(ylo2, yhi2); ax22.grid(alpha=0.3)

        fig.suptitle(f"{name}: Ca vs ACh • with/without background (ACh dead region excluded)",
                     y=1.02, fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name}_CaACh_compare_bg.pdf", format="pdf", dpi=150)
        plt.close(fig)

        # wide CSV add
        wide[f"{name}__Ca_core_only_pct"]        = ca_core_only
        wide[f"{name}__Ca_core_minus_shell_pct"] = ca_core_minus
        wide[f"{name}__ACh_core_only_pct"]       = ach_core_only
        wide[f"{name}__ACh_core_minus_shell_pct"]= ach_core_minus

    # write wide CSV
    pd.DataFrame(wide).to_csv(CSV_WIDE, index=False)
    print(f"✅ Wrote per-mask plots to: {FIG_DIR}")
    print(f"✅ Combined CSV: {CSV_WIDE}")
    print("Done.")

if __name__ == "__main__":
    main()
