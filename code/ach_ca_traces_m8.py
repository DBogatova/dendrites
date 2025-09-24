#!/usr/bin/env python
"""
Module 5D: Ca with BG subtraction (core - shell) + ACh neighborhood (no subtraction)

For each curated 3D mask:
- Load Ca (green) and ACh (red) stacks, compute ΔF/F (F0 = first 20%).
- Ca trace: core - shell, %ΔF/F, smoothed.
- ACh traces: core, near ring, far ring (no subtraction), %ΔF/F, smoothed.
- Exclude dead/cropped red pixels (≈zero temporal variance).
- Optional: exclude other dendrite cores from ACh rings to avoid contamination.
- Save per-mask 2-row PDF and per-mask CSV; also write a wide CSV.

Rings use true μm distances with anisotropic voxel sizes.
"""

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import (
    gaussian_filter1d,
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
)
from skimage.morphology import ball
import pandas as pd
import gc

# --------- Matplotlib fixes (minus sign etc.) ----------
#mpl.rcParams['font.family'] = 'CMU Serif'
#mpl.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['mathtext.default'] = 'regular'

# ========= CONFIG =========
DATE = "2025-08-27"
MOUSE = "rAi162_18"
RUN   = "run7"

FRAME_RATE   = 10.0    # Hz
Y_CROP       = 3
SMOOTH_SIGMA = 0.5    # frames

# Voxel sizes (μm) for accurate 3D distances
VOXEL_SIZE_UM = (9.4, 1.0, 1.2)  # (Z, Y, X)

# ACh neighborhood radii in μm
R1_MIN_UM, R1_MAX_UM = 2.0, 6.0     # near ring
R2_MIN_UM, R2_MAX_UM = 6.0, 12.0    # far ring

# Exclude other dendrite cores from ACh rings?
EXCLUDE_OTHER_MASKS_IN_RINGS = True

# Ca core/shell morphology
ERODE_RAD  = 1   # ball(1) for core
DILATE_RAD = 3   # ball(3) for shell radius

# Tiny shells are useless → treat as empty
MIN_SHELL_FRAC = 0.001

# Dead-region detection (ACh/red)
ACH_STD_TOL = 1e-7
ACH_MIN_MAX = 1e-12

# ========= PATHS =========
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "data" / DATE / MOUSE / RUN

RAW_CA_PATH  = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green_reslice.tif"
RAW_ACH_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_red_reslice.tif"

MASK_FOLDER  = BASE / "labelmaps_curated_dynamic"

OUT_DIR  = BASE / "traces_ca_ach"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR  = OUT_DIR / "per_mask_plots"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_WIDE = OUT_DIR / "traces_ca_ach_all_masks.csv"

# ========= Helpers =========
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
        m = np.pad(m, ((0,0), (0, Yt - Ym), (0,0)), mode="constant")
    elif Ym > Yt:
        m = m[:, :Yt, :]
    return m

def core_shell(mask_bool):
    core = binary_erosion(mask_bool, structure=ball(ERODE_RAD))
    if not core.any():
        core = mask_bool.copy()
    dil  = binary_dilation(mask_bool, structure=ball(DILATE_RAD))
    shell = dil & (~mask_bool)
    if shell.sum() < MIN_SHELL_FRAC * mask_bool.size:
        shell = np.zeros_like(mask_bool, bool)
    return core, shell

def mean_over(mask_bool, stack_tzyx):
    if not mask_bool.any():
        return np.zeros(stack_tzyx.shape[0], np.float32)
    return stack_tzyx[:, mask_bool].mean(axis=1)

def rings_from_mask(mask_bool, vs_zyx, r1, r2):
    """ACh rings based on physical distance (μm)."""
    outside = ~mask_bool
    dist_um = distance_transform_edt(outside, sampling=vs_zyx)
    ring1 = (dist_um >= r1[0]) & (dist_um < r1[1]) & outside
    ring2 = (dist_um >= r2[0]) & (dist_um < r2[1]) & outside
    return ring1, ring2

def detect_valid_region(stack_tzyx, std_tol=ACH_STD_TOL, min_max=ACH_MIN_MAX):
    std = stack_tzyx.std(axis=0)
    mn  = stack_tzyx.min(axis=0)
    mx  = stack_tzyx.max(axis=0)
    valid = (
        (std > std_tol)
        & ((mx - mn) > min_max)
        & np.isfinite(std) & np.isfinite(mn) & np.isfinite(mx)
    )
    return valid

# ========= Main =========
def main():
    # Get shapes
    with tifffile.TiffFile(RAW_CA_PATH) as tif:
        T, Z, Y, X = tif.series[0].shape
    Y_eff = Y - Y_CROP if Y_CROP > 0 else Y
    t_sec = np.arange(T) / FRAME_RATE
    print(f"Movie: T={T}, Z={Z}, Y={Y_eff}, X={X}")

    # Load stacks and crop
    print("Loading stacks…")
    ca_raw  = tifffile.imread(RAW_CA_PATH).astype(np.float32)
    ach_raw = tifffile.imread(RAW_ACH_PATH).astype(np.float32)
    if Y_CROP > 0:
        ca_raw  = ca_raw[:, :, :-Y_CROP, :]
        ach_raw = ach_raw[:, :, :-Y_CROP, :]

    assert ca_raw.shape == ach_raw.shape

    # Detect dead region in ACh
    print("Detecting dead/black pixels in ACh…")
    ach_valid = detect_valid_region(ach_raw)
    print(f"ACh valid voxels: {ach_valid.sum()} / {ach_valid.size} ({ach_valid.mean()*100:.1f}%)")

    # ΔF/F stacks
    print("Computing ΔF/F…")
    ca_dff  = dff_stack_from_raw(ca_raw);  del ca_raw
    ach_dff = dff_stack_from_raw(ach_raw); del ach_raw
    gc.collect()

    # Load curated masks
    mask_paths = sorted(MASK_FOLDER.glob("dend_*.tif"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks in {MASK_FOLDER}")
    masks = []
    names = []
    for p in mask_paths:
        name = p.stem.replace("_labelmap", "")
        m = (tifffile.imread(p).astype(np.uint16) > 0)
        if m.shape != (Z, Y_eff, X):
            m = ensure_mask_shape(m, (Z, Y_eff, X))
        if m.any():
            masks.append(m)
            names.append(name)

    # Union of cores (for ACh ring contamination control)
    union_cores = None
    if EXCLUDE_OTHER_MASKS_IN_RINGS:
        union_cores = np.zeros((Z, Y_eff, X), bool)
        for m in masks:
            union_cores |= binary_erosion(m, structure=ball(ERODE_RAD)) | m  # robust core union

    # Wide CSV buffers
    wide = {"time_s": t_sec}

    # Colors
    col_ca  = (0.2, 0.7, 0.2)
    col_core= (0.85, 0.1, 0.1)
    col_r1  = (0.75, 0.15, 0.15)
    col_r2  = (0.60, 0.20, 0.20)

    # Process each mask
    for name, m in zip(names, masks):
        # ---- Ca core − shell ----
        core_ca, shell_ca = core_shell(m)
        ca_core  = mean_over(core_ca,  ca_dff)
        ca_shell = mean_over(shell_ca, ca_dff) if shell_ca.any() else 0.0
        ca_bgsub = gaussian_filter1d(ca_core - (ca_shell if np.ndim(ca_shell) else 0.0),
                                     SMOOTH_SIGMA) * 100.0

        # ---- ACh neighborhood (no subtraction) ----
        # Build rings in μm
        r1_mask, r2_mask = rings_from_mask(m, VOXEL_SIZE_UM, (R1_MIN_UM, R1_MAX_UM), (R2_MIN_UM, R2_MAX_UM))

        # Exclude other dendrite cores from rings (optional)
        if EXCLUDE_OTHER_MASKS_IN_RINGS and union_cores is not None:
            others = union_cores & (~m)
            r1_mask &= ~others
            r2_mask &= ~others

        # Intersect with valid ACh region to avoid dead pixels
        core_ach = m & ach_valid
        r1_mask &= ach_valid
        r2_mask &= ach_valid

        if not core_ach.any():
            print(f"Skip {name}: ACh core falls in dead region.")
            continue

        ach_core = mean_over(core_ach, ach_dff)
        ach_r1   = mean_over(r1_mask,   ach_dff) if r1_mask.any() else np.zeros(T, np.float32)
        ach_r2   = mean_over(r2_mask,   ach_dff) if r2_mask.any() else np.zeros(T, np.float32)

        ach_core_pct = gaussian_filter1d(ach_core, SMOOTH_SIGMA) * 100.0
        ach_r1_pct   = gaussian_filter1d(ach_r1,   SMOOTH_SIGMA) * 100.0
        ach_r2_pct   = gaussian_filter1d(ach_r2,   SMOOTH_SIGMA) * 100.0

        # ---- Save per-mask CSV ----
        df = pd.DataFrame({
            "time_s": t_sec,
            "Ca_bgsub_pct": ca_bgsub,
            "ACh_core_pct": ach_core_pct,
            "ACh_ring1_pct": ach_r1_pct,
            "ACh_ring2_pct": ach_r2_pct,
        })
        df.to_csv(OUT_DIR / f"{name}_CaBG_AChNeighborhood.csv", index=False)

        # ---- Plot (2 rows) ----
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        axes[0].plot(t_sec, ca_bgsub, color=col_ca, lw=1.6)
        axes[0].set_title(f"{name}: Ca (core − shell)  •  ACh core/near/far")
        axes[0].set_ylabel("Ca ΔF/F (%)")
        axes[0].grid(alpha=0.3)

        axes[1].plot(t_sec, ach_core_pct, color=col_core, lw=1.4, label="ACh core")
        axes[1].plot(t_sec, ach_r1_pct,   color=col_r1,   lw=1.2, label=f"ACh {R1_MIN_UM}-{R1_MAX_UM} μm")
        axes[1].plot(t_sec, ach_r2_pct,   color=col_r2,   lw=1.0, label=f"ACh {R2_MIN_UM}-{R2_MAX_UM} μm")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("ACh ΔF/F (%)")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8, frameon=False)

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name}_CaBG_AChNeighborhood.pdf", format="pdf")
        plt.close(fig)

        # Wide CSV columns
        wide[f"{name}__Ca_bgsub_pct"] = ca_bgsub
        wide[f"{name}__ACh_core_pct"] = ach_core_pct
        wide[f"{name}__ACh_ring1_pct"] = ach_r1_pct
        wide[f"{name}__ACh_ring2_pct"] = ach_r2_pct

    # ---- Save combined CSV ----
    pd.DataFrame(wide).to_csv(CSV_WIDE, index=False)
    print(f"Per-mask plots → {FIG_DIR}")
    print(f"Combined CSV → {CSV_WIDE}")
    print("Done.")

if __name__ == "__main__":
    main()
