#!/usr/bin/env python
"""
Motion correction for resliced stacks (T,Z,Y,X) — memory safe.

Default MODE="y1d": align *axial drift* mapped onto Y
  - Build reference: median of sparse Z-MIPs (every REF_STRIDE frames)
  - For each frame: Z-MIP → X-median → 1-D vertical profile
  - High-pass profiles, estimate dy via phase cross-correlation (subpixel)
  - Apply integer-pixel roll along Y, then tiny residual via warp (min blur)

ALT MODE "yx2d": classic 2D phase correlation on Z-MIPs (with high-pass).
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import tifffile, zarr
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform

# ---------- CONFIG ----------
DATE  = "2025-08-18"
MOUSE = "rAi162_15"
RUN   = "run9"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_TIF = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green_reslice.tif"
PRE     = BASE / "preprocessed"
OUT_TIF = PRE / "processed_raw_mc.tif"   # matches your screenshots
SHIFTS  = PRE / "motion_shifts.npy"
META    = PRE / "mc_meta.json"

# Strategy
MODE = "yx2d"          # "y1d" (recommended) or "yx2d"
REF_STRIDE = 10       # build reference from every Nth frame
SUBPIX = 20           # subpixel refinement
Y_CROP_BOTTOM = 3     # keep consistent with downstream
CHUNK_T = 128

# ---------- IO helpers ----------
def as_zarr(tf: tifffile.TiffFile):
    store = tf.series[0].aszarr()
    z = zarr.open(store, mode="r")
    return z if z.ndim == 4 else z[None, ...]

def crop_y_bottom(arr: np.ndarray) -> np.ndarray:
    if Y_CROP_BOTTOM <= 0: return arr
    if arr.shape[-2] <= Y_CROP_BOTTOM: raise ValueError("Y_CROP_BOTTOM too large")
    return arr[..., :-Y_CROP_BOTTOM, :]

def read_time_chunk(tf: tifffile.TiffFile, t0: int, t1: int) -> np.ndarray:
    z = as_zarr(tf)
    chunk = np.asarray(z[t0:t1]).astype(np.float32)   # (k,Z,Y,X)
    return crop_y_bottom(chunk)

# ---------- registration preproc ----------
def hp2d(img: np.ndarray, sigma=5.0) -> np.ndarray:
    """High-pass to remove low-freq brightness drift."""
    return img - gaussian_filter(img, sigma)

def build_reference_y1d(tf, T):
    """Median Z-MIPs → median across X → 1-D ref profile (Yc,)."""
    picks = np.arange(0, T, max(1, REF_STRIDE), dtype=int)
    profs = []
    for t in picks:
        fr = read_time_chunk(tf, t, t+1)[0]     # (Z,Yc,X)
        mip = np.percentile(fr, 90, axis=0)              # (Yc,X)
        profs.append(np.median(hp2d(mip), axis=1))  # (Yc,)
    ref = np.median(np.stack(profs, axis=0), axis=0).astype(np.float32)
    return ref

def build_reference_yx2d(tf, T):
    """Median of sparse Z-MIPs (2D)."""
    picks = np.arange(0, T, max(1, REF_STRIDE), dtype=int)
    mips = []
    for t in picks:
        fr = read_time_chunk(tf, t, t+1)[0]
        mips.append(hp2d(fr.max(axis=0)))
    ref = np.median(np.stack(mips, axis=0), axis=0).astype(np.float32)
    return ref

# ---------- estimate shifts ----------
def estimate_shifts_y1d(tf, T, ref_prof):
    """Return (T,2) dy,dx with dx=0 (Y-only)."""
    shifts = np.zeros((T, 2), dtype=np.float32)
    # frame 0 aligned to ref already -> 0 shift
    for t0 in range(0, T, CHUNK_T):
        t1 = min(T, t0 + CHUNK_T)
        chunk = read_time_chunk(tf, t0, t1)         # (k,Z,Yc,X)
        mips  = chunk.max(axis=1)                   # (k,Yc,X)
        for i in range(mips.shape[0]):
            prof = np.median(hp2d(mips[i]), axis=1)[:, None]  # (Yc,1) so PCC accepts it
            # phase correlation on 1D profile
            shift, _, _ = phase_cross_correlation(ref_prof[:, None], prof, upsample_factor=SUBPIX)
            dy = -float(shift[0])
            shifts[t0 + i] = (dy, 0.0)
    return shifts

def estimate_shifts_yx2d(tf, T, ref_img):
    shifts = np.zeros((T, 2), dtype=np.float32)
    for t0 in range(0, T, CHUNK_T):
        t1 = min(T, t0 + CHUNK_T)
        chunk = read_time_chunk(tf, t0, t1)
        mips  = chunk.max(axis=1)
        for i in range(mips.shape[0]):
            mov = hp2d(mips[i])
            shift, _, _ = phase_cross_correlation(ref_img, mov, upsample_factor=SUBPIX)
            dy, dx = -float(shift[0]), -float(shift[1])
            shifts[t0 + i] = (dy, dx)
    return shifts

# ---------- apply shifts ----------
def apply_shifts_inplace(chunk: np.ndarray, shifts: np.ndarray, t0: int, int_first=True):
    """Apply translation to (k,Z,Yc,X) in place. Integer Y roll first, then residual warp."""
    k, Z, Yc, X = chunk.shape
    for i in range(k):
        dy, dx = float(shifts[t0 + i, 0]), float(shifts[t0 + i, 1])
        ry, rx = dy, dx
        if int_first:
            iy = int(np.round(dy)); ry = dy - iy
            if iy != 0:
                chunk[i] = np.roll(chunk[i], shift=iy, axis=-2)  # Y axis
        if abs(ry) > 1e-6 or abs(rx) > 1e-6:
            tfm = AffineTransform(translation=(rx, ry))
            for z in range(Z):
                chunk[i, z] = warp(chunk[i, z], tfm, order=1, preserve_range=True).astype(np.float32)

# ---------- main ----------
def main():
    PRE.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(str(RAW_TIF)) as tf:
        z = as_zarr(tf)
        T, Z, Y, X = map(int, z.shape)
        Yc = Y - max(0, Y_CROP_BOTTOM)
        print(f"Input (T,Z,Y,X)=({T},{Z},{Y},{X})  → Y crop: {Y}->{Yc}")
        print(f"Mode: {MODE}")

        if MODE == "y1d":
            print("Building 1-D reference profile…")
            ref_prof = build_reference_y1d(tf, T)
            print("Estimating Y-only shifts…")
            shifts = estimate_shifts_y1d(tf, T, ref_prof)
        else:
            print("Building 2-D reference…")
            ref_img = build_reference_yx2d(tf, T)
            print("Estimating YX shifts…")
            shifts = estimate_shifts_yx2d(tf, T, ref_img)

        np.save(SHIFTS, shifts)
        print(f"Saved shifts → {SHIFTS}")

        print("Writing motion-corrected stack (chunked)…")
        first = True
        for t0 in range(0, T, CHUNK_T):
            t1 = min(T, t0 + CHUNK_T)
            chunk = read_time_chunk(tf, t0, t1)   # (k,Z,Yc,X)
            apply_shifts_inplace(chunk, shifts, t0, int_first=True)
            tifffile.imwrite(
                str(OUT_TIF),
                chunk.astype(np.float32),
                imagej=False,
                append=not first,
                bigtiff=True,
                compression='zlib'
            )
            first = False
            print(f"Wrote frames {t0}..{t1-1}")

    meta = {
        "raw_input": str(RAW_TIF),
        "output": str(OUT_TIF),
        "shifts_npy": str(SHIFTS),
        "mode": MODE,
        "ref_stride": REF_STRIDE,
        "subpix": SUBPIX,
        "y_crop_bottom": Y_CROP_BOTTOM,
        "chunk_t": CHUNK_T
    }
    with open(META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Meta → {META}\nDone.")

if __name__ == "__main__":
    main()
