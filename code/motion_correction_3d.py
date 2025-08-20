#!/usr/bin/env python
"""
3D Motion Correction

This script performs 3D optical flow motion correction on 4D calcium imaging data:
1. Loads raw 4D stack (T, Z, Y, X)
2. Creates a reference template from specified frames
3. Applies 3D optical flow correction to align each frame to the reference
4. Saves the motion-corrected stack
"""

from pathlib import Path
import numpy as np
import tifffile
import scipy.ndimage as spnd
import sys

# Try to import opticalflow3D
try:
    import opticalflow3D as opticalflow_3d
except ImportError:
    print("ERROR: opticalflow3D module not found.")
    print("Please install it by running:")
    print("  git clone https://github.com/VolumeGraphics/opticalflow3D.git")
    print("  cd opticalflow3D")
    print("  pip install -e .")
    print("Or add the path to your sys.path if installed elsewhere.")
    sys.exit(1)

# === CONFIGURATION ===
DATE = "2025-08-06"
MOUSE = "organoid"
RUN = "run4"
Y_CROP = 3

# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runB_{RUN}_reslice_test.tif"
OUTPUT_PATH = BASE / "raw" / f"runA_{RUN}_reslice_test_mc.tif"

# === PARAMETERS ===
# Reference template options
REF_START, REF_END = 0, 200       # Frames used to build the template
TEMPLATE_STAT = "median"          # "median" or "mean"

# Photometric normalization
DO_NORMALIZE = True               # Match mean/std to reference per frame

# Flow estimation parameters
FLOW_PARAMS = dict(
    iters=5,
    num_levels=5,
    scale=0.5,
    spatial_size=7,
    presmoothing=5,
    filter_type="gaussian",
    filter_size=64
)

# Confidence and smoothing
CONF_MIN = 0.05                   # Ignore flow where confidence is very low
FLOW_SMOOTH_SIGMA = (1.0, 1.0, 1.0)  # Gaussian sigma on (Z,Y,X) flow components

# Anisotropy handling
UPSAMPLE_Z_FOR_FLOW = False       # Set True for quasi-isotropic flow
Z_UPSAMPLE_FACTOR = 4             # Z upsampling factor

# Processing
TIME_CHUNK = 64                   # Process this many frames at a time

def build_template(stack, t0, t1, stat="median"):
    """
    Build reference template from specified frames.
    
    Args:
        stack: 4D array (T, Z, Y, X)
        t0, t1: Start and end frame indices
        stat: "median" or "mean"
        
    Returns:
        3D reference volume
    """
    vol = stack[t0:t1]
    if stat == "median":
        ref = np.median(vol, axis=0)
    else:
        ref = np.mean(vol, axis=0)
    return ref.astype(stack.dtype)

def normalize_to_ref(vol, ref):
    """
    Normalize volume intensity to match reference.
    
    Args:
        vol: Volume to normalize
        ref: Reference volume
        
    Returns:
        Normalized volume
    """
    v = vol.astype(np.float32)
    r = ref.astype(np.float32)
    v_mean, v_std = np.mean(v), np.std(v) + 1e-6
    r_mean, r_std = np.mean(r), np.std(r) + 1e-6
    v_norm = (v - v_mean) / v_std * r_std + r_mean
    
    # Keep within data range
    if np.issubdtype(vol.dtype, np.integer):
        info = np.iinfo(vol.dtype)
        v_norm = np.clip(v_norm, info.min, info.max)
        return v_norm.astype(vol.dtype)
    return v_norm.astype(vol.dtype)

def maybe_upsample_z(vol, factor):
    """Upsample Z dimension if needed for isotropic flow estimation."""
    if factor == 1:
        return vol, 1.0
    up = spnd.zoom(vol, zoom=(factor, 1.0, 1.0), order=1)
    return up, factor

def maybe_downsample_flow_z(flow_zyx, factor):
    """Downsample flow field back to original Z resolution."""
    if factor == 1:
        return flow_zyx
    dz, dy, dx = flow_zyx
    target_z = dz.shape[0] // factor
    dz_resized = spnd.zoom(dz, zoom=(1/factor, 1.0, 1.0), order=1) / factor
    dy_resized = spnd.zoom(dy, zoom=(1/factor, 1.0, 1.0), order=1)
    dx_resized = spnd.zoom(dx, zoom=(1/factor, 1.0, 1.0), order=1)
    return np.stack([dz_resized, dy_resized, dx_resized], axis=0)

def smooth_flow(flow_zyx, sigma=(1,1,1), conf=None, conf_min=0.0):
    """
    Smooth flow field with confidence-based masking.
    
    Args:
        flow_zyx: Flow field (3, Z, Y, X)
        sigma: Gaussian smoothing sigma
        conf: Confidence map
        conf_min: Minimum confidence threshold
        
    Returns:
        Smoothed flow field
    """
    f = flow_zyx.copy()
    if conf is not None and conf_min > 0:
        mask = conf >= conf_min
        f[:, ~mask] = 0
    
    # Smooth each component
    for i in range(3):
        f[i] = spnd.gaussian_filter(f[i], sigma=sigma, mode='nearest')
    return f

def warp_to_reference(mov, flow_zyx):
    """
    Warp moving volume to reference using flow field.
    
    Args:
        mov: Moving volume (Z, Y, X)
        flow_zyx: Flow field (3, Z, Y, X)
        
    Returns:
        Warped volume
    """
    Z, Y, X = mov.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    
    # Sample coordinates
    coords = np.array([
        zz - flow_zyx[0],
        yy - flow_zyx[1],
        xx - flow_zyx[2],
    ])
    
    warped = spnd.map_coordinates(mov, coords, order=1, mode='nearest', prefilter=False)
    return warped

def main():
    print("Loading stack...")
    stack = tifffile.imread(RAW_STACK_PATH)  # (T, Z, Y, X)
    assert stack.ndim == 4, "Expected (T, Z, Y, X)"
    
    # Apply Y cropping if specified
    if Y_CROP > 0:
        stack = stack[:, :, :-Y_CROP, :]
    
    T, Z, Y, X = stack.shape
    print(f"Stack: T={T}, Z={Z}, Y={Y}, X={X}")

    print("Building reference template...")
    ref = build_template(stack, REF_START, min(REF_END, T), stat=TEMPLATE_STAT)

    # Initialize optical flow
    farneback = opticalflow_3d.Farneback3D(**FLOW_PARAMS)

    # Prepare output
    print(f"Creating output file: {OUTPUT_PATH}")
    mc_memmap = tifffile.memmap(OUTPUT_PATH, shape=stack.shape, dtype=stack.dtype)

    print("Starting motion correction...")
    for t0 in range(0, T, TIME_CHUNK):
        t1 = min(t0 + TIME_CHUNK, T)
        block = stack[t0:t1]  # View of current chunk
        corrected = np.empty_like(block)

        for i, vol in enumerate(block):
            v = vol
            r = ref
            
            # Normalize intensity if requested
            if DO_NORMALIZE:
                v = normalize_to_ref(v, r)

            # Optional Z upsampling for better isotropic flow
            if UPSAMPLE_Z_FOR_FLOW:
                v_up, fac = maybe_upsample_z(v, Z_UPSAMPLE_FACTOR)
                r_up, _  = maybe_upsample_z(r, Z_UPSAMPLE_FACTOR)
                *vv, conf = farneback.calculate_flow(v_up, r_up, total_vol=None)
                flow = np.stack(vv, axis=0)  # (3, Z', Y, X)
                flow = maybe_downsample_flow_z(flow, Z_UPSAMPLE_FACTOR)
            else:
                *vv, conf = farneback.calculate_flow(v, r, total_vol=None)
                flow = np.stack(vv, axis=0)  # (3, Z, Y, X)

            # Smooth flow field with confidence gating
            flow = smooth_flow(flow, sigma=FLOW_SMOOTH_SIGMA, conf=conf, conf_min=CONF_MIN)

            # Warp moving volume to reference coordinates
            corrected[i] = warp_to_reference(vol, flow)

        mc_memmap[t0:t1] = corrected
        print(f"Frames {t0}:{t1} corrected.")

    # Ensure data is flushed
    del mc_memmap
    print(f"✅ Motion-corrected stack saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()