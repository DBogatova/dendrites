#!/usr/bin/env python
"""
Module 0D — Build dff_stack.tif (chunked, 20th-percentile F0 via P² estimator)
- Crops only the BOTTOM 3 pixels in Y before baseline & output.
- Uses Zarr-backed slicing for true chunked reads of compressed TIFFs.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import tifffile
import zarr

# ======= CONFIG =======
DATE  = "2025-08-06"
MOUSE = "organoid"
RUN   = "run4"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_IN   = BASE / "raw" / f"runB_{RUN}_reslice.tif"
DFF_OUT  = BASE / "preprocessed" / f"runB_{RUN}_reslice_dff_stack.tif"

Q = 0.20          # quantile for F0
EPS = 1e-6        # avoid divide by zero
CHUNK_T = 118     # time frames per chunk
Y_CROP_BOTTOM = 3 # crop ONLY bottom in Y

# ======= HELPERS =======

def open_series(path: Path):
    """Open TIFF and return (tf, series, shape (T,Z,Y,X), dtype)."""
    tf = tifffile.TiffFile(str(path))
    ser = tf.series[0]
    shape = ser.shape
    dtype = ser.dtype
    if len(shape) == 3:      # (Z,Y,X)
        T = 1; Z, Y, X = shape
    elif len(shape) == 4:    # (T,Z,Y,X)
        T, Z, Y, X = shape
    else:
        tf.close()
        raise ValueError(f"Unsupported TIFF shape: {shape}")
    return tf, ser, (T, Z, Y, X), dtype

def crop_y_bottom(arr: np.ndarray) -> np.ndarray:
    """Crop ONLY bottom Y_CROP_BOTTOM pixels along the Y axis."""
    if Y_CROP_BOTTOM < 0:
        raise ValueError("Y_CROP_BOTTOM must be ≥ 0")
    if Y_CROP_BOTTOM == 0:
        return arr
    if arr.shape[-2] <= Y_CROP_BOTTOM:
        raise ValueError("Y_CROP_BOTTOM too large for array height")
    return arr[..., :-Y_CROP_BOTTOM, :]

def read_time_chunk(tf: tifffile.TiffFile, t0: int, t1: int, Z: int) -> np.ndarray:
    """
    Read frames [t0, t1) as (k, Z, Yc, X) without loading the whole file.
    Uses TiffFile.series[0].asarray() with slicing for chunked reads.
    """
    try:
        # Try the zarr approach first (for older zarr versions)
        store = tf.series[0].aszarr()
        z = zarr.open(store, mode="r")
        arr = z if z.ndim == 4 else z[None, ...]
        chunk = np.asarray(arr[t0:t1])
    except (TypeError, AttributeError):
        # Fallback for newer zarr versions - use direct slicing
        full_arr = tf.series[0].asarray()
        if full_arr.ndim == 3:
            full_arr = full_arr[None, ...]  # add time dimension
        chunk = full_arr[t0:t1]
    
    return crop_y_bottom(chunk)

# ---------- P² estimator (per voxel, vectorized) ----------
class P2State:
    """
    Vectorized P² quantile estimator (q in (0,1)), keeping 5 markers per voxel.
    """
    __slots__ = ("q1","q2","q3","q4","q5","n1","n2","n3","n4","n5","np1","np2","np3","np4","np5","q")
    def __init__(self, init5: np.ndarray, q: float):
        # init5: (5,Z,Y,X) sorted along the first axis
        self.q1, self.q2, self.q3, self.q4, self.q5 = [a.astype(np.float32) for a in init5]
        base = np.ones(self.q1.shape, dtype=np.float32)
        self.n1 = 1*base; self.n2 = 2*base; self.n3 = 3*base; self.n4 = 4*base; self.n5 = 5*base
        self.np1 = self.n1.copy(); self.np2 = self.n2.copy(); self.np3 = self.n3.copy(); self.np4 = self.n4.copy(); self.np5 = self.n5.copy()
        self.q = float(q)

    def update_batch(self, x: np.ndarray):
        q = self.q
        less  = x < self.q1
        mid12 = (~less) & (x < self.q2)
        mid23 = (~less) & (~mid12) & (x < self.q3)
        mid34 = (~less) & (~mid12) & (~mid23) & (x < self.q4)
        geq4  = (~less) & (~mid12) & (~mid23) & (~mid34)

        # adjust extremes
        self.q1 = np.where(less & (x < self.q1), x, self.q1)
        self.q5 = np.where(geq4 & (x > self.q5), x, self.q5)

        # marker counts
        self.n1 += less.astype(np.float32)
        self.n2 += (less | mid12).astype(np.float32)
        self.n3 += (less | mid12 | mid23).astype(np.float32)
        self.n4 += (less | mid12 | mid23 | mid34).astype(np.float32)
        self.n5 += 1.0

        # desired positions
        self.np1 += 0.0
        self.np2 += q/2.0
        self.np3 += q
        self.np4 += (1.0+q)/2.0
        self.np5 += 1.0

        self._adjust_marker(2)
        self._adjust_marker(3)
        self._adjust_marker(4)

    def _adjust_marker(self, k:int):
        if k == 2:
            qk, qm, qp = self.q2, self.q1, self.q3
            nk, nm, np_, nkm1, nkp1 = self.n2, self.n1, self.np2, self.n1, self.n3
        elif k == 3:
            qk, qm, qp = self.q3, self.q2, self.q4
            nk, nm, np_, nkm1, nkp1 = self.n3, self.n2, self.np3, self.n2, self.n4
        else:
            qk, qm, qp = self.q4, self.q3, self.q5
            nk, nm, np_, nkm1, nkp1 = self.n4, self.n3, self.np4, self.n3, self.n5

        d = np_ - nk
        move_up = d >= 1.0
        move_dn = d <= -1.0

        denom_kp = np.where((nkp1 - nk)==0, 1.0, (nkp1 - nk))
        denom_km = np.where((nk - nkm1)==0, 1.0, (nk - nkm1))
        denom    = np.where((nkp1 - nkm1)==0, 1.0, (nkp1 - nkm1))

        qpar   = qk + (((nk - nkm1 + d) * (qp - qk) / denom_kp) + ((nkp1 - nk - d) * (qk - qm) / denom_km)) / denom
        qlin_u = qk + (qp - qk) / denom_kp
        qlin_d = qk - (qk - qm) / denom_km

        if np.any(move_up):
            use_lin = (qpar >= qp) | (qpar <= qm)
            qk[move_up] = np.where(use_lin[move_up], qlin_u[move_up], qpar[move_up]); nk[move_up] += 1.0
        if np.any(move_dn):
            use_lin = (qpar >= qp) | (qpar <= qm)
            qk[move_dn] = np.where(use_lin[move_dn], qlin_d[move_dn], qpar[move_dn]); nk[move_dn] -= 1.0

    def f0(self) -> np.ndarray:
        return self.q3.astype(np.float32)

# ======= BADMASK =======

def compute_badmask(tf: tifffile.TiffFile, T: int, Z: int) -> np.ndarray:
    """Keep voxels that are finite and non-constant using samples at t=0, mid, last (with crop)."""
    picks = [0, T//2, T-1] if T >= 3 else list(range(T))
    samples = []
    for t0 in picks:
        arr = read_time_chunk(tf, t0, t0+1, Z)[0].astype(np.float32)  # (Z, Yc, X)
        samples.append(arr)
    stack = np.stack(samples, axis=0)  # (S,Z,Yc,X)
    finite = np.isfinite(stack).all(axis=0)
    var_ok = stack.var(axis=0) > 0
    mean_ok = stack.mean(axis=0) != 0
    keep = finite & var_ok & mean_ok
    return keep

# ======= MAIN =======

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', type=Path, default=RAW_IN)
    ap.add_argument('--out', dest='out_path', type=Path, default=DFF_OUT)
    ap.add_argument('--eps', type=float, default=EPS)
    ap.add_argument('--chunk', type=int, default=CHUNK_T)
    args = ap.parse_args()

    with tifffile.TiffFile(str(args.in_path)) as tf:
        ser = tf.series[0]
        shape = ser.shape
        if len(shape) == 3:
            T = 1; Z, Y, X = shape
        elif len(shape) == 4:
            T, Z, Y, X = shape
        else:
            raise ValueError(f"Unsupported TIFF shape: {shape}")

        print(f"Loaded raw header: (T,Z,Y,X)=({T},{Z},{Y},{X}) dtype={ser.dtype}")
        print(f"✂️ Cropping bottom Y: {Y} → {Y - Y_CROP_BOTTOM}")

        keepmask = compute_badmask(tf, T, Z)  # with crop applied inside read_time_chunk

        # ----- Pass 1: P² init with first up-to-5 frames -----
        init_frames = min(5, T)
        init = read_time_chunk(tf, 0, init_frames, Z).astype(np.float32)  # (k,Z,Yc,X)
        if init.ndim == 3:
            init = init[None]
        pad_needed = 5 - init.shape[0]
        if pad_needed > 0:
            pad = np.repeat(init[-1:,...], pad_needed, axis=0)
            init = np.concatenate([init, pad], axis=0)
        init_sorted = np.sort(init, axis=0)
        p2 = P2State(init_sorted[:5], q=Q)

        # Streaming update
        for t0 in range(init_frames, T, args.chunk):
            t1 = min(T, t0 + args.chunk)
            chunk = read_time_chunk(tf, t0, t1, Z).astype(np.float32)
            for k in range(chunk.shape[0]):
                x = np.where(keepmask, chunk[k], 0.0)
                p2.update_batch(x)
            print(f"P² pass: processed frames {t0}..{t1-1}")

        f0 = np.where(keepmask, p2.f0(), 0.0)   # (Z, Yc, X)

        # ----- Pass 2: write ΔF/F in chunks -----
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        first = True
        for t0 in range(0, T, args.chunk):
            t1 = min(T, t0 + args.chunk)
            raw_chunk = read_time_chunk(tf, t0, t1, Z).astype(np.float32)  # (k,Z,Yc,X)

            # In-place ΔF/F to minimize allocations
            np.subtract(raw_chunk, f0, out=raw_chunk)
            np.divide(raw_chunk, (f0 + args.eps), out=raw_chunk)
            raw_chunk[:, ~keepmask] = 0.0
            dff_chunk = raw_chunk  # alias

            tifffile.imwrite(
                str(args.out_path),
                dff_chunk,
                imagej=False,
                append=not first,
                bigtiff=True,
                compression='zlib'
            )
            first = False
            print(f"Wrote frames {t0}..{t1-1} → {args.out_path}")

    # Meta written next to output
    Yc = int(Y - Y_CROP_BOTTOM)
    meta = {
        "input": str(args.in_path),
        "output": str(args.out_path),
        "shape": [int(T), int(Z), Yc, int(X)],  # (T,Z,Yc,X)
        "dtype": "float32",
        "f0_quantile": float(Q),
        "eps": float(args.eps),
        "chunk_t": int(args.chunk),
        "y_crop_top": 0,
        "y_crop_bottom": int(Y_CROP_BOTTOM)
    }
    meta_out = args.out_path.with_suffix("").with_name(args.out_path.stem + "_meta.json")
    with open(meta_out, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Meta → {meta_out}")

if __name__ == "__main__":
    main()
