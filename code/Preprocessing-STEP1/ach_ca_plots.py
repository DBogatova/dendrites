#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dual-Channel Quicklook (ACh first, then Calcium)
- Global ΔF/F traces
- Bleach correction
- Cross-correlation (ACh vs Ca ΔF/F)
No filtering, no coherence.
"""

import gc
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# ===== CONFIGURATION =====
DATE = "2025-08-29"
MOUSE = "rAi162_18"
RUN   = "run6"

# Acquisition
FS_HZ = 10.0   # frames per second

# Z-handling
ZMIN = 0
ZMAX = -1

# Projections
ACH_PROJ = "mean"    # 'mean' | 'mip' | 'percentile'
ACH_PCT  = 80.0
CA_PROJ  = "mean"

# ΔF/F baseline
DFF_WINDOW = 301
DFF_PCT    = 20.0

# Cross-correlation
MAX_LAG_FRAMES = 100
CC_NORMALIZE   = True

# Paths
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_ACH_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_red_reslice.tif"
RAW_CA_PATH  = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green_reslice.tif"

OUT_DIR = BASE / "quicklook"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CSV_PATH     = OUT_DIR / "dual_quicklook_traces.csv"
FIG_PATH     = OUT_DIR / "dual_quicklook_traces.pdf"
CC_CSV_PATH  = OUT_DIR / "cross_correlation.csv"
CC_FIG_PATH  = OUT_DIR / "cross_correlation.pdf"
SUMMARY_PATH = OUT_DIR / "correlation_summary.txt"

# ===== Utilities =====
def load_4d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D (T,Z,Y,X); got {arr.shape}")
    return arr.astype(np.float32)

def z_substack(arr: np.ndarray, zmin: int = 0, zmax: int | None = None) -> np.ndarray:
    if zmax is None:
        return arr[:, zmin:, :, :]
    return arr[:, zmin:zmax, :, :]

def z_project(arr: np.ndarray, mode: str = "mean", percentile: float = 80.0) -> np.ndarray:
    if mode == "mean":
        return arr.mean(axis=1)
    if mode == "mip":
        return arr.max(axis=1)
    if mode == "percentile":
        return np.percentile(arr, percentile, axis=1)
    raise ValueError("mode must be 'mean', 'mip', or 'percentile'")

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def bleach_correct_trace(trace: np.ndarray):
    x = np.arange(len(trace), dtype=np.float32)
    y = trace.astype(np.float32)
    p0 = (float(y.max()-y.min()), 1e-4, float(np.median(y)))
    try:
        (a, b, c), _ = curve_fit(exp_decay, x, y, p0=p0, maxfev=10000)
        fit = exp_decay(x, a, b, c)
        corrected = y - fit + np.median(fit)
        ss_res = np.sum((y - fit)**2)
        ss_tot = np.sum((y - y.mean())**2) + 1e-12
        r2 = 1 - ss_res/ss_tot
        return corrected.astype(np.float32), (a, b, c), float(r2), fit.astype(np.float32)
    except Exception:
        med = np.median(y)
        return y.copy(), (0,0,med), 0.0, np.full_like(y, med, dtype=np.float32)

def rolling_percentile_baseline(trace: np.ndarray, win: int = 301, pct: float = 20.0):
    if win % 2 == 0: win += 1
    half = win // 2
    pad = np.pad(trace, (half, half), mode="reflect")
    out = np.empty_like(trace, dtype=np.float32)
    for i in range(len(trace)):
        out[i] = np.percentile(pad[i:i+win], pct)
    return out

def compute_dff_from_global(trace: np.ndarray, win: int = 301, pct: float = 20.0):
    F0 = rolling_percentile_baseline(trace, win, pct)
    dff = (trace - F0) / (F0 + 1e-6)
    return dff.astype(np.float32), F0.astype(np.float32)

def _zscore(x): return (x - np.mean(x)) / (np.std(x) + 1e-12)

def cross_correlation(x, y, max_lag, zscore=True):
    assert x.shape == y.shape
    T = len(x)
    xx = _zscore(x) if zscore else x - x.mean()
    yy = _zscore(y) if zscore else y - y.mean()
    lags = np.arange(-max_lag, max_lag+1, dtype=int)
    r = np.zeros_like(lags, dtype=np.float32)
    for i, lag in enumerate(lags):
        if lag < 0:   a, b = xx[:T+lag], yy[-lag:]
        elif lag > 0: a, b = xx[lag:], yy[:T-lag]
        else:         a, b = xx, yy
        r[i] = np.corrcoef(a, b)[0,1] if len(a) > 5 else np.nan
    return lags, r

# ===== Main =====
def main():
    print(f"Loading ACh: {RAW_ACH_PATH}")
    ach_4d = load_4d(RAW_ACH_PATH)
    print(f"Loading Ca:  {RAW_CA_PATH}")
    ca_4d  = load_4d(RAW_CA_PATH)

    if ach_4d.shape != ca_4d.shape:
        raise ValueError(f"Shape mismatch: ACh {ach_4d.shape} vs Ca {ca_4d.shape}")
    T,Z,Y,X = ach_4d.shape
    print(f"Stacks shape: T={T}, Z={Z}, Y={Y}, X={X}, fs={FS_HZ} Hz")

    # crop Z
    zmax_eff = None if ZMAX < 0 else ZMAX
    ach_4d = z_substack(ach_4d, ZMIN, zmax_eff)
    ca_4d  = z_substack(ca_4d, ZMIN, zmax_eff)

    # Z-projection
    ach_3d = z_project(ach_4d, ACH_PROJ, ACH_PCT)
    ca_3d  = z_project(ca_4d, CA_PROJ)

    del ach_4d, ca_4d; gc.collect()

    # global raw traces
    ach_raw = ach_3d.mean(axis=(1,2))
    ca_raw  = ca_3d.mean(axis=(1,2))

    # bleach correction
    ach_corr, (aa,ab,ac), aR2, ach_fit = bleach_correct_trace(ach_raw)
    ca_corr, (caa,cab,cac), cR2, ca_fit = bleach_correct_trace(ca_raw)

    # ΔF/F
    ach_dff, ach_F0 = compute_dff_from_global(ach_raw, DFF_WINDOW, DFF_PCT)
    ca_dff,  ca_F0  = compute_dff_from_global(ca_raw,  DFF_WINDOW, DFF_PCT)

    # save CSV
    t = np.arange(T)
    df = pd.DataFrame({
        "frame": t,
        "ACh_raw": ach_raw, "ACh_corr": ach_corr, "ACh_fit": ach_fit, "ACh_dFF": ach_dff,
        "Ca_raw": ca_raw,   "Ca_corr": ca_corr,   "Ca_fit": ca_fit,   "Ca_dFF": ca_dff
    })
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved traces CSV → {CSV_PATH}")

    # figure
    fig, axes = plt.subplots(2,2, figsize=(12,6), sharex=True)
    axes[0,0].plot(t, ach_raw, label="raw"); axes[0,0].plot(t, ach_fit, "--", label="fit"); axes[0,0].plot(t, ach_corr, label="corr")
    axes[0,0].set_title("ACh raw/fit/corr"); axes[0,0].legend()
    axes[0,1].plot(t, ach_dff, label="ACh ΔF/F"); axes[0,1].set_title("ACh ΔF/F")
    axes[1,0].plot(t, ca_raw, label="raw"); axes[1,0].plot(t, ca_fit, "--", label="fit"); axes[1,0].plot(t, ca_corr, label="corr")
    axes[1,0].set_title("Ca raw/fit/corr"); axes[1,0].legend()
    axes[1,1].plot(t, ca_dff, label="Ca ΔF/F"); axes[1,1].set_title("Ca ΔF/F")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=200); plt.close(fig)
    print(f"Saved figure → {FIG_PATH}")

    # cross-correlation
    lags, r = cross_correlation(ach_dff, ca_dff, MAX_LAG_FRAMES, zscore=CC_NORMALIZE)
    pd.DataFrame({"lag_frames": lags, "r": r}).to_csv(CC_CSV_PATH, index=False)
    plt.figure(); plt.plot(lags, r); plt.axvline(0, ls="--", c="k"); plt.xlabel("Lag (frames)"); plt.ylabel("r")
    plt.title("Cross-corr ACh vs Ca ΔF/F"); plt.savefig(CC_FIG_PATH, dpi=200); plt.close()
    print(f"Saved cross-corr → {CC_FIG_PATH}")

if __name__ == "__main__":
    main()
