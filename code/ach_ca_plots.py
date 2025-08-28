#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 1b: Initial Dual-Channel Quicklook (ACh first, then Calcium) + Correlation Analysis

This script performs initial processing of raw dual-channel imaging data:
1) Loads two raw 4D stacks (T, Z, Y, X): ACh (red) FIRST, then Calcium (green)
2) Applies identical Z cropping (to remove bright surface if desired)
3) Projects each frame over Z (ACh→mean by default; Ca→MIP by default)
4) Computes global raw traces (image mean per frame)
5) Fits & applies exponential bleach correction to the raw traces
6) Computes ΔF/F using a rolling-percentile baseline (robust detrending)
7) Plots ACh (top) and Calcium (bottom): raw+fit+corrected (left), ΔF/F (right)
8) Computes cross-correlation of ΔF/F traces:
   - Pearson r at lag 0
   - Peak r and the lag (in frames)
   - Saves cross-correlation curve and a summary
"""

import gc
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.optimize import curve_fit

# ===== Matplotlib style =====
mpl.rcParams['font.family'] = 'CMU Serif'

# ===== CONFIGURATION =====
DATE = "2025-08-27"
MOUSE = "rAi162_18"
RUN   = "run2"

# Z-handling
ZMIN = 0            # inclusive
ZMAX = -1           # exclusive; -1 means "use full Z"

# Projections
ACH_PROJ = "mip"   # 'mean' | 'mip' | 'percentile'
ACH_PCT  = 80.0     # if ACH_PROJ == 'percentile'
CA_PROJ  = "mip"    # 'mean' | 'mip' | 'percentile'

# ΔF/F baseline (global trace)
DFF_WINDOW = 301    # frames (odd); e.g., ~30 s at 10 Hz
DFF_PCT    = 20.0   # rolling percentile for F0

# Cross-correlation
MAX_LAG_FRAMES = 100   # search lags in [-MAX_LAG_FRAMES, +MAX_LAG_FRAMES]
CC_NORMALIZE   = True  # z-score ΔF/F traces before correlation

# Paths
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_ACH_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_red_reslice.tif"    # ACh FIRST
RAW_CA_PATH  = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green_reslice.tif"  # Calcium SECOND

OUT_DIR = BASE / "quicklook"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CSV_PATH = OUT_DIR / "dual_quicklook_traces.csv"
FIG_PATH = OUT_DIR / "dual_quicklook_traces.pdf"
CC_CSV_PATH = OUT_DIR / "cross_correlation.csv"
CC_FIG_PATH = OUT_DIR / "cross_correlation.pdf"
SUMMARY_PATH = OUT_DIR / "correlation_summary.txt"

# ===== Utilities =====
def load_4d(path: Path) -> np.ndarray:
    """Load TIFF as float32 4D stack (T, Z, Y, X)."""
    arr = tifffile.imread(str(path))
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D (T,Z,Y,X); got {arr.shape} from {path}")
    return arr.astype(np.float32)

def z_substack(arr: np.ndarray, zmin: int = 0, zmax: int | None = None) -> np.ndarray:
    """Trim Z dimension [zmin:zmax)."""
    if zmax is None:
        return arr[:, zmin:, :, :]
    return arr[:, zmin:zmax, :, :]

def z_project(arr: np.ndarray, mode: str = "mean", percentile: float = 80.0) -> np.ndarray:
    """Project over Z → (T, Y, X)."""
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
    """
    Fit y = a*exp(-b*x)+c to raw global trace.
    Returns: corrected, (a,b,c), R^2, fitted_curve
    """
    x = np.arange(len(trace), dtype=np.float32)
    y = trace.astype(np.float32)
    p0 = (float(y.max() - y.min()), 1e-4, float(np.median(y)))
    try:
        (a, b, c), _ = curve_fit(exp_decay, x, y, p0=p0, maxfev=10000)
        fit = exp_decay(x, a, b, c)
        corrected = y - fit + np.median(fit)  # flatten around median
        ss_res = np.sum((y - fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return corrected.astype(np.float32), (float(a), float(b), float(c)), float(r2), fit.astype(np.float32)
    except Exception:
        med = np.median(y)
        return y.copy(), (0.0, 0.0, float(med)), 0.0, np.full_like(y, med, dtype=np.float32)

def rolling_percentile_baseline(trace: np.ndarray, win: int = 301, pct: float = 20.0) -> np.ndarray:
    """Rolling-percentile baseline (reflect padding). win must be odd."""
    if win % 2 == 0:
        win += 1
    half = win // 2
    pad = np.pad(trace, (half, half), mode='reflect')
    out = np.empty_like(trace, dtype=np.float32)
    for i in range(len(trace)):
        seg = pad[i:i + win]
        out[i] = np.percentile(seg, pct)
    return out

def compute_dff_from_global(trace: np.ndarray, win: int = 301, pct: float = 20.0):
    """ΔF/F for a single global trace using rolling-percentile F0."""
    F0 = rolling_percentile_baseline(trace, win=win, pct=pct)
    dff = (trace - F0) / (F0 + 1e-6)
    return dff.astype(np.float32), F0.astype(np.float32)

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x)
    sd = np.std(x) + 1e-12
    return (x - mu) / sd

def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int, zscore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation between x and y for lags in [-max_lag, +max_lag].
    Returns (lags, r), where r[lag==0] is the zero-lag correlation.
    """
    assert x.shape == y.shape
    T = len(x)
    xx = _zscore(x) if zscore else x - x.mean()
    yy = _zscore(y) if zscore else y - y.mean()

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    r = np.zeros_like(lags, dtype=np.float32)

    for i, lag in enumerate(lags):
        if lag < 0:
            a = xx[:T + lag]
            b = yy[-lag:]
        elif lag > 0:
            a = xx[lag:]
            b = yy[:T - lag]
        else:
            a = xx
            b = yy
        # Pearson r on overlapping region
        if len(a) > 5:
            r[i] = np.corrcoef(a, b)[0, 1]
        else:
            r[i] = np.nan
    return lags, r

# ===== Main =====
def main():
    # --- Load ACh FIRST, then Calcium ---
    print(f"Loading ACh: {RAW_ACH_PATH}")
    ach_4d = load_4d(RAW_ACH_PATH)
    print(f"Loading Calcium: {RAW_CA_PATH}")
    ca_4d  = load_4d(RAW_CA_PATH)

    if ach_4d.shape != ca_4d.shape:
        raise ValueError(f"Shape mismatch: ACh {ach_4d.shape} vs Ca {ca_4d.shape}")

    T, Z, Y, X = ach_4d.shape
    print(f"Stacks shape: T={T}, Z={Z}, Y={Y}, X={X}")

    # --- Shared Z cropping ---
    zmax_eff = None if ZMAX < 0 else ZMAX
    ach_4d = z_substack(ach_4d, ZMIN, zmax_eff)
    ca_4d  = z_substack(ca_4d,  ZMIN, zmax_eff)
    T, Z, Y, X = ach_4d.shape
    print(f"After Z-substack: T={T}, Z={Z}, Y={Y}, X={X}")

    # --- Per-frame Z projection ---
    print(f"Projecting ACh over Z: mode={ACH_PROJ}, pct={ACH_PCT if ACH_PROJ=='percentile' else 'n/a'}")
    ach_3d = z_project(ach_4d, mode=ACH_PROJ, percentile=ACH_PCT)  # (T, Y, X)

    print(f"Projecting Ca over Z: mode={CA_PROJ}")
    ca_3d  = z_project(ca_4d,  mode=CA_PROJ)                      # (T, Y, X)

    # Free big arrays we no longer need
    del ach_4d, ca_4d
    gc.collect()

    # --- Global raw traces (image means) ---
    ach_raw = ach_3d.mean(axis=(1, 2))
    ca_raw  = ca_3d.mean(axis=(1, 2))

    # --- Bleach correction on raw traces ---
    ach_corr, (aa, ab, ac), aR2, ach_fit = bleach_correct_trace(ach_raw)
    ca_corr,  (caa, cab, cac), cR2,  ca_fit = bleach_correct_trace(ca_raw)

    # --- ΔF/F from raw traces (robust F0) ---
    ach_dff, ach_F0 = compute_dff_from_global(ach_raw, win=DFF_WINDOW, pct=DFF_PCT)
    ca_dff,  ca_F0  = compute_dff_from_global(ca_raw,  win=DFF_WINDOW, pct=DFF_PCT)

    # --- Save CSV with traces ---
    t = np.arange(T, dtype=int)
    df = pd.DataFrame({
        "frame": t,
        # ACh
        "ACh_raw": ach_raw.astype(np.float32),
        "ACh_raw_bleach_fit": ach_fit.astype(np.float32),
        "ACh_raw_bleachcorr": ach_corr.astype(np.float32),
        "ACh_dFF": ach_dff.astype(np.float32),
        "ACh_F0": ach_F0.astype(np.float32),
        "ACh_fit_a": np.full(T, aa, dtype=np.float32),
        "ACh_fit_b": np.full(T, ab, dtype=np.float32),
        "ACh_fit_c": np.full(T, ac, dtype=np.float32),
        "ACh_fit_R2": np.full(T, aR2, dtype=np.float32),
        # Ca
        "Ca_raw": ca_raw.astype(np.float32),
        "Ca_raw_bleach_fit": ca_fit.astype(np.float32),
        "Ca_raw_bleachcorr": ca_corr.astype(np.float32),
        "Ca_dFF": ca_dff.astype(np.float32),
        "Ca_F0": ca_F0.astype(np.float32),
        "Ca_fit_a": np.full(T, caa, dtype=np.float32),
        "Ca_fit_b": np.full(T, cab, dtype=np.float32),
        "Ca_fit_c": np.full(T, cac, dtype=np.float32),
        "Ca_fit_R2": np.full(T, cR2, dtype=np.float32),
    })
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved traces CSV → {CSV_PATH}")

    # --- Plot: ACh (top), Ca (bottom): raw+fit+corr (left) and ΔF/F (right) ---
    print("Saving figure (traces)…")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    # ACh raw/fit/corr
    ax = axes[0, 0]
    ax.plot(t, ach_raw, label="ACh raw", linewidth=1.25)
    ax.plot(t, ach_fit, label="ACh bleach fit", linestyle="--", linewidth=1.0)
    ax.plot(t, ach_corr, label="ACh bleach-corrected", linewidth=1.0, alpha=0.9)
    ax.set_title("ACh: raw, fit, corrected")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend(loc="best", fontsize=9)

    # ACh ΔF/F
    ax = axes[0, 1]
    ax.plot(t, ach_dff, label=f"ACh ΔF/F (win={DFF_WINDOW}, pct={DFF_PCT})", linewidth=1.25)
    ax.set_title("ACh: ΔF/F")
    ax.set_ylabel("ΔF/F")
    ax.legend(loc="best", fontsize=9)

    # Ca raw/fit/corr
    ax = axes[1, 0]
    ax.plot(t, ca_raw, label="Ca raw", linewidth=1.25)
    ax.plot(t, ca_fit, label="Ca bleach fit", linestyle="--", linewidth=1.0)
    ax.plot(t, ca_corr, label="Ca bleach-corrected", linewidth=1.0, alpha=0.9)
    ax.set_title("Calcium: raw, fit, corrected")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend(loc="best", fontsize=9)

    # Ca ΔF/F
    ax = axes[1, 1]
    ax.plot(t, ca_dff, label=f"Ca ΔF/F (win={DFF_WINDOW}, pct={DFF_PCT})", linewidth=1.25)
    ax.set_title("Calcium: ΔF/F")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ΔF/F")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {FIG_PATH}")

    # ---------- Cross-correlation on ΔF/F ----------
    print("Computing cross-correlation (ΔF/F traces)…")
    lags, r = cross_correlation(ach_dff, ca_dff, MAX_LAG_FRAMES, zscore=CC_NORMALIZE)
    # Identify zero-lag and max
    zero_idx = np.where(lags == 0)[0][0]
    r0 = float(r[zero_idx])
    # Ignore NaNs for max
    valid = np.isfinite(r)
    if valid.any():
        peak_idx = int(np.nanargmax(np.abs(r[valid])) + np.where(valid)[0][0]) if np.isnan(r).any() else int(np.argmax(np.abs(r)))
        r_peak = float(r[peak_idx])
        lag_peak = int(lags[peak_idx])
    else:
        r_peak, lag_peak = np.nan, 0

    # Save CC CSV
    pd.DataFrame({"lag_frames": lags, "r": r}).to_csv(CC_CSV_PATH, index=False)
    print(f"Saved cross-correlation CSV → {CC_CSV_PATH}")

    # Plot CC
    print("Saving cross-correlation figure…")
    plt.figure(figsize=(8, 4))
    plt.plot(lags, r, linewidth=1.5)
    plt.axvline(0, linestyle="--", alpha=0.5, label=f"zero lag (r0={r0:.3f})")
    if np.isfinite(r_peak):
        plt.axvline(lag_peak, color="orange", linestyle="--", alpha=0.7, label=f"peak lag={lag_peak} (r={r_peak:.3f})")
    plt.title("Cross-correlation: ACh ΔF/F vs Ca ΔF/F")
    plt.xlabel("Lag (frames)  |  positive = ACh leads")
    plt.ylabel("Correlation (r)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CC_FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved cross-correlation plot → {CC_FIG_PATH}")

    # Text summary + quick interpretation hint
    def _hint(r0, r_peak, lag_peak):
        if not np.isfinite(r0):
            return "Zero-lag correlation is undefined — check traces length/NaNs."
        if abs(r0) >= 0.85:
            return "Very high zero-lag r (≥0.85): POSSIBLE crosstalk/bleed-through or shared artifact."
        if abs(r_peak) > abs(r0) and lag_peak != 0:
            return "Peak correlation occurs at non-zero lag: supports distinct kinetics (less likely bleed-through)."
        if abs(r0) >= 0.5:
            return "Moderate zero-lag correlation: could be shared drive; inspect shapes/widths of events."
        return "Low zero-lag correlation: unlikely to be simple bleed-through."

    summary = (
        f"Zero-lag r (ΔF/F): {r0:.3f}\n"
        f"Peak |r|: {abs(r_peak):.3f} at lag {lag_peak} frames (signed r={r_peak:.3f})\n"
        f"MAX_LAG_FRAMES searched: ±{MAX_LAG_FRAMES}\n"
        f"Normalization (z-score): {CC_NORMALIZE}\n"
        f"Hint: { _hint(r0, r_peak, lag_peak) }\n"
    )
    print("\n" + summary)

    with open(SUMMARY_PATH, "w") as f:
        f.write(summary)
    print(f"Wrote summary → {SUMMARY_PATH}")

    print("Dual-channel quicklook + correlation complete.")
    print("Next step: extract masks from Calcium and run ROI-based comparisons & per-ROI cross-correlations.")

if __name__ == "__main__":
    main()
