#!/usr/bin/env python
"""
Motion Filter (pre-Module-1): median-normalized corr with rolling MAD threshold
-------------------------------------------------------------------------------
- Uses either an existing MIP/ΔF/F movie OR derives MIP from RAW (T,Z,Y,X).
- Normalizes each frame by its median to remove bleaching / slow gain drift.
- Computes motion score = 1 - corr(MIP_t, MIP_{t-1}), optionally tile-wise.
- Thresholds with a rolling (local) median + MAD so early spikes are caught.
- Unites auto + manual exclusions, pads neighbors, writes NaN-masked RAW.

Outputs:
  preprocessed/raw_nanmasked.tif
  preprocessed/excluded_frames.npy
  preprocessed/excluded_frames.txt
  preprocessed/motion_filter_qc.pdf
"""

from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc

# ===================== CONFIG =====================

DATE  = "2025-08-06"
MOUSE = "organoid"
RUN   = "run4-crop"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH       = BASE / "raw" / "runB_run4_reslice-crop.tif"

# Point to an existing MIP or ΔF/F stack (T,Y,X) or (T,Z,Y,X)
MIP_OR_DFF_PATH      = BASE / "raw" / "runB_run4_reslice-crop_processed.tif"
USE_MIP_OR_DFF       = True   # set False to derive MIP from RAW

PREPROCESSED         = BASE / "preprocessed"
PREPROCESSED.mkdir(exist_ok=True)

FILTERED_STACK_PATH  = PREPROCESSED / "raw_nanmasked.tif"
EXCLUDED_NPY_PATH    = PREPROCESSED / "excluded_frames.npy"
EXCLUDED_TXT_PATH    = PREPROCESSED / "excluded_frames.txt"
QC_PDF               = PREPROCESSED / "motion_filter_qc.pdf"

# ---- Correlation detector knobs ----
# If TILES_YX is (1,1) or None -> global correlation. Use (3,3) or (4,4) to catch local motion.
TILES_YX         = (3, 3)        # (ny, nx) grid over the MIP; set to (1,1) for global only
AGGREGATE        = "max"         # "max" or "p90" aggregation of tile scores per frame
MIN_VALID_PIX    = 200           # minimum finite pixels per tile to compute corr

# ---- Rolling threshold (adaptive to bleaching/drift) ----
ROLL_WIN         = 400           # frames (≈ 1–2 min at 10 VPS). Try 300–600.
K_MAD            = 3.0           # strictness; lower (2.5) to catch more spikes

# ---- Common knobs ----
PAD_NEIGHBOR     = 2             # also exclude ±k neighbors around each hit
Y_CROP_RAW       = 3             # when deriving MIP from RAW, crop that many rows off the top

# ---- Manual exclusions (optional) ----
EXCLUDE_RANGES_STR  = ""         # e.g., "10-25, 301, 450-455"
EXCLUDE_RANGES_FILE = None       # e.g., BASE/"exclude_frames.txt"

IMWRITE_KW = dict(imagej=True)
EPS = 1e-12

# ===================== HELPERS =====================

def _parse_ranges_line(line: str):
    line = line.strip().split('#')[0].strip()
    if not line: return []
    parts = [p.strip() for p in line.replace(',', ' ').split()]
    out = []
    for p in parts:
        if '-' in p:
            a,b = p.split('-',1)
            a,b = int(a), int(b)
            if b < a: a,b = b,a
            out.extend(range(a, b+1))
        else:
            out.append(int(p))
    return out

def build_exclude_set(T: int, ranges_str: str = "", ranges_file=None):
    excl = set()
    if ranges_str:
        for f in _parse_ranges_line(ranges_str):
            if 0 <= f < T: excl.add(int(f))
    if ranges_file:
        with open(ranges_file, "r") as fh:
            for line in fh:
                for f in _parse_ranges_line(line):
                    if 0 <= f < T: excl.add(int(f))
    return sorted(excl)

def load_mip_movie(path: Path):
    """Return (T,Y,X) MIP/ΔF/F movie from (T,Y,X) or (T,Z,Y,X) file."""
    with tifffile.TiffFile(path) as tif:
        arr = tif.series[0].asarray().astype(np.float32)
    if arr.ndim == 3:   # (T,Y,X)
        return arr
    elif arr.ndim == 4: # (T,Z,Y,X)
        return np.nanmax(arr, axis=1)
    else:
        raise ValueError(f"Unsupported ndim={arr.ndim} for {path}")

def load_mip_from_raw(path: Path, ycrop: int):
    """Derive (T,Y,X) Z-MIP from RAW (T,Z,Y,X)."""
    with tifffile.TiffFile(path) as tif:
        T,Z,Y,X = tif.series[0].shape
        out = np.zeros((T, Y - ycrop if ycrop>0 else Y, X), dtype=np.float32)
        for t in range(T):
            vol = tif.series[0].asarray()[t].astype(np.float32)
            if ycrop>0: vol = vol[:, :-ycrop, :]
            out[t] = np.max(vol, axis=0)
            if (t+1)%200==0: print(f"  RAW→MIP {t+1}/{T}")
            del vol
    return out

def norm_frame(F):
    """Per-frame median normalization to suppress bleaching/drift."""
    med = np.nanmedian(F)
    return F / (med + EPS)

def rolling_stat(x, w, func):
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        seg = xp[i:i+w]
        out[i] = func(seg)
    return out

def rolling_median(x, w): return rolling_stat(x, w, lambda s: np.nanmedian(s))
def rolling_mad(x, w):
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        seg = xp[i:i+w]
        med = np.nanmedian(seg)
        out[i] = np.nanmedian(np.abs(seg - med))
    return out

def corr_scores_global(mip):
    """Global 1-corr scores (T,)."""
    T = mip.shape[0]
    scores = np.full(T, np.nan, dtype=np.float32)
    prev = norm_frame(mip[0]).ravel()
    for t in range(1, T):
        cur = norm_frame(mip[t]).ravel()
        m = np.isfinite(prev) & np.isfinite(cur)
        if m.sum() >= MIN_VALID_PIX:
            a = prev[m]; b = cur[m]
            a = (a - a.mean()) / (a.std() + EPS)
            b = (b - b.mean()) / (b.std() + EPS)
            corr = float(np.clip(np.dot(a, b) / max(1, len(a)), -1.0, 1.0))
            scores[t] = 1.0 - corr
        prev = cur
        if (t+1)%200==0: print(f"  corr {t+1}/{T}")
    return scores

def corr_scores_tiled(mip, tiles_yx=(3,3)):
    """Tile-wise 1-corr aggregated per frame. Returns (T,)."""
    T, H, W = mip.shape
    ny, nx = tiles_yx
    ys = np.linspace(0, H, ny+1, dtype=int)
    xs = np.linspace(0, W, nx+1, dtype=int)
    scores = np.full(T, np.nan, dtype=np.float32)
    prev = norm_frame(mip[0])
    for t in range(1, T):
        cur = norm_frame(mip[t])
        tile_vals = []
        for i in range(ny):
            for j in range(nx):
                y0,y1 = ys[i], ys[i+1]
                x0,x1 = xs[j], xs[j+1]
                a = prev[y0:y1, x0:x1].ravel()
                b = cur [y0:y1, x0:x1].ravel()
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() >= MIN_VALID_PIX:
                    aa = a[m]; bb = b[m]
                    aa = (aa - aa.mean()) / (aa.std() + EPS)
                    bb = (bb - bb.mean()) / (bb.std() + EPS)
                    corr = float(np.clip(np.dot(aa, bb) / max(1, len(aa)), -1.0, 1.0))
                    tile_vals.append(1.0 - corr)
        if tile_vals:
            if AGGREGATE == "p90":
                scores[t] = np.percentile(tile_vals, 90.0)
            else:  # "max"
                scores[t] = np.max(tile_vals)
        prev = cur
        if (t+1)%200==0: print(f"  corr-tiles {t+1}/{T}")
    return scores

# ===================== MAIN =====================

def main():
    # Load or build MIP
    if USE_MIP_OR_DFF and MIP_OR_DFF_PATH.exists():
        print(f"Using provided MIP/ΔF/F: {MIP_OR_DFF_PATH}")
        mip = load_mip_movie(MIP_OR_DFF_PATH)
    else:
        print("No MIP/ΔF/F provided; deriving MIP from RAW.")
        mip = load_mip_from_raw(RAW_STACK_PATH, Y_CROP_RAW)

    T_mip = mip.shape[0]
    with tifffile.TiffFile(RAW_STACK_PATH) as tif:
        T_raw, Z, Y, X = tif.series[0].shape
    if T_mip != T_raw:
        print(f"WARNING: MIP T={T_mip} but RAW T={T_raw}. Truncating to RAW.")
        mip = mip[:T_raw]
    T = mip.shape[0]

    # Manual exclusions
    manual = build_exclude_set(T, EXCLUDE_RANGES_STR, EXCLUDE_RANGES_FILE)
    print(f"Manual exclusions: {len(manual)}")

    # Motion scores
    if TILES_YX and tuple(TILES_YX) != (1,1):
        print(f"Detector: TILE-wise correlation, tiles={TILES_YX}, agg={AGGREGATE}")
        scores = corr_scores_tiled(mip, tiles_yx=TILES_YX)
    else:
        print("Detector: GLOBAL correlation")
        scores = corr_scores_global(mip)

    # Rolling MAD threshold
    med_loc = rolling_median(scores, ROLL_WIN)
    mad_loc = rolling_mad(scores, ROLL_WIN) + EPS
    thr     = med_loc + K_MAD * mad_loc
    hits    = np.where(scores > thr)[0]

    # Pad neighbors
    padset = set()
    for f in hits.tolist():
        for k in range(max(0, f - PAD_NEIGHBOR), min(T, f + PAD_NEIGHBOR + 1)):
            padset.add(k)
    auto = sorted(padset)
    print(f"Auto-detected frames (±{PAD_NEIGHBOR}): {len(auto)}")

    # Union with manual
    excluded = sorted(set(auto).union(manual))
    print(f"TOTAL excluded frames: {len(excluded)}")
    if excluded and len(excluded) <= 80:
        print(f"indices (preview): {excluded[:80]}{' ...' if len(excluded)>80 else ''}")

    # QC plot
    plt.figure(figsize=(12,4))
    plt.plot(scores, label="1 - corr(MIP_t, MIP_{t-1})")
    plt.plot(thr, 'r--', label=f"rolling thr (win={ROLL_WIN}, k={K_MAD})")
    if auto:
        xs = np.array(auto, int); ys = scores[np.clip(xs, 0, len(scores)-1)]
        plt.scatter(xs, ys, s=10, label="auto", zorder=3)
    if manual:
        xs = np.array(manual, int); ys = scores[np.clip(xs, 0, len(scores)-1)]
        plt.scatter(xs, ys, s=18, marker='x', label="manual", zorder=4)
    tile_lbl = f"tiles={TILES_YX}, agg={AGGREGATE}" if TILES_YX else "global"
    plt.title(f"Motion score (corr; {tile_lbl})")
    plt.xlabel("Frame"); plt.ylabel("Score")
    plt.legend(); plt.tight_layout(); plt.savefig(QC_PDF); plt.close()
    print(f"Saved QC → {QC_PDF}")

    # Write NaN-masked RAW
    excluded_mask = np.zeros(T_raw, dtype=bool)
    if excluded:
        ex = np.array([e for e in excluded if e < T_raw], int)
        excluded_mask[ex] = True

    print(f"Writing NaN-masked RAW → {FILTERED_STACK_PATH}")
    with tifffile.TiffWriter(FILTERED_STACK_PATH, bigtiff=True) as writer:
        with tifffile.TiffFile(RAW_STACK_PATH) as tif:
            for t in range(T_raw):
                vol = tif.series[0].asarray()[t].astype(np.float32)  # (Z,Y,X)
                if excluded_mask[t]:
                    vol[...] = np.nan
                writer.write(vol)
                if (t+1)%100==0: print(f"  wrote {t+1}/{T_raw}")
                del vol
    gc.collect()

    # Save lists
    np.save(EXCLUDED_NPY_PATH, np.array(sorted(np.where(excluded_mask)[0]), dtype=np.int32))
    with open(EXCLUDED_TXT_PATH, "w") as f:
        f.write(",".join(map(str, np.where(excluded_mask)[0])) + "\n")
    print(f"Saved excluded frames → {EXCLUDED_NPY_PATH.name}, {EXCLUDED_TXT_PATH.name}")
    print("Done. Feed preprocessed/raw_nanmasked.tif into Module 1.")

if __name__ == "__main__":
    main()
