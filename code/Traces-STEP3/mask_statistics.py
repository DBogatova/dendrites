#!/usr/bin/env python
"""
Quick Morphology–Activity Analysis
----------------------------------
Computes 3D dendrite geometry (length, radius, tortuosity),
fits a 2-component GMM to split short/thin vs long/thick branches,
and compares calcium activity metrics (spike rate, amplitude, kinetics).

Requirements:
    pip install numpy pandas matplotlib scikit-image scikit-learn scipy tifffile seaborn
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label

# Simple 3D skeletonization function
def skeletonize_3d(mask):
    """Simple 3D skeletonization by applying 2D skeletonize to each Z-slice"""
    skel = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        if mask[z].any():
            skel[z] = skeletonize(mask[z])
    return skel
from sklearn.mixture import GaussianMixture
from scipy.stats import skew, kurtosis, gaussian_kde

# ------------------- CONFIG -------------------
DATE  = "2025-10-29"
MOUSE = "rAi162_15"
RUN   = "run1-crop"
VOXEL_SIZE = (4.7, 1.0, 1.2)   # (z,y,x) µm per voxel
SPIKE_THRESHOLD = 2.0
MIN_SPIKE_DURATION = 3
FRAME_RATE = 4.0  # Hz (→ spikes/min)
BASE = Path("/Volumes/IMAC/data") / DATE / MOUSE / RUN
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
TRACE_FILE = BASE / "traces" / "dff_traces_curated_bgsub.csv"
OUTDIR = BASE / "mask_analysis_quick"
OUTDIR.mkdir(exist_ok=True)

# ------------------- HELPERS -------------------
def detect_spikes(trace, thr=SPIKE_THRESHOLD, min_dur=MIN_SPIKE_DURATION):
    above = trace > thr
    labeled, n = label(above)
    count = 0
    for i in range(1, n + 1):
        if np.sum(labeled == i) >= min_dur:
            count += 1
    return count

def event_metrics(trace, thr=SPIKE_THRESHOLD, min_dur=MIN_SPIKE_DURATION):
    """Return number, mean amplitude, rise, decay (in frames)"""
    labeled, n = label(trace > thr)
    amps, rises, decays = [], [], []
    for i in range(1, n + 1):
        mask = labeled == i
        if np.sum(mask) < min_dur:
            continue
        idx = np.where(mask)[0]
        peak = np.max(trace[idx])
        amps.append(peak)
        rises.append(idx[np.argmax(trace[idx])] - idx[0])
        decays.append(idx[-1] - idx[np.argmax(trace[idx])])
    if not amps:
        return 0, np.nan, np.nan, np.nan
    return len(amps), np.mean(amps), np.mean(rises), np.mean(decays)

def mask_geometry(mask):
    """Return (volume µm³, length µm, radius µm, tortuosity)"""
    zres, yres, xres = VOXEL_SIZE
    voxel_vol = zres * yres * xres
    volume = np.sum(mask) * voxel_vol

    # Skeletonize 3D
    skel = skeletonize_3d(mask)
    skel_pts = np.argwhere(skel)
    if len(skel_pts) < 2:
        return volume, 0, 0, np.nan

    # Length along skeleton (connectivity via neighbors)
    length = 0.0
    for p in skel_pts:
        neigh = skel[
            max(p[0]-1,0):p[0]+2,
            max(p[1]-1,0):p[1]+2,
            max(p[2]-1,0):p[2]+2]
        length += np.sum(neigh) - 1
    length = length/2 * np.mean(VOXEL_SIZE)  # crude µm estimate

    # Straight-line distance
    pmin, pmax = np.min(skel_pts,axis=0), np.max(skel_pts,axis=0)
    straight = np.linalg.norm((pmax - pmin) * VOXEL_SIZE)
    tortuosity = np.clip(length/straight, 1.0, None)

    # Radius (distance transform inside mask)
    dist = distance_transform_edt(mask, sampling=VOXEL_SIZE)
    mean_r = np.mean(dist[mask])
    return volume, length, mean_r, tortuosity

# ------------------- MAIN -------------------
def main():
    print("Loading curated masks ...")
    mask_files = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    if not mask_files:
        raise FileNotFoundError("No masks found!")

    traces = pd.read_csv(TRACE_FILE, index_col=0)
    results = []

    for f in mask_files:
        name = f.stem.replace("_labelmap","")
        mask = tifffile.imread(f).astype(bool)
        volume, length, radius, tort = mask_geometry(mask)

        if name in traces.columns:
            trace = traces[name].values
            n, amp, rise, decay = event_metrics(trace)
        else:
            n, amp, rise, decay = 0, np.nan, np.nan, np.nan

        rate = n * (60.0 / len(trace) * FRAME_RATE) if len(trace)>0 else 0
        results.append(dict(name=name, volume_um3=volume,
                            length_um=length, radius_um=radius,
                            tortuosity=tort, spikes=n,
                            spike_rate_per_min=rate,
                            amp_mean=amp, rise_mean=rise, decay_mean=decay))

    df = pd.DataFrame(results)
    df.dropna(subset=["length_um"], inplace=True)
    df.to_csv(OUTDIR/"mask_quick_metrics.csv", index=False)

    # ------------- GMM grouping -------------
    feats = df[["length_um","radius_um","tortuosity"]].fillna(0).values
    gmm = GaussianMixture(n_components=2, random_state=0).fit(feats)
    df["group"] = gmm.predict(feats)
    # label groups by mean length (short=0,long=1)
    means = df.groupby("group")["length_um"].mean().sort_values()
    mapping = {means.index[0]:"short_thin", means.index[1]:"long_thick"}
    df["group"] = df["group"].map(mapping)

    df.to_csv(OUTDIR/"mask_quick_metrics_gmm.csv", index=False)
    print(df.groupby("group")[["length_um","radius_um","tortuosity","spike_rate_per_min"]].mean())

    # ------------- PLOTS -------------
    plt.rcParams['font.family'] = 'Arial'
    fig, axs = plt.subplots(2,3, figsize=(15,9))

    # Morphology distributions (box plots instead of violin)
    for i,(col,ax) in enumerate(zip(["length_um","radius_um","tortuosity"], axs[0])):
        groups = df.groupby("group")[col]
        ax.boxplot([groups.get_group(g).dropna() for g in groups.groups.keys()], 
                   labels=list(groups.groups.keys()))
        ax.set_title(col.replace("_um"," (µm)").capitalize())
        ax.grid(True, alpha=0.3)

    # Activity comparisons
    for i,(col,ax) in enumerate(zip(["spike_rate_per_min","amp_mean","rise_mean"], axs[1])):
        groups = df.groupby("group")[col]
        ax.boxplot([groups.get_group(g).dropna() for g in groups.groups.keys()], 
                   labels=list(groups.groups.keys()))
        ax.set_title(col.replace("_"," ").capitalize())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTDIR/"quick_morpho_activity_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # ECDFs for spike rate
    plt.figure(figsize=(6,4))
    for grp, d in df.groupby("group"):
        x = np.sort(d["spike_rate_per_min"].dropna())
        y = np.arange(1,len(x)+1)/len(x)
        plt.step(x,y,where="post",label=f"{grp} (N={len(x)})")
    plt.xlabel("Spike rate (events/min)")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR/"ecdf_spike_rate.pdf", bbox_inches="tight")

    print(f"✅ Saved metrics to {OUTDIR}")
    print(f"✅ Figures: quick_morpho_activity_comparison.pdf + ecdf_spike_rate.pdf")

if __name__ == "__main__":
    main()
