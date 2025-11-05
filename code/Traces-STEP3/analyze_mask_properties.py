#!/usr/bin/env python
"""
Analyze Mask Properties: Plot distributions of mask lengths, volumes, and spike occurrences

Generates:
1. Mask volume distribution (in pixels)
2. Mask length distribution (max extent in any dimension)
3. Spike occurrence distribution (number of events per mask)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile
import pandas as pd
from pathlib import Path
from scipy.ndimage import label
from scipy import stats

# Set Arial font
mpl.rcParams['font.family'] = 'Arial'

# Configuration
DATE = "2025-10-29"
MOUSE = "rAi162_15"
RUN = "run1-crop"

# Analysis options
TEST_SMOOTHED = True  # Also test smoothed version for comparison

# Selected masks (set USE_ALL=True to analyze all masks)
SELECTED_MASKS = [
    "dend_001","dend_003","dend_008", "dend_012", "dend_014", "dend_015", "dend_016", "dend_019"
]
USE_ALL = True

# Spike detection parameters
SPIKE_THRESHOLD = 2.0  # ΔF/F threshold for spike detection
MIN_SPIKE_DURATION = 3  # Minimum frames for a spike

# Paths
BASE = Path("/Volumes/IMAC/data") / DATE / MOUSE / RUN
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"  # Use curated masks
TRACE_FILE = BASE / "traces" / "dff_traces_curated_bgsub.csv"
OUTPUT_FOLDER = BASE / "mask_analysis"
OUTPUT_FOLDER.mkdir(exist_ok=True)

def detect_spikes(trace, threshold=SPIKE_THRESHOLD, min_duration=MIN_SPIKE_DURATION):
    """Detect spike events in a trace"""
    above_threshold = trace > threshold
    
    # Find connected regions above threshold
    labeled, n_spikes = label(above_threshold)
    
    # Filter by minimum duration
    valid_spikes = 0
    for i in range(1, n_spikes + 1):
        spike_mask = labeled == i
        if np.sum(spike_mask) >= min_duration:
            valid_spikes += 1
    
    return valid_spikes

def calculate_mask_length(mask):
    """Calculate dendrite length - curved path if Y>X, otherwise Y-extent"""
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return 0
    
    # Check if Y dimension is longer than X
    y_coords = coords[1]
    x_coords = coords[2]
    y_extent = np.max(y_coords) - np.min(y_coords) + 1
    x_extent = np.max(x_coords) - np.min(x_coords) + 1
    
    if y_extent > x_extent:
        # Calculate curved path length using skeleton
        from skimage.morphology import skeletonize
        from scipy.ndimage import distance_transform_edt
        
        # Create 2D projection (max over Z)
        mask_2d = np.max(mask, axis=0)
        
        # Skeletonize to get centerline
        skeleton = skeletonize(mask_2d)
        
        if np.sum(skeleton) > 0:
            # Calculate path length along skeleton
            skel_coords = np.where(skeleton)
            if len(skel_coords[0]) > 1:
                # Sort coordinates by Y to trace path
                y_skel, x_skel = skel_coords
                sorted_idx = np.argsort(y_skel)
                y_sorted = y_skel[sorted_idx]
                x_sorted = x_skel[sorted_idx]
                
                # Calculate cumulative distance along path
                path_length = 0
                for i in range(1, len(y_sorted)):
                    dy = y_sorted[i] - y_sorted[i-1]
                    dx = x_sorted[i] - x_sorted[i-1]
                    path_length += np.sqrt(dy**2 + dx**2)
                
                return path_length
        
        # Fallback to Y-extent if skeletonization fails
        return y_extent
    else:
        # Use simple Y-extent for non-elongated masks
        return y_extent

def main():
    print("Loading masks...")
    
    # Load masks
    if USE_ALL:
        mask_files = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
        mask_names = [f.stem.replace("_labelmap", "") for f in mask_files]
        print(f"Using all {len(mask_names)} available masks")
    else:
        mask_names = SELECTED_MASKS
        print(f"Using {len(mask_names)} selected masks")
    
    # Analyze mask properties
    volumes = []
    lengths = []
    widths = []
    spike_counts = []
    valid_names = []
    
    for name in mask_names:
        mask_path = MASK_FOLDER / f"{name}_labelmap.tif"
        if not mask_path.exists():
            print(f"Warning: {name} not found")
            continue
        
        # Load mask and calculate properties
        mask = tifffile.imread(mask_path).astype(bool)
        volume = np.sum(mask)
        length = calculate_mask_length(mask)
        
        # Calculate width (X-dimension extent)
        coords = np.where(mask)
        if len(coords[0]) > 0:
            x_coords = coords[2]  # X is third dimension
            width = np.max(x_coords) - np.min(x_coords) + 1
        else:
            width = 0
        
        volumes.append(volume)
        lengths.append(length)
        widths.append(width)
        valid_names.append(name)
    
    print(f"Analyzed {len(valid_names)} masks")
    
    # Load traces and count spikes
    if TRACE_FILE.exists():
        traces_df = pd.read_csv(TRACE_FILE, index_col=0)
        
        for name in valid_names:
            if name in traces_df.columns:
                trace = traces_df[name].values
                n_spikes = detect_spikes(trace)
                spike_counts.append(n_spikes)
            else:
                spike_counts.append(0)
    else:
        print(f"Trace file not found: {TRACE_FILE}")
        spike_counts = [0] * len(valid_names)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Volume distribution
    ax = axes[0, 0]
    ax.hist(volumes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Volume (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Mask Volume Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_vol = np.mean(volumes)
    median_vol = np.median(volumes)
    ax.axvline(mean_vol, color='red', linestyle='--', label=f'Mean: {mean_vol:.0f}')
    ax.axvline(median_vol, color='orange', linestyle='--', label=f'Median: {median_vol:.0f}')
    ax.legend()
    
    # 2. Length distribution
    ax = axes[0, 1]
    ax.hist(lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_xlabel('Y-axis Length (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Dendrite Length Distribution (Y-axis)')
    ax.grid(True, alpha=0.3)
    
    # Multiple bimodality tests
    skew = stats.skew(lengths)
    kurt = stats.kurtosis(lengths)
    n = len(lengths)
    
    # Sarle's bimodality coefficient
    bimodality_coeff = (skew**2 + 1) / (kurt + 3 * (n-1)**2 / ((n-2)*(n-3)))
    
    # Silverman's test: compare kernel density modes
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(lengths)
    x_range = np.linspace(min(lengths), max(lengths), 200)
    density = kde(x_range)
    
    # Find peaks (local maxima) - more sensitive
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(density, height=np.max(density)*0.05, distance=10)  # 5% height, min distance
    n_modes = len(peaks)
    
    # Valley test: check if there's a significant dip between peaks
    valley_ratio = 0
    if n_modes >= 2:
        # Find minimum between first two peaks
        peak1, peak2 = peaks[0], peaks[1]
        valley_idx = np.argmin(density[peak1:peak2]) + peak1
        valley_height = density[valley_idx]
        min_peak_height = min(density[peak1], density[peak2])
        valley_ratio = valley_height / min_peak_height
    
    # Add statistics
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    ax.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.0f}')
    ax.axvline(median_len, color='orange', linestyle='--', label=f'Median: {median_len:.0f}')
    
    # Add test results
    bimodal_text = f'Modes: {n_modes}\nSarle: {bimodality_coeff:.3f}'
    if n_modes >= 2 and valley_ratio < 0.8:
        bimodal_text += '\nValley: {:.2f}\n(Bimodal)'.format(valley_ratio)
    elif bimodality_coeff > 0.555:
        bimodal_text += '\n(Bimodal)'
    elif bimodality_coeff > 0.4:
        bimodal_text += '\n(Weak bimodal)'
    else:
        bimodal_text += '\n(Unimodal)'
    
    ax.text(0.7, 0.75, bimodal_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=9)
    ax.legend()
    
    # 3. Length vs Spike count
    ax = axes[1, 0]
    ax.scatter(lengths, spike_counts, alpha=0.7, color='salmon', s=50)
    ax.set_xlabel('Y-axis Length (pixels)')
    ax.set_ylabel('Number of Spikes')
    ax.set_title('Spike Count vs Dendrite Length')
    ax.grid(True, alpha=0.3)
    
    # Add correlation
    if len(lengths) > 1 and max(spike_counts) > 0:
        corr, p_val = stats.pearsonr(lengths, spike_counts)
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Volume vs Spike count
    ax = axes[1, 1]
    ax.scatter(volumes, spike_counts, alpha=0.7, color='lightcoral', s=50)
    ax.set_xlabel('Volume (pixels)')
    ax.set_ylabel('Number of Spikes')
    ax.set_title('Spike Count vs Volume')
    ax.grid(True, alpha=0.3)
    
    # Add correlation
    if len(volumes) > 1 and max(spike_counts) > 0:
        corr, p_val = stats.pearsonr(volumes, spike_counts)
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plots
    output_pdf = OUTPUT_FOLDER / "mask_properties_analysis.pdf"
    output_svg = OUTPUT_FOLDER / "mask_properties_analysis.svg"
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    
    # Save summary statistics
    summary_data = {
        'mask_name': valid_names,
        'volume_pixels': volumes,
        'max_length_pixels': lengths,
        'spike_count': spike_counts
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = OUTPUT_FOLDER / "mask_properties_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    # Print summary
    print(f"\nSummary Statistics:")
    print(f"Volume: {np.mean(volumes):.1f} ± {np.std(volumes):.1f} pixels")
    print(f"Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} pixels")
    print(f"Spikes: {np.mean(spike_counts):.1f} ± {np.std(spike_counts):.1f} events")
    print(f"\nBimodality Tests (Length - RAW):")
    print(f"Number of modes detected: {n_modes}")
    print(f"Sarle's bimodality coefficient: {bimodality_coeff:.3f}")
    if n_modes >= 2:
        print(f"Valley ratio (lower = more separated): {valley_ratio:.3f}")
        if valley_ratio < 0.8:
            print("→ Strong evidence for bimodal distribution (distinct peaks)")
        else:
            print("→ Weak bimodal (peaks not well separated)")
    elif bimodality_coeff > 0.555:
        print("→ Bimodal by coefficient but single mode detected")
    else:
        print("→ Unimodal distribution")
    
    # Test smoothed version for comparison
    if TEST_SMOOTHED and len(lengths) > 5:
        from scipy.ndimage import gaussian_filter1d
        hist, bin_edges = np.histogram(lengths, bins=20)
        smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=1.0)
        smooth_peaks, _ = find_peaks(smoothed_hist, height=np.max(smoothed_hist)*0.1)
        n_smooth_modes = len(smooth_peaks)
        
        print(f"\nSmoothed comparison:")
        print(f"Modes in smoothed histogram: {n_smooth_modes}")
        print(f"Recommendation: Use {'smoothed' if n_smooth_modes > n_modes else 'raw'} data")
    
    print(f"✅ Analysis plots: {output_pdf}")
    print(f"✅ Summary data: {summary_csv}")

if __name__ == "__main__":
    main()