#!/usr/bin/env python
"""
Find Dendrite Branches: Group masks that overlap spatially and are active simultaneously

Logic:
1. Load all 3D masks and activity traces
2. Find pairs with spatial overlap (>threshold)
3. Check temporal correlation during active periods
4. Group connected masks into dendrites
5. Create merged masks for each dendrite
"""

import numpy as np
import tifffile
import pandas as pd
from pathlib import Path
from scipy.ndimage import label
from collections import defaultdict

# Configuration
DATE = "2025-10-29"
MOUSE = "rAi162_15"
RUN = "run1-crop"

# Thresholds
SPATIAL_OVERLAP_THRESHOLD = 0.05  # Minimum Jaccard index for spatial overlap
TEMPORAL_CORR_THRESHOLD = 0.3     # Minimum correlation during active periods
MIN_ACTIVE_FRAMES = 10            # Minimum frames to consider for correlation

# Paths
BASE = Path("/Volumes/IMAC/data") / DATE / MOUSE / RUN
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"  # Use curated masks
TRACE_FILE = BASE / "traces" / "dff_traces_curated_bgsub.csv"
OUTPUT_FOLDER = BASE / "dendrite_branches"
OUTPUT_FOLDER.mkdir(exist_ok=True)

def load_masks():
    """Load all 3D masks"""
    masks = {}
    mask_files = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    
    for mask_path in mask_files:
        name = mask_path.stem.replace("_labelmap", "")
        masks[name] = tifffile.imread(mask_path).astype(bool)
    
    return masks

def calculate_spatial_overlap(mask1, mask2):
    """Calculate Jaccard index between two 3D masks"""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union if union > 0 else 0

def calculate_temporal_correlation(trace1, trace2, threshold=1.0):
    """Calculate correlation during periods when both traces are active"""
    # Find active periods (above threshold)
    active1 = trace1 > threshold
    active2 = trace2 > threshold
    both_active = active1 & active2
    
    if np.sum(both_active) < MIN_ACTIVE_FRAMES:
        return 0.0
    
    # Calculate correlation during co-active periods
    corr = np.corrcoef(trace1[both_active], trace2[both_active])[0, 1]
    return corr if not np.isnan(corr) else 0.0

def find_connected_components(pairs):
    """Group pairs into connected components (dendrites)"""
    # Build adjacency list
    graph = defaultdict(set)
    all_nodes = set()
    
    for mask1, mask2 in pairs:
        graph[mask1].add(mask2)
        graph[mask2].add(mask1)
        all_nodes.update([mask1, mask2])
    
    # Find connected components using DFS
    visited = set()
    components = []
    
    def dfs(node, component):
        if node in visited:
            return
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, component)
    
    for node in all_nodes:
        if node not in visited:
            component = []
            dfs(node, component)
            if len(component) > 1:  # Only keep multi-mask dendrites
                components.append(sorted(component))
    
    return components

def create_merged_mask(masks, mask_names):
    """Create merged mask from multiple individual masks"""
    merged = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for name in mask_names:
        if name in masks:
            merged |= masks[name]
    return merged

def main():
    print("Loading masks and traces...")
    masks = load_masks()
    
    if not TRACE_FILE.exists():
        print(f"Trace file not found: {TRACE_FILE}")
        return
    
    traces_df = pd.read_csv(TRACE_FILE, index_col=0)
    
    print(f"Loaded {len(masks)} masks and {len(traces_df.columns)} traces")
    
    # Find spatially overlapping pairs
    print("Finding spatially overlapping pairs...")
    spatial_pairs = []
    mask_names = list(masks.keys())
    
    for i in range(len(mask_names)):
        for j in range(i+1, len(mask_names)):
            name1, name2 = mask_names[i], mask_names[j]
            overlap = calculate_spatial_overlap(masks[name1], masks[name2])
            
            if overlap > SPATIAL_OVERLAP_THRESHOLD:
                spatial_pairs.append((name1, name2, overlap))
    
    print(f"Found {len(spatial_pairs)} spatially overlapping pairs")
    
    # Filter by temporal correlation
    print("Checking temporal correlations...")
    valid_pairs = []
    
    for name1, name2, overlap in spatial_pairs:
        if name1 in traces_df.columns and name2 in traces_df.columns:
            trace1 = traces_df[name1].values
            trace2 = traces_df[name2].values
            
            corr = calculate_temporal_correlation(trace1, trace2)
            
            # Debug specific pair
            if (name1 == "dend_025" and name2 == "dend_027") or (name1 == "dend_027" and name2 == "dend_025"):
                print(f"  DEBUG {name1} ↔ {name2}: spatial={overlap:.3f}, temporal={corr:.3f} (threshold={TEMPORAL_CORR_THRESHOLD})")
            
            if corr > TEMPORAL_CORR_THRESHOLD:
                valid_pairs.append((name1, name2))
                print(f"  {name1} ↔ {name2}: spatial={overlap:.3f}, temporal={corr:.3f}")
            elif overlap > 0.01:  # Show failed pairs with some overlap
                print(f"  FAILED {name1} ↔ {name2}: spatial={overlap:.3f}, temporal={corr:.3f}")
    
    print(f"Found {len(valid_pairs)} valid branch pairs")
    
    # Group into dendrites
    dendrites = find_connected_components(valid_pairs)
    print(f"Identified {len(dendrites)} multi-branch dendrites")
    
    # Create merged masks and save results
    results = []
    for i, dendrite_masks in enumerate(dendrites):
        dendrite_name = f"dendrite_{i:03d}"
        print(f"{dendrite_name}: {dendrite_masks}")
        
        # Create merged mask
        merged_mask = create_merged_mask(masks, dendrite_masks)
        
        # Save merged mask
        output_path = OUTPUT_FOLDER / f"{dendrite_name}_merged.tif"
        tifffile.imwrite(output_path, merged_mask.astype(np.uint8))
        
        results.append({
            'dendrite': dendrite_name,
            'branches': ','.join(dendrite_masks),
            'n_branches': len(dendrite_masks),
            'total_voxels': np.sum(merged_mask)
        })
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_FOLDER / "dendrite_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Create merged dendrite outputs
    if len(dendrites) > 0:
        print("Creating merged dendrite outputs...")
        
        for i, dendrite_masks in enumerate(dendrites):
            dendrite_name = f"dendrite_{i:03d}"
            
            # Create merged mask
            merged_mask = create_merged_mask(masks, dendrite_masks)
            
            # Save merged mask
            mask_path = OUTPUT_FOLDER / f"{dendrite_name}_merged_labelmap.tif"
            tifffile.imwrite(mask_path, merged_mask.astype(np.uint16))
            
            # Create preview with MIP + trace (like trace_curated_preview)
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from skimage.measure import find_contours
            
            # Get merged trace data
            merged_trace = None
            if TRACE_FILE.exists():
                traces_df = pd.read_csv(TRACE_FILE, index_col=0)
                branch_traces = []
                
                for branch_name in dendrite_masks:
                    if branch_name in traces_df.columns:
                        branch_traces.append(traces_df[branch_name].values)
                
                if branch_traces:
                    # Volume-weighted average
                    weights = [np.sum(masks[name]) for name in dendrite_masks if name in masks]
                    if len(weights) == len(branch_traces):
                        weights = np.array(weights) / np.sum(weights)
                        merged_trace = np.average(branch_traces, axis=0, weights=weights)
                    else:
                        merged_trace = np.mean(branch_traces, axis=0)
            
            # Create figure with MIP + trace (2:1 ratio for traces)
            if merged_trace is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 2]})
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
                ax2 = None
            
            # MIP with colored branches
            mip_mask = np.max(merged_mask, axis=0).astype(bool)
            ax1.imshow(np.zeros_like(mip_mask), cmap="gray")
            
            # Nicer color palette
            if len(dendrite_masks) <= 6:
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'][:len(dendrite_masks)]
            else:
                colors = cm.Set3(np.linspace(0, 1, len(dendrite_masks)))
            for j, branch_name in enumerate(dendrite_masks):
                if branch_name in masks:
                    branch_mip = np.max(masks[branch_name], axis=0)
                    contours = find_contours(branch_mip.astype(np.uint8), 0.5)
                    for contour in contours:
                        ax1.plot(contour[:, 1], contour[:, 0], color=colors[j], linewidth=2)
            
            ax1.set_title(f"{dendrite_name} MIP")
            ax1.axis("off")
            
            # Individual branch traces with matching colors
            if ax2 is not None:
                if TRACE_FILE.exists():
                    traces_df = pd.read_csv(TRACE_FILE, index_col=0)
                    time_axis = np.arange(len(traces_df)) / 10.0  # Assuming 10Hz
                    
                    for j, branch_name in enumerate(dendrite_masks):
                        if branch_name in traces_df.columns:
                            trace = traces_df[branch_name].values
                            ax2.plot(time_axis, trace, color=colors[j], linewidth=1.5, label=branch_name, alpha=0.8)
                    
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('ΔF/F (%)')
                    ax2.set_title(f"{dendrite_name} Branch Traces")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(fontsize=8)
            
            fig.tight_layout()
            preview_path = OUTPUT_FOLDER / f"{dendrite_name}_preview.png"
            fig.savefig(preview_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            # Create merged trace if traces available
            if TRACE_FILE.exists():
                traces_df = pd.read_csv(TRACE_FILE, index_col=0)
                branch_traces = []
                
                for branch_name in dendrite_masks:
                    if branch_name in traces_df.columns:
                        branch_traces.append(traces_df[branch_name].values)
                
                if branch_traces:
                    # Average traces weighted by mask volume
                    weights = [np.sum(masks[name]) for name in dendrite_masks if name in masks]
                    if len(weights) == len(branch_traces):
                        weights = np.array(weights) / np.sum(weights)
                        merged_trace = np.average(branch_traces, axis=0, weights=weights)
                    else:
                        merged_trace = np.mean(branch_traces, axis=0)
                    
                    # Save merged trace
                    trace_df = pd.DataFrame({dendrite_name: merged_trace}, index=traces_df.index)
                    trace_path = OUTPUT_FOLDER / f"{dendrite_name}_trace.csv"
                    trace_df.to_csv(trace_path)
    
    print(f"✅ Results saved to: {OUTPUT_FOLDER}")
    print(f"✅ Summary: {summary_path}")

if __name__ == "__main__":
    main()