#!/usr/bin/env python
"""
Check overlap between selected masks
"""

import numpy as np
import tifffile
import pandas as pd
from pathlib import Path

# Configuration
DATE = "2025-08-06"
MOUSE = "organoid"
RUN = "run4-crop"

SELECTED_NAMES = [
    "dend_001","dend_003","dend_005","dend_006","dend_008","dend_011","dend_012", "dend_013", "dend_014", "dend_015", "dend_016", "dend_019"
    ]
USE_ALL = True  # Set to True to use all available masks

# Paths
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "data" / DATE / MOUSE / RUN
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"

def main():
    print("Checking mask overlaps...")
    
    # Load masks
    masks = {}
    if USE_ALL:
        # Use all available masks
        mask_files = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
        for mask_path in mask_files:
            name = mask_path.stem.replace("_labelmap", "")
            masks[name] = tifffile.imread(mask_path).astype(bool)
        print(f"Using all {len(masks)} available masks")
    else:
        # Use selected masks
        for name in SELECTED_NAMES:
            mask_path = MASK_FOLDER / f"{name}_labelmap.tif"
            if mask_path.exists():
                masks[name] = tifffile.imread(mask_path).astype(bool)
            else:
                print(f"Warning: {name} not found")
        print(f"Using {len(masks)} selected masks")
    
    if len(masks) < 2:
        print("Need at least 2 masks to check overlap")
        return
    
    # Check pairwise overlaps
    results = []
    names = list(masks.keys())
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            mask1, mask2 = masks[name1], masks[name2]
            
            # Calculate overlap
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)
            size1, size2 = np.sum(mask1), np.sum(mask2)
            
            # Overlap metrics
            jaccard = intersection / union if union > 0 else 0
            overlap_pct1 = 100 * intersection / size1 if size1 > 0 else 0
            overlap_pct2 = 100 * intersection / size2 if size2 > 0 else 0
            
            results.append({
                'mask1': name1,
                'mask2': name2,
                'intersection_voxels': intersection,
                'jaccard_index': jaccard,
                'overlap_pct_mask1': overlap_pct1,
                'overlap_pct_mask2': overlap_pct2
            })
            
            if intersection > 0:
                print(f"{name1} ↔ {name2}: {intersection} voxels ({overlap_pct1:.1f}% / {overlap_pct2:.1f}%)")
    
    # Summary
    df = pd.DataFrame(results)
    high_overlap = df[df['jaccard_index'] > 0.1]
    
    print(f"\nSummary:")
    print(f"- Total pairs checked: {len(results)}")
    print(f"- Pairs with >10% Jaccard overlap: {len(high_overlap)}")
    
    if len(high_overlap) > 0:
        print("\nHigh overlap pairs:")
        for _, row in high_overlap.iterrows():
            print(f"  {row['mask1']} ↔ {row['mask2']}: {row['jaccard_index']:.3f} Jaccard")

if __name__ == "__main__":
    main()