#!/usr/bin/env python
"""
Module 2: Automatic Mask Generation

This script automatically generates masks for dendrites by:
1. Loading event crops from Module 1
2. Thresholding to identify active regions
3. Cleaning and filtering masks based on size and morphology
4. Deduplicating similar masks
5. Saving unique masks and visualizations
"""

import tifffile
import numpy as np
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, ball, label
from skimage.measure import regionprops
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import gc

# === CONFIGURATION ===
DATE = "2025-03-26"
MOUSE = "organoid"
RUN = "run6"

# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
EVENT_FOLDER = BASE / "preprocessed" / "event_crops"
OUTPUT_LABELMAP_FOLDER = BASE / "labelmaps"
PREVIEW_FOLDER = BASE / "labelmap_previews"
OUTPUT_LABELMAP_FOLDER.mkdir(parents=True, exist_ok=True)
PREVIEW_FOLDER.mkdir(parents=True, exist_ok=True)

# === PARAMETERS ===
VOXEL_SIZE = (4.7, 0.5, 0.6)  # (Z, Y, X) in microns
VOXEL_VOL = np.prod(VOXEL_SIZE)
MIN_VOL = 600       # Minimum dendrite volume in µm³
MAX_VOL = 15000     # Maximum dendrite volume in µm³
Y_CROP = 0          # Number of pixels to crop from bottom of Y dimension
INTENSITY_PERCENTILE = 99.9  # Percentile for thresholding
MAX_DENDRITES_TOTAL = 200    # Maximum number of dendrites to extract
DUPLICATE_THRESHOLD = 0.9    # Dice coefficient threshold for duplicate detection
MIN_EVENT_LENGTH = 1         # Minimum number of frames in an event
MAX_FRAME_GAP = 2            # Maximum gap between frames in an event

def group_frames(frames, gap):
    """
    Group consecutive frame numbers into events, allowing for small gaps.
    
    Args:
        frames: Array of frame numbers
        gap: Maximum allowed gap between consecutive frames to be in same group
        
    Returns:
        List of lists, where each inner list contains frame numbers for one event
    """
    if len(frames) == 0:
        return []
        
    groups = []
    group = [frames[0]]
    
    for f in frames[1:]:
        if f - group[-1] <= gap:
            group.append(f)
        else:
            groups.append(group)
            group = [f]
            
    groups.append(group)
    return groups

def dice_coeff(a, b):
    """
    Calculate Dice coefficient between two binary masks.
    
    Args:
        a, b: Binary masks to compare
        
    Returns:
        Dice coefficient (0-1), where 1 means perfect overlap
    """
    # Return 0 if shapes don't match
    if a.shape != b.shape:
        return 0
        
    intersection = np.logical_and(a, b).sum()
    volume = a.sum() + b.sum()
    return 2 * intersection / volume if volume > 0 else 0

def main():
    # === LOAD EVENTS ===
    event_paths = sorted(EVENT_FOLDER.glob("event_group_*.tif"))
    print(f"Found {len(event_paths)} grouped event files.")

    dendrite_id = 0
    raw_masks = []
    dendrite_volumes = []

    # === EXTRACT DENDRITES FROM EVENTS ===
    for path in tqdm(event_paths, desc="Extracting dendrites"):
        if dendrite_id >= MAX_DENDRITES_TOTAL:
            break

        # Load event stack
        stack = tifffile.imread(path).astype(np.float32)  # (T, Z, Y, X)
        if Y_CROP > 0:
            stack = stack[:, :, :-Y_CROP, :]

        # Threshold to identify active regions
        threshold = np.percentile(stack.flatten(), INTENSITY_PERCENTILE)
        binary_stack = stack > threshold
        activity_curve = binary_stack.max(axis=(1, 2, 3))
        active_ts = np.where(activity_curve > 0)[0]

        # Group active frames
        frame_groups = group_frames(active_ts, MAX_FRAME_GAP)

        for group in frame_groups:
            if len(group) < MIN_EVENT_LENGTH:
                continue

            # Create mask from active frames
            t_start, t_end = group[0], group[-1] + 1
            mask_3d = np.max(binary_stack[t_start:t_end], axis=0)

            # Clean mask with morphological operations
            cleaned_mask = np.zeros_like(mask_3d, dtype=np.uint8)
            for z in range(mask_3d.shape[0]):
                slice_mask = (mask_3d[z] * 255).astype(np.uint8)
                # Opening (erosion followed by dilation) to remove small noise
                cleaned = cv2.morphologyEx(slice_mask, cv2.MORPH_OPEN, 
                                          kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                # Closing (dilation followed by erosion) to fill small holes
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, 
                                          kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                cleaned_mask[z] = cleaned > 0

            # Remove objects smaller than minimum volume
            cleaned_mask = remove_small_objects(cleaned_mask.astype(bool), 
                                              int(MIN_VOL / VOXEL_VOL))

            if not np.any(cleaned_mask):
                continue

            # Label connected components and extract properties
            labeled = label(cleaned_mask)
            props = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)

            # Process each region
            for region in props:
                real_vol = region.area * VOXEL_VOL
                if not (MIN_VOL <= real_vol <= MAX_VOL):
                    continue

                mask = (labeled == region.label).astype(np.uint8)
                raw_masks.append(mask)
                dendrite_volumes.append(real_vol)
                dendrite_id += 1

                if dendrite_id >= MAX_DENDRITES_TOTAL:
                    break

            gc.collect()

    # === DEDUPLICATE MASKS ===
    print("Deduplicating masks...")
    unique_masks = []
    for i, mask in enumerate(raw_masks):
        is_duplicate = False
        for other in unique_masks:
            if dice_coeff(mask, other) > DUPLICATE_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_masks.append(mask)

    print(f"\n[✓] Retained {len(unique_masks)} unique dendrites out of {len(raw_masks)}")

    # === SAVE UNIQUE MASKS + MIP PREVIEWS ===
    print("Saving masks and previews...")
    for i, mask in enumerate(unique_masks):
        # Save mask as TIFF
        outpath = OUTPUT_LABELMAP_FOLDER / f"dend_{i:03d}_labelmap.tif"
        tifffile.imwrite(outpath, mask.astype(np.uint8) * (i + 1))

        # Create and save maximum intensity projection preview
        mip = np.max(mask, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(mip, cmap="cividis")
        ax.set_title(f"dend_{i:03d} MIP")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(PREVIEW_FOLDER / f"dend_{i:03d}_preview.png", dpi=150)
        plt.close(fig)

    # === CREATE VOLUME HISTOGRAM ===
    if dendrite_volumes:
        plt.figure(figsize=(6, 4))
        plt.hist(dendrite_volumes, bins=20, color='teal', edgecolor='black')
        plt.xlabel("Dendrite Volume (µm³)")
        plt.ylabel("Count")
        plt.title("Dendrite Volume Distribution")
        hist_path = OUTPUT_LABELMAP_FOLDER / "dendrite_volume_histogram.png"
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        print(f"Volume histogram saved to: {hist_path}")

    print("\n Module 2 complete: retained submasks of large dendrites, deduplicated, and saved.")

if __name__ == "__main__":
    main()