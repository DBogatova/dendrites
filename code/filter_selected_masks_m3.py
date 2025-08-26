#!/usr/bin/env python
"""
Module 2c: Filter and Edit Selected Masks

This interactive script allows manual curation of automatically generated masks:
1. Loads all masks from the labelmaps folder
2. Provides a Napari interface for viewing and editing masks
3. Allows navigation through masks with arrow keys
4. Supports operations like delete, keep, merge, and subtract
5. Saves curated masks to the output folder

Key bindings:
- Left/Right: Navigate between masks
- d: Delete current mask
- k: Keep current mask and move to next
- m: Merge drawing with current mask
- x: Subtract drawing from current mask
- Ctrl+R: Reset drawing layer
- Ctrl+S: Save all kept masks
"""

import napari
import tifffile
import numpy as np
from pathlib import Path
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist

# === CONFIGURATION ===
DATE = "2025-08-19"
MOUSE = "rAi162_15"
RUN = "run7"

# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
LABELMAP_FOLDER = BASE / "labelmaps"
OUTPUT_FOLDER = BASE / "labelmaps_curated_dynamic"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def show_mask(i, highlight=False):
    """
    Add a mask to the viewer.
    
    Args:
        i: Index of the mask to show
        highlight: Whether to highlight this mask (higher opacity)
    """
    name = names[i]
    if name in viewer.layers:
        return
    mask = edited.get(i, masks[i])
    viewer.add_labels(mask, name=name, opacity=1.0 if highlight else 0.5)
    visible.add(i)

def clear_visible():
    """Remove all currently visible masks from the viewer."""
    for i in visible:
        name = names[i]
        if name in viewer.layers:
            viewer.layers.remove(name)
    visible.clear()

def refresh(center_idx, radius=3):
    """
    Refresh the viewer to show the selected mask and its neighbors.
    
    Args:
        center_idx: Index of the mask to center on
        radius: Number of neighboring masks to show
    """
    clear_visible()
    show_mask(center_idx, highlight=True)
    dists = cdist([centroids[center_idx]], centroids)[0]
    nearby = np.argsort(dists)[1:radius+1]
    for i in nearby:
        show_mask(i, highlight=False)
    print(f"🔍 Focus: {names[center_idx]} with {len(nearby)} neighbors")

def save_all():
    """Save all kept masks to the output folder."""
    print("💾 Saving...")
    count = 0
    for i, name in enumerate(names):
        if i in deleted:
            continue
        mask = edited.get(i, masks[i])
        path = OUTPUT_FOLDER / f"dend_{count:03d}_labelmap.tif"
        tifffile.imwrite(path, mask * (count + 1))
        count += 1
    print(f"✅ Saved {count} masks.")

def main():
    global viewer, masks, names, centroids, selected, visible, deleted, edited
    
    # === LOAD ALL MASKS ===
    print("Loading masks...")
    mask_paths = sorted(LABELMAP_FOLDER.glob("dend_*.tif"))
    masks = [tifffile.imread(p).astype(np.uint8) for p in mask_paths]
    names = [p.stem for p in mask_paths]
    
    # Calculate centroids for each mask
    centroids = []
    for m in masks:
        props = regionprops(label(m))
        if props:
            centroids.append(props[0].centroid)
        else:
            # Fallback for empty masks
            centroids.append((0, 0, 0))
            
    print(f"Loaded {len(masks)} masks.")

    # === STATE ===
    selected = [0]  # Using list for mutable reference in callbacks
    visible = set()  # Currently visible masks
    deleted = set()  # Masks marked for deletion
    edited = {}      # Edited versions of masks

    # === INITIALIZE VIEWER ===
    global viewer
    viewer = napari.Viewer(ndisplay=3)
    
    # Show first mask
    refresh(selected[0])
    
    # Add drawing layer
    viewer.add_labels(np.zeros_like(masks[0]), name="draw", opacity=0.6)

    # === NAVIGATION ===
    @viewer.bind_key("Right")
    def next_mask(viewer):
        if selected[0] < len(masks) - 1:
            selected[0] += 1
            refresh(selected[0])

    @viewer.bind_key("Left")
    def prev_mask(viewer):
        if selected[0] > 0:
            selected[0] -= 1
            refresh(selected[0])

    # === DELETE ===
    @viewer.bind_key("d")
    def delete_mask(viewer):
        deleted.add(selected[0])
        print(f"❌ Deleted: {names[selected[0]]}")
        next_mask(viewer)

    # === KEEP ===
    @viewer.bind_key("k")
    def keep_mask(viewer):
        print(f"✅ Kept: {names[selected[0]]}")
        next_mask(viewer)

    # === MERGE ===
    @viewer.bind_key("m")
    def merge_draw(viewer):
        i = selected[0]
        base = edited.get(i, masks[i]) > 0
        draw = viewer.layers["draw"].data > 0
        result = np.logical_or(base, draw).astype(np.uint8)
        edited[i] = result
        print(f"🟣 Merged into: {names[i]}")
        refresh(i)

    # === SUBTRACT ===
    @viewer.bind_key("x")
    def subtract_draw(viewer):
        i = selected[0]
        base = edited.get(i, masks[i]) > 0
        draw = viewer.layers["draw"].data > 0
        result = np.logical_and(base, ~draw).astype(np.uint8)
        edited[i] = result
        print(f"✂️ Subtracted from: {names[i]}")
        refresh(i)

    # === RESET DRAW LAYER ===
    @viewer.bind_key("Control-R")
    def reset_draw(viewer):
        if "draw" in viewer.layers:
            viewer.layers.remove("draw")
        viewer.add_labels(np.zeros_like(masks[0]), name="draw", opacity=0.6)
        print("🎨 Reset draw layer.")

    # === SAVE ===
    @viewer.bind_key("Control-S")
    def save_all_masks(viewer):
        save_all()

    # Print instructions
    print("\n=== INSTRUCTIONS ===")
    print("Left/Right: Navigate between masks")
    print("d: Delete current mask")
    print("k: Keep current mask and move to next")
    print("m: Merge drawing with current mask")
    print("x: Subtract drawing from current mask")
    print("Ctrl+R: Reset drawing layer")
    print("Ctrl+S: Save all kept masks")
    
    # Start the application
    napari.run()

if __name__ == "__main__":
    main()