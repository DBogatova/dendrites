#!/usr/bin/env python
"""
Quick Napari viewer for 4D ΔF/F data
"""

import napari
import tifffile
from pathlib import Path

# Path to your 4D stack
STACK_PATH = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data/2025-08-19/rAi162_15/run7/overlays/chunk_02_0600-1200_dff.tif"

# Voxel scaling (T, Z, Y, X) - adjust time scale as needed
VOXEL_SCALE = (1, 9.4, 1.0, 1.2)  # Time in frames, spatial in microns

# Load and view
print(f"Loading: {STACK_PATH}")
stack = tifffile.imread(STACK_PATH)
print(f"Stack shape: {stack.shape}")

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(
    stack,
    name="ΔF/F Branches",
    scale=VOXEL_SCALE,
    colormap="turbo",
    rendering="attenuated_mip"
)

napari.run()