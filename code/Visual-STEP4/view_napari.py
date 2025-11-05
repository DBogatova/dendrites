#!/usr/bin/env python
"""
Quick Napari viewer for 4D ΔF/F data
"""

import napari
import tifffile
import numpy as np
from pathlib import Path

# Path to your 4D stack
STACK_PATH = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data/2025-08-06/organoid/run4-crop/overlays/chunk_01_0000-0600_dff.tif"

# Voxel scaling (T, Z, Y, X) - adjust time scale as needed
VOXEL_SCALE = (1, 4.7, 1.0, 1.2)  # Time in frames, spatial in microns

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

# Add bounding box outline
t, z, y, x = stack.shape
box_coords = np.array([
    [0, 0, 0, 0], [0, 0, 0, x-1], [0, 0, y-1, 0], [0, 0, y-1, x-1],
    [0, z-1, 0, 0], [0, z-1, 0, x-1], [0, z-1, y-1, 0], [0, z-1, y-1, x-1]
])
viewer.add_points(
    box_coords,
    name="Field of View",
    size=0.5,
    face_color="white",
    opacity=0.8
)

napari.run()