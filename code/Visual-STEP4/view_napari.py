#!/usr/bin/env python
"""
Quick Napari viewer for 4D ΔF/F data with FOV box edges and scale bar (µm)
"""

import napari
import tifffile
import numpy as np
from pathlib import Path

# ---- Config ----
STACK_PATH = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data/2025-10-29/rAi162_15/run1-crop/overlays/chunk_01_0000-0600_dff.tif"

# (T, Z, Y, X): time left as frames; spatial voxels in µm
VOXEL_SCALE = (1.0, 4.7, 1.0, 1.2)  # T, Z, Y, X

print(f"Loading: {STACK_PATH}")
stack = tifffile.imread(STACK_PATH)
print(f"Stack shape: {stack.shape}")  # (T, Z, Y, X)
_, Z, Y, X = stack.shape

# ---- Napari viewer ----
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(
    stack,
    name="ΔF/F Branches",
    scale=VOXEL_SCALE,
    colormap="turbo",
    rendering="attenuated_mip",
)

# ---- 3D Field of View edges ----
z0, z1 = 0, Z - 1
y0, y1 = 0, Y - 1
x0, x1 = 0, X - 1

C = {
    "000": np.array([z0, y0, x0]),
    "001": np.array([z0, y0, x1]),
    "010": np.array([z0, y1, x0]),
    "011": np.array([z0, y1, x1]),
    "100": np.array([z1, y0, x0]),
    "101": np.array([z1, y0, x1]),
    "110": np.array([z1, y1, x0]),
    "111": np.array([z1, y1, x1]),
}

edges = [
    np.stack([C["000"], C["001"]]),  # bottom rectangle
    np.stack([C["001"], C["011"]]),
    np.stack([C["011"], C["010"]]),
    np.stack([C["010"], C["000"]]),
    np.stack([C["100"], C["101"]]),  # top rectangle
    np.stack([C["101"], C["111"]]),
    np.stack([C["111"], C["110"]]),
    np.stack([C["110"], C["100"]]),
    np.stack([C["000"], C["100"]]),  # verticals
    np.stack([C["001"], C["101"]]),
    np.stack([C["010"], C["110"]]),
    np.stack([C["011"], C["111"]]),
]


# ---- Scale bar ----
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "µm"
viewer.scale_bar.position = "bottom_right"
viewer.scale_bar.color = "white"
viewer.scale_bar.ticks = True  # show tick marks
viewer.scale_bar.font_size = 10


napari.run()
