import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
DATE = "2025-04-22"
MOUSE = "rAi162_15"
RUN = "run6"
Y_CROP = 3

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_reslice_bin.tif"
F0_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_F0_stack.tif"
LABELMAP_FOLDER = BASE / "labelmaps_curated_dynamic"
OUTPUT_PATH = BASE / "overlays" / "multi_dendrites_dff_branches.tif"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# === LOAD RAW DATA AND F₀ ===
print(f"Loading raw stack from: {RAW_STACK_PATH}")
raw_stack = tifffile.imread(RAW_STACK_PATH).astype(np.float32)
raw_stack = raw_stack[:, :, :-Y_CROP, :]

print(f"Loading F₀ from: {F0_PATH}")
F0 = tifffile.imread(F0_PATH).astype(np.float32)  # F₀ is already cropped

# === CALCULATE ΔF/F ===
print("Calculating ΔF/F stack...")
dff_stack = (raw_stack - F0[None]) / (F0[None] + 1e-6)
T, Z, Y, X = dff_stack.shape
del raw_stack, F0  # free memory

# === PREPARE OUTPUT OVERLAY ===
overlay_stack = np.zeros_like(dff_stack, dtype=np.float32)

# === LOAD MASKS AND OVERLAY ===
mask_paths = sorted(LABELMAP_FOLDER.glob("dend_*.tif"))
print(f"Found {len(mask_paths)} curated masks")

for path in tqdm(mask_paths, desc="Overlaying dendrites"):
    mask = tifffile.imread(path).astype(bool)
    if not np.any(mask):
        continue

    # Broadcast mask to time dimension
    mask_4d = np.broadcast_to(mask, (T, Z, Y, X))
    overlay_stack[mask_4d] = dff_stack[mask_4d]

# === SAVE 4D MOVIE ===
print(f"Saving overlay movie to: {OUTPUT_PATH}")
tifffile.imwrite(OUTPUT_PATH, overlay_stack.astype(np.float32))
print("✅ Done. You can now view this in Napari as a 4D movie.")
