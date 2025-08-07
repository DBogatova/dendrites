import numpy as np
import tifffile
from pathlib import Path
import napari
from tqdm import tqdm
import imageio
import time

# === CONFIGURATION ===
DATE = "2025-04-22"
MOUSE = "rAi162_15"
RUN = "run6"
Y_CROP = 3
VOXEL_SCALE = (4.7, 0.5, 0.6)  # (Z, Y, X) in microns

# Specify which cells to include
SELECTED_CELLS = [
    "dend_006", "dend_014", "dend_019", "dend_035",
    "dend_023", "dend_025", "dend_027"
]

# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_reslice_bin.tif"
OUTPUT_PATH =  BASE / "overlays" / "selected_cells_dff.tif"
VIDEO_PATH = BASE / "overlays" / "selected_cells_rotation.mp4"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# === LOAD RAW DATA ===
print(f"Loading raw stack from: {RAW_STACK_PATH}")
raw_stack = tifffile.imread(RAW_STACK_PATH).astype(np.float32)
if Y_CROP > 0:
    raw_stack = raw_stack[:, :, :-Y_CROP, :]
T, Z, Y, X = raw_stack.shape

# === CALCULATE F0 AND ΔF/F ===
print("Calculating F0 (lowest 20% of values)...")
sorted_stack = np.sort(raw_stack, axis=0)
lowest_20pct = sorted_stack[:int(T * 0.2)]
F0 = np.mean(lowest_20pct, axis=0)

print("Calculating ΔF/F stack...")
dff_stack = (raw_stack - F0[None]) / (F0[None] + 1e-6)
del raw_stack, sorted_stack, lowest_20pct  # free memory

# === PREPARE OUTPUT OVERLAY ===
overlay_stack = np.zeros_like(dff_stack, dtype=np.float32)

# === LOAD SELECTED MASKS AND OVERLAY ===
print(f"Processing {len(SELECTED_CELLS)} selected cells")
mask_paths = [MASK_FOLDER / f"{cell_name}_labelmap.tif" for cell_name in SELECTED_CELLS]

for path in tqdm(mask_paths, desc="Overlaying cells"):
    if not path.exists():
        print(f"  Warning: Mask not found at {path}")
        continue
        
    mask = tifffile.imread(path).astype(bool)
    
    # Ensure mask dimensions match raw stack
    if mask.shape != (Z, Y, X):
        print(f"Adjusting mask {path.name} from {mask.shape} to {(Z, Y, X)}")
        # If mask is smaller in Y, pad it; if larger, crop it
        if mask.shape[1] < Y:
            pad_y = Y - mask.shape[1]
            mask = np.pad(mask, ((0,0), (0,pad_y), (0,0)), mode='constant')
        elif mask.shape[1] > Y:
            crop_y = mask.shape[1] - Y
            mask = mask[:, :-crop_y, :]
    
    if not np.any(mask):
        print(f"  Warning: Empty mask in {path.name}")
        continue

    # Broadcast mask to time dimension
    mask_4d = np.broadcast_to(mask, (T, Z, Y, X))
    overlay_stack[mask_4d] = dff_stack[mask_4d]

# === SAVE 4D MOVIE ===
print(f"Saving overlay movie to: {OUTPUT_PATH}")
tifffile.imwrite(OUTPUT_PATH, overlay_stack.astype(np.float32))

# === LAUNCH NAPARI VIEWER ===
print("Launching Napari viewer...")
viewer = napari.Viewer(ndisplay=3)

# Add the image layer
image_layer = viewer.add_image(
    overlay_stack,
    name="Selected Cells ΔF/F",
    scale=(1, *VOXEL_SCALE),  # (T, Z, Y, X)
    colormap="turbo",
    rendering="attenuated_mip",
)

# Add more translucent outline cube
outline = np.zeros((Z, Y, X), dtype=bool)
outline[[0, -1], :, :] = True
outline[:, [0, -1], :] = True
outline[:, :, [0, -1]] = True

cube_layer = viewer.add_labels(outline, name="Grid Cube", scale=VOXEL_SCALE, opacity=0.05)

# === CREATE ROTATING VIDEO ===
print(f"Creating rotating video at {VIDEO_PATH}...")

# Set up the camera for 3D view
viewer.dims.ndisplay = 3
viewer.camera.zoom = 0.5  # Adjust as needed

# Create a video writer
fps = 5
writer = imageio.get_writer(VIDEO_PATH, fps=fps)

# Set the time point to use (middle of the stack)
time_point = min(T // 2, T - 1)
viewer.dims.set_point(0, time_point)

# Capture frames while rotating
n_frames = 180  # 6 seconds at 30 fps
for i in tqdm(range(n_frames), desc="Capturing frames"):
    # Rotate the camera around the vertical axis
    angle = i * 2  # 2 degrees per frame = 360 degrees in 180 frames
    viewer.camera.angles = (0, angle, 90)  # (elevation, azimuth, roll)
    
    # Small delay to ensure the view is updated
    time.sleep(0.01)
    
    # Capture the frame
    frame = viewer.screenshot(canvas_only=True)
    writer.append_data(frame)

writer.close()
print(f"✅ Video saved to: {VIDEO_PATH}")

# Keep the viewer open
napari.run()