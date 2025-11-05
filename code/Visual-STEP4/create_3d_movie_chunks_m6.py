import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import gc

# === CONFIGURATION ===
DATE = "2025-10-29"
MOUSE = "rAi162_15"
RUN = "run1-crop"
Y_CROP = 3
FRAME_RATE = 10  # Hz
CHUNK_DURATION = 60  # seconds per chunk

# Cell selection options
USE_ALL_CELLS = True  # Set to False to use only SELECTED_CELLS
SELECTED_CELLS = [
    "dend_001","dend_003","dend_008", "dend_012", 
    "dend_014", "dend_015", "dend_016", "dend_019"
]


# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
RAW_CLEAN_PATH = BASE / "preprocessed" / "raw_clean.tif"
RAW_ORIG_PATH = BASE / "raw" / f"runA_run1_{MOUSE}_v1_reslice_crop.tif"
RAW_STACK_PATH = RAW_CLEAN_PATH if RAW_CLEAN_PATH.exists() else RAW_ORIG_PATH

def main():
    # === LOAD RAW DATA ===
    print(f"Loading {'clean' if RAW_STACK_PATH == RAW_CLEAN_PATH else 'original'} raw stack from: {RAW_STACK_PATH}")
    raw_stack = tifffile.imread(RAW_STACK_PATH).astype(np.float32)
    if Y_CROP > 0:
        raw_stack = raw_stack[:, :, :-Y_CROP, :]
    T, Z, Y, X = raw_stack.shape
    print(f"Stack shape: {raw_stack.shape}, Memory: {raw_stack.nbytes / 1e9:.1f} GB")

    # Calculate chunk parameters
    frames_per_chunk = CHUNK_DURATION * FRAME_RATE
    n_chunks = min(3, (T + frames_per_chunk - 1) // frames_per_chunk)  # Max 3 chunks
    print(f"Splitting into {n_chunks} chunks of {frames_per_chunk} frames each")

    # === CALCULATE F0 ONCE ===
    print("Calculating F0 baseline...")
    F0 = np.percentile(raw_stack, 20, axis=0)

    # === LOAD MASKS ===
    if USE_ALL_CELLS:
        mask_paths = sorted(MASK_FOLDER.glob("dend_*.tif"))
        print(f"Processing all {len(mask_paths)} cells")
    else:
        mask_paths = [MASK_FOLDER / f"{cell_name}_labelmap.tif" for cell_name in SELECTED_CELLS]
        print(f"Processing {len(SELECTED_CELLS)} selected cells")

    # Load all masks once
    masks = []
    for path in mask_paths:
        if path.exists():
            mask = tifffile.imread(path).astype(bool)
            # Adjust mask dimensions to match cropped stack
            if mask.shape != (Z, Y, X):
                print(f"Adjusting mask {path.name} from {mask.shape} to ({Z}, {Y}, {X})")
                if mask.shape[1] > Y:
                    mask = mask[:, :-Y_CROP, :]
                elif mask.shape[1] < Y:
                    pad_y = Y - mask.shape[1]
                    mask = np.pad(mask, ((0,0), (0,pad_y), (0,0)), mode='constant')
            if np.any(mask):
                masks.append(mask)

    print(f"Loaded {len(masks)} valid masks")

    # === PROCESS EACH CHUNK ===
    for chunk_idx in range(n_chunks):
        print(f"\n=== Processing chunk {chunk_idx + 1}/{n_chunks} ===")
        
        # Define time range for this chunk
        t_start = chunk_idx * frames_per_chunk
        t_end = min(t_start + frames_per_chunk, T)
        
        print(f"Time range: {t_start}-{t_end} ({t_end - t_start} frames)")
        
        # Extract chunk
        chunk = raw_stack[t_start:t_end]
        
        # Calculate ΔF/F for chunk
        dff_chunk = (chunk - F0[None]) / (F0[None] + 1e-6)
        
        # Create overlay for chunk
        overlay_chunk = np.zeros_like(dff_chunk)
        
        # Apply masks
        for mask in tqdm(masks, desc=f"Overlaying chunk {chunk_idx + 1}"):
            # Expand mask to 4D by adding time dimension
            mask_4d = mask[None, :, :, :]
            mask_4d = np.broadcast_to(mask_4d, dff_chunk.shape)
            overlay_chunk[mask_4d] = dff_chunk[mask_4d]
        
        # Save chunk
        chunk_name = f"chunk_{chunk_idx + 1:02d}_{t_start:04d}-{t_end:04d}"
        chunk_path = BASE / "overlays" / f"{chunk_name}_dff.tif"
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving chunk to: {chunk_path}")
        tifffile.imwrite(chunk_path, overlay_chunk.astype(np.float32))
        
        del chunk, dff_chunk, overlay_chunk
        gc.collect()

    del raw_stack
    gc.collect()
    print("All chunks processed and saved!")

    print("\nTo view chunks in Napari, load them individually:")
    for chunk_idx in range(n_chunks):
        t_start = chunk_idx * frames_per_chunk
        t_end = min(t_start + frames_per_chunk, T)
        chunk_name = f"chunk_{chunk_idx + 1:02d}_{t_start:04d}-{t_end:04d}"
        chunk_path = BASE / "overlays" / f"{chunk_name}_dff.tif"
        print(f"  Chunk {chunk_idx + 1}: {chunk_path}")

    print("\nUse view_napari.py to visualize individual chunks.")

if __name__ == "__main__":
    main()