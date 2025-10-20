#!/usr/bin/env python
"""
Module 4: Save Traces and Generate Visualizations (using pre-computed ΔF/F)

What it does:
1) Loads pre-computed ΔF/F stack from dff_stack_m0.py
2) Applies ROI masks to extract traces with background subtraction
3) Fixes negative motion artifacts, smooths, and saves
4) Makes per-ROI preview PDFs and a combined stack plot
5) Writes CSV and PKL
"""

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d, binary_erosion, binary_dilation
from skimage.morphology import ball
import pickle
import pandas as pd
import gc
import sys

try:
    import zarr
except ImportError:
    zarr = None

# ===== Matplotlib config =====
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False

# ===== CONFIG =====
DATE = "2025-08-06"
MOUSE = "organoid"
RUN = "run8"
FRAME_RATE = 10  # Hz
ARTIFACT_Z = -0.5  # replace ΔF/F < -0.5 with 0 (before smoothing)
SMOOTH_SIGMA = 0.5  # for gaussian_filter1d
CHUNK_T = 118  # time frames per chunk for memory efficiency

# ===== PATHS =====
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "data" / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runB_{RUN}_reslice.tif"
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
TRACE_FOLDER = BASE / "traces"; TRACE_FOLDER.mkdir(exist_ok=True)
TRACE_PKL = TRACE_FOLDER / "dff_traces_curated_bgsub.pkl"
TRACE_CSV = TRACE_FOLDER / "dff_traces_curated_bgsub.csv"
PREVIEW_FOLDER = BASE / "trace_previews_curated"; PREVIEW_FOLDER.mkdir(exist_ok=True)

PLOT_ALL_TRACES = False
SELECTED_NAMES = ["dend_003","dend_008","dend_010","dend_011","dend_029","dend_028", "dend_035", "dend_036", "dend_037", "dend_039"]


# ================== ΔF/F Stack Reader ==================
class DFFReader:
    def __init__(self, path):
        self.path = str(path)
        self.tf = tifffile.TiffFile(self.path)
        s = self.tf.series[0]
        self.shape = s.shape
        if len(self.shape) != 4:
            self._cleanup()
            raise RuntimeError(f"Expected 4D ΔF/F stack, got shape: {self.shape}")
        self.T, self.Z, self.Y, self.X = self.shape
        print(f"[DFFReader] ΔF/F stack shape: (T,Z,Y,X)=({self.T},{self.Z},{self.Y},{self.X})")

    def iter_chunks(self, chunk_size=CHUNK_T):
        """Yield (t0, t1, chunk) where chunk is (k,Z,Y,X) float32."""
        for t0 in range(0, self.T, chunk_size):
            t1 = min(self.T, t0 + chunk_size)
            try:
                # Try zarr approach first
                if zarr is not None:
                    store = self.tf.series[0].aszarr()
                    z = zarr.open(store, mode="r")
                    chunk = np.asarray(z[t0:t1]).astype(np.float32)
                else:
                    raise ImportError("zarr not available")
            except:
                # Fallback to direct array access
                full_arr = self.tf.series[0].asarray()
                chunk = full_arr[t0:t1].astype(np.float32)
            yield t0, t1, chunk
            del chunk
            gc.collect()

    def _cleanup(self):
        try: self.tf.close()
        except Exception: pass

    def close(self):
        self._cleanup()


# ================== Helpers ==================
def load_masks_and_indices(mask_folder, Z, Y, X):
    """Load curated masks, adjust to (Z, Y, X), build core/shell and flatten indices."""
    mask_paths = sorted(Path(mask_folder).glob("dend_*.tif"))
    rois = []  # list of dicts per ROI
    for path in mask_paths:
        name = path.stem.replace("_labelmap", "")
        m = tifffile.imread(path).astype(bool)

        # Adjust to (Z, Y, X) - ΔF/F stack already has Y cropping applied
        mz, my, mx = m.shape
        if mz != Z or mx != X:
            print(f"[WARN] {name}: expected Z={Z}, X={X}, got (Z={mz},Y={my},X={mx})")
        # Handle Y dimension mismatch
        if my < Y:
            pad_y = Y - my
            m = np.pad(m, ((0,0),(0,pad_y),(0,0)), mode='constant')
        elif my > Y:
            crop_y = my - Y
            m = m[:, :-crop_y, :]

        if not m.any():
            print(f"[SKIP] {name}: empty mask after adjustment")
            continue

        # Core & shell (3D)
        core = binary_erosion(m, structure=ball(1))
        if not core.any():
            core = m.copy()
        shell = binary_dilation(m, structure=ball(3)) & ~m

        core_idx = np.flatnonzero(core.ravel())
        shell_idx = np.flatnonzero(shell.ravel()) if shell.any() else np.array([], dtype=np.int64)

        rois.append({
            "name": name,
            "mask": m,  # keep for preview
            "core_idx": core_idx,
            "shell_idx": shell_idx,
        })
        del m, core, shell
    return rois


def main():
    # ===== Use raw stack directly =====
    print(f"Using raw stack: {RAW_STACK_PATH}")
    if not RAW_STACK_PATH.exists():
        print(f"ERROR: Raw stack not found: {RAW_STACK_PATH}")
        return
    stack_reader = DFFReader(RAW_STACK_PATH)
    T, Z, Y, X = stack_reader.T, stack_reader.Z, stack_reader.Y, stack_reader.X
    print(f"Raw stack shape: T={T}, Z={Z}, Y={Y}, X={X}")
    
    # Check expected vs actual duration
    expected_frames = 180 * FRAME_RATE  # 180 seconds * 10 Hz = 1800 frames
    actual_duration = T / FRAME_RATE
    print(f"Expected frames: {expected_frames} ({180}s at {FRAME_RATE}Hz)")
    print(f"Actual frames: {T} ({actual_duration:.1f}s at {FRAME_RATE}Hz)")
    
    if T < expected_frames:
        print(f"WARNING: Raw stack is shorter than expected!")
        print(f"This suggests the original raw data is incomplete.")

    # ===== Load curated masks and indices =====
    print(f"Loading curated masks from: {MASK_FOLDER}")
    rois = load_masks_and_indices(MASK_FOLDER, Z, Y, X)
    if not rois:
        stack_reader.close()
        print("No masks found. Exiting.")
        return

    # ===== Compute F0 baseline =====
    print("Computing F0 baseline (20th percentile)...")
    f0_data = []
    for t0, t1, chunk in stack_reader.iter_chunks():
        f0_data.append(chunk)
        if len(f0_data) * CHUNK_T > 500:  # Use first ~500 frames for F0
            break
    f0_stack = np.concatenate(f0_data, axis=0)
    f0_vol = np.percentile(f0_stack, 20, axis=0)  # (Z,Y,X)
    del f0_data, f0_stack
    gc.collect()
    
    # ===== Extract traces from raw stack =====
    print("Computing ΔF/F and extracting traces...")
    
    for roi in rois:
        roi["trace"] = np.empty(T, dtype=np.float32)

    for t0, t1, chunk in stack_reader.iter_chunks():
        for t_rel in range(t1 - t0):
            t_abs = t0 + t_rel
            vol = chunk[t_rel]  # (Z, Y, X)
            
            # Convert to ΔF/F: (F - F0) / F0
            vol = (vol - f0_vol) / (f0_vol + 1e-6)
            
            vol_flat = vol.reshape(-1)
            
            for roi in rois:
                # Extract core and shell values
                core_val = vol_flat[roi["core_idx"]].mean()
                
                if roi["shell_idx"].size > 0:
                    shell_val = vol_flat[roi["shell_idx"]].mean()
                else:
                    shell_val = 0.0
                
                # Background subtraction
                roi["trace"][t_abs] = core_val - shell_val
            
            del vol_flat
        
        print(f"Processed frames {t0}..{t1-1}")
        del chunk
        gc.collect()

    stack_reader.close()
    print(f"Extracted traces for {len(rois)} ROIs from raw stack")

    # ===== Artifact fix + smoothing + to % =====
    print("Fixing artifacts and smoothing…")
    labels, traces_pct = [], []
    for roi in rois:
        dff = roi["trace"]
        artifact_mask = dff < ARTIFACT_Z
        if artifact_mask.any():
            print(f"  {roi['name']}: {artifact_mask.sum()} points < {ARTIFACT_Z}, setting to 0")
            dff = dff.copy()
            dff[artifact_mask] = 0.0

        smoothed = gaussian_filter1d(dff, sigma=SMOOTH_SIGMA) * 100.0
        roi["trace_pct"] = smoothed
        labels.append(roi["name"])
        traces_pct.append(smoothed.astype(np.float32))

    # ===== Per-ROI previews (MIP + trace) =====
    print("Saving previews…")
    t_axis = np.arange(T) / float(FRAME_RATE)
    print(f"Time axis: 0 to {t_axis[-1]:.1f} seconds ({len(t_axis)} frames)")
    for roi in rois:
        name = roi["name"]
        mip = roi["mask"].max(axis=0).astype(bool)

        fig, (ax_img, ax_trace) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.2, 2]})

        # Left: colored MIP
        rgb = np.zeros((*mip.shape, 3), dtype=np.float32)
        color = (0.2, 0.7, 0.2)
        for c in range(3): rgb[..., c][mip] = color[c]
        ax_img.imshow(rgb)
        try:
            cell_number = name.split('_')[1]
            title_left = f"Cell {cell_number} — MIP"
        except Exception:
            title_left = f"{name} — MIP"
        ax_img.set_title(title_left)
        ax_img.axis("off")

        # Right: ΔF/F%
        ax_trace.plot(t_axis, roi["trace_pct"], color='teal', lw=1.5)
        ax_trace.set_title("ΔF/F Trace (%)")
        ax_trace.set_xlabel("Time (s)")
        ax_trace.set_ylabel("ΔF/F (%)")
        ax_trace.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(PREVIEW_FOLDER / f"{name}_trace_mip.pdf", format="pdf")
        plt.close(fig)

    # ===== Save PKL & CSV =====
    print("Saving traces…")
    with open(TRACE_PKL, "wb") as f:
        pickle.dump(list(zip(labels, traces_pct)), f)
    pd.DataFrame(np.vstack(traces_pct).T, columns=labels).to_csv(TRACE_CSV, index_label="Frame")
    print(f"✅ Traces saved:\n  {TRACE_PKL}\n  {TRACE_CSV}")

    # ===== Combined stacked plot =====
    print("Generating combo plot…")
    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 6.0
    count = 0
    for i, (name, trace) in enumerate(zip(labels, traces_pct)):
        if not PLOT_ALL_TRACES and name not in SELECTED_NAMES:
            continue
        ax.plot(t_axis, trace + count * offset, color=cm.turbo(i / max(1, len(traces_pct))), lw=1)
        # label at right
        try:
            cell_number = name.split('_')[1]
            cell_label = f"Cell {cell_number}"
        except Exception:
            cell_label = name
        ax.text(t_axis[-1] + 1, count * offset, cell_label, va='center', fontsize=8)
        count += 1

    # scale bar (10%)
    if count > 0:
        scale_bar_x = t_axis[-1] - 7
        scale_bar_y = offset * (count - 0.5)
        scale_bar_h = 10  # 10% ΔF/F
        ax.plot([scale_bar_x, scale_bar_x], [scale_bar_y, scale_bar_y + scale_bar_h], color='k', lw=2)
        ax.text(scale_bar_x + 1, scale_bar_y + scale_bar_h / 10, "10%", va='center', ha='left', fontsize=10)

    ax.set_xlim(0, t_axis[-1])
    ax.set_ylim(-offset, max(1, count) * offset + 5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (%) + offset")
    ax.set_yticks([])
    ax.set_title("Stacked ΔF/F Traces")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PREVIEW_FOLDER / "combo_traces_curated_bgsub.pdf", format="pdf")
    plt.close(fig)

    print(f"✅ Combo plot saved to: {PREVIEW_FOLDER / 'combo_traces_curated_bgsub.pdf'}")
    print("Module 4 processing complete!")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("\n[ERROR]", e, file=sys.stderr)
        sys.exit(1)