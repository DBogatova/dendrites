#!/usr/bin/env python
"""
Module 5: Save Traces and Generate Visualizations (memory-safe)

What it does (streaming, low RAM):
1) Reads raw frames in a streaming way (no big array in memory)
2) For each curated mask, computes per-voxel F0 = mean(first 20% frames)
3) Streams all frames again to compute mean((F/F0)-1) over each ROI (core) and shell
4) Subtracts shell (background), fixes negative motion artifacts, smooths, and saves
5) Makes per-ROI preview PDFs and a combined stack plot
6) Writes CSV and PKL

Works with:
- Multipage OME-TIFF (T*Z pages)   → native
- SubIFD hyperstacks               → via tifffile.aszarr + zarr>=3 (if installed)
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

# ===== Matplotlib config =====
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False

# ===== CONFIG =====
DATE = "2025-08-18"
MOUSE = "rAi162_15"
RUN = "run9"
Y_CROP = 3
FRAME_RATE = 10  # Hz
ARTIFACT_Z = -0.5  # replace ΔF/F < -0.5 with 0 (before smoothing)
SMOOTH_SIGMA = 0.5  # for gaussian_filter1d
BASELINE_FRAC = 0.20  # first 20% of frames for F0

# ===== PATHS =====
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "data" / DATE / MOUSE / RUN
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green_reslice.tif"
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
TRACE_FOLDER = BASE / "traces"; TRACE_FOLDER.mkdir(exist_ok=True)
TRACE_PKL = TRACE_FOLDER / "dff_traces_curated_bgsub.pkl"
TRACE_CSV = TRACE_FOLDER / "dff_traces_curated_bgsub.csv"
PREVIEW_FOLDER = BASE / "trace_previews_curated"; PREVIEW_FOLDER.mkdir(exist_ok=True)

PLOT_ALL_TRACES = False
SELECTED_NAMES = ["dend_004","dend_005","dend_006","dend_011","dend_014","dend_015","dend_019", "dend_020", "dend_021", "dend_022", "dend_024", "dend_025", "dend_026", "dend_037"]


# ================== Robust frame reader (handles OME pages or SubIFDs via aszarr) ==================
class FrameReader:
    def __init__(self, path, y_crop=0):
        self.path = str(path)
        self.y_crop = int(y_crop) if y_crop else 0
        self.tf = tifffile.TiffFile(self.path)
        s = self.tf.series[0]
        self.axes = s.axes
        if len(s.shape) != 4 or not all(ax in self.axes for ax in "TZYX"):
            self._cleanup()
            raise RuntimeError(f"Unsupported TIFF: shape={s.shape}, axes='{s.axes}' (need 4D with T/Z/Y/X)")
        self.T = s.shape[self.axes.index('T')]
        self.Z = s.shape[self.axes.index('Z')]
        self.Y = s.shape[self.axes.index('Y')]
        self.X = s.shape[self.axes.index('X')]
        self.Yeff = self.Y - (self.y_crop if self.y_crop else 0)

        # Try multipage (T*Z pages) first
        self.pages = s.pages
        if len(self.pages) == self.T * self.Z:
            self.backend = "pages"
        else:
            # Fall back to aszarr (requires zarr>=3)
            try:
                import zarr  # noqa
                za = self.tf.aszarr()
                self.z = zarr.open(za, mode='r')
                if self.z.shape != (self.T, self.Z, self.Y, self.X):
                    self._cleanup()
                    raise RuntimeError(f"aszarr shape mismatch: got {self.z.shape}, expected {(self.T, self.Z, self.Y, self.X)}")
                self.backend = "zarr"
            except Exception as e:
                self._cleanup()
                raise RuntimeError(
                    "This TIFF uses SubIFDs (single top-level page). "
                    "To stream it without huge RAM, please install zarr>=3:\n"
                    "    pip install 'zarr>=3'\n"
                    "or convert once to a multipage OME-TIFF (e.g., with bfconvert)."
                ) from e

        print(f"[Reader] backend={self.backend}  shape=(T,Z,Y,X)=({self.T},{self.Z},{self.Y},{self.X})  Yeff={self.Yeff}")

    def iter_frames(self, t0=0, t1=None):
        """Yield (t, vol) where vol is (Z, Yeff, X), float32."""
        if t1 is None: t1 = self.T
        if self.backend == "pages":
            for t in range(t0, t1):
                start = t * self.Z
                vol = np.empty((self.Z, self.Y, self.X), dtype=np.float32)
                for z in range(self.Z):
                    vol[z] = self.pages[start + z].asarray().astype(np.float32)
                if self.y_crop: vol = vol[:, :self.Yeff, :]
                yield t, vol
                del vol
                if (t % 32) == 0: gc.collect()
        else:  # zarr
            for t in range(t0, t1):
                vol = self.z[t, :, :, :].astype(np.float32)
                if self.y_crop: vol = vol[:, :self.Yeff, :]
                yield t, vol
                del vol
                if (t % 32) == 0: gc.collect()

    def _cleanup(self):
        try: self.tf.close()
        except Exception: pass

    def close(self):
        self._cleanup()


# ================== Helpers ==================
def load_masks_and_indices(mask_folder, Z, Yeff, X):
    """Load curated masks, adjust to (Z, Yeff, X), build core/shell and flatten indices."""
    mask_paths = sorted(Path(mask_folder).glob("dend_*.tif"))
    rois = []  # list of dicts per ROI
    for path in mask_paths:
        name = path.stem.replace("_labelmap", "")
        m = tifffile.imread(path).astype(bool)

        # Adjust to (Z, Yeff, X)
        mz, my, mx = m.shape
        if mz != Z or mx != X:
            print(f"[WARN] {name}: expected Z={Z}, X={X}, got (Z={mz},Y={my},X={mx})")
        # Handle Y cropping/padding like the original
        if my < Yeff:
            pad_y = Yeff - my
            m = np.pad(m, ((0,0),(0,pad_y),(0,0)), mode='constant')
        elif my > Yeff:
            crop_y = my - Yeff
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
    # ===== Open reader and get shape =====
    fr = FrameReader(RAW_STACK_PATH, y_crop=Y_CROP)
    T, Z, Yeff, X = fr.T, fr.Z, fr.Yeff, fr.X
    print(f"Stack shape: T={T}, Z={Z}, Y={Yeff}, X={X}")

    # ===== Load curated masks and indices =====
    print(f"Loading curated masks from: {MASK_FOLDER}")
    rois = load_masks_and_indices(MASK_FOLDER, Z, Yeff, X)
    if not rois:
        fr.close()
        print("No masks found. Exiting.")
        return

    # ===== Compute per-voxel F0 (mean over first 20% of frames) for each ROI region =====
    n_baseline = max(1, int(T * BASELINE_FRAC))
    print(f"Computing F0 from first {n_baseline} frames…")

    # Per-ROI F0 arrays (per-voxel, not averaged) so we can compute mean((F/F0)-1) exactly
    for roi in rois:
        roi["F0_core_sum"] = np.zeros(roi["core_idx"].shape[0], dtype=np.float32)
        if roi["shell_idx"].size:
            roi["F0_shell_sum"] = np.zeros(roi["shell_idx"].shape[0], dtype=np.float32)

    vox_count = Z * Yeff * X
    for t, vol in fr.iter_frames(0, n_baseline):
        vflat = vol.reshape(-1)
        for roi in rois:
            roi["F0_core_sum"] += vflat[roi["core_idx"]]
            if roi["shell_idx"].size:
                roi["F0_shell_sum"] += vflat[roi["shell_idx"]]
        if (t % 64) == 0: gc.collect()
        del vflat, vol

    for roi in rois:
        roi["F0_core"] = roi["F0_core_sum"] / float(n_baseline)
        del roi["F0_core_sum"]
        roi["invF0_core"] = 1.0 / (roi["F0_core"] + 1e-6)

        if roi["shell_idx"].size:
            roi["F0_shell"] = roi["F0_shell_sum"] / float(n_baseline)
            del roi["F0_shell_sum"]
            roi["invF0_shell"] = 1.0 / (roi["F0_shell"] + 1e-6)
        else:
            roi["invF0_shell"] = None

    gc.collect()
    print("F0 computed for all ROIs.")

    # ===== Stream all frames to compute ΔF/F traces per ROI =====
    print("Streaming frames to compute traces…")
    for roi in rois:
        roi["trace"] = np.empty(T, dtype=np.float32)

    for t, vol in fr.iter_frames(0, T):
        vflat = vol.reshape(-1)
        for roi in rois:
            # mean over voxels of (F/F0 - 1) = mean(F * invF0) - 1
            core_term = vflat[roi["core_idx"]] * roi["invF0_core"]
            core_dff = core_term.mean() - 1.0

            if roi["invF0_shell"] is not None and roi["shell_idx"].size > 0:
                shell_term = vflat[roi["shell_idx"]] * roi["invF0_shell"]
                shell_dff = shell_term.mean() - 1.0
            else:
                shell_dff = 0.0

            roi["trace"][t] = core_dff - shell_dff
        if (t % 64) == 0: gc.collect()
        del vflat, vol

    fr.close()

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

    # scale bar (2%)
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
    print("Module 5 processing complete!")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("\n[ERROR]", e, file=sys.stderr)
        sys.exit(1)
