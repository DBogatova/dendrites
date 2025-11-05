#!/usr/bin/env python
"""
Module Mask Outlines: Curated Mask Overlays (XY / XZ / YZ)

Generates overlays of curated masks on background MIPs from the raw stack.
- Supports **combined** overlays for a subset of masks (recommended)
- Optional **per-mask** overlays
- Matches Module 1/4 Y-crop and Module 2 voxel sizes

Examples
--------
# Single combined overlay of selected masks
python module_overlays_triplanar.py --select dend_004,dend_013,dend_016 \
  --combined --method p99

# Combined + per-mask outputs
python module_overlays_triplanar.py --select dend_004,dend_013 --combined --per-mask
"""

from pathlib import Path
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import exposure
from skimage.measure import find_contours

# Set Arial font
mpl.rcParams['font.family'] = 'Arial'

# ===================== CONFIG =====================
DATE  = "2025-08-06"
MOUSE = "organoid"
RUN   = "run4-crop"

VOXEL_SIZE = (4.7, 1.0, 1.2)   # (dz, dy, dx) µm
PROJ_METHOD = "max"            # default; can override with --method
CLAHE_ON = True                # contrast boost on background
MARGIN = (2, 12, 12)           # (z,y,x) padding (voxels) around union bbox
Y_CROP = 3                     # crop Y to match Modules 1 & 4
MASK_OFFSET = (0, 10, 0)     # (Z, Y, X) offset correction in voxels

# ===================== PATHS =====================
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_STACK = BASE / "preprocessed" / f"raw_clean.tif"          # (T,Z,Y,X) or (Z,Y,X)
CURATED_DIR = BASE / "labelmaps_curated_dynamic"  # masks: dend_XXX_labelmap.tif (ZYX)
OUT_DIR = BASE / "overlays_curated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== MASK SELECTION (default list) =====================
SELECTED_MASKS = [
    "dend_001","dend_003","dend_008", "dend_012", "dend_014", "dend_015", "dend_016", "dend_019"
]

# ===================== ARGS =====================
parser = argparse.ArgumentParser(description="Overlay curated masks (XY/XZ/YZ)")
parser.add_argument("--select", type=str, default="",
                    help="Comma-separated mask names (e.g. 'dend_004,dend_013')")
parser.add_argument("--combined", action="store_true",
                    help="Write a single triplanar figure with ALL selected masks together")
parser.add_argument("--per-mask", action="store_true",
                    help="Additionally write per-mask triplanar figures")
parser.add_argument("--method", type=str, default=PROJ_METHOD, choices=["max","mean","p99"],
                    help="Projection method for background MIPs")
args = parser.parse_args()
if args.select:
    SELECTED_MASKS = [s.strip() for s in args.select.split(',') if s.strip()]
PROJ_METHOD = args.method

# ===================== HELPERS =====================
def _load_raw(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 4:
        arr = arr.mean(axis=0)   # average over time for stable background
    assert arr.ndim == 3, f"Expected 3D or 4D raw; got {arr.shape}"
    if Y_CROP > 0:
        arr = arr[:, :-Y_CROP, :]  # apply same Y crop as Modules 1 & 4
    return arr


def _project(stack: np.ndarray, axis: int = 0, method: str = "max") -> np.ndarray:
    if method == "max":
        return stack.max(axis=axis)
    if method == "mean":
        return stack.mean(axis=axis)
    if method == "p99":
        return np.percentile(stack, 99, axis=axis)
    raise ValueError(f"Unknown projection method: {method}")


def _core_stem(p: Path) -> str:
    return p.stem.replace("_labelmap", "")


# ===================== MAIN =====================
def main():
    print("Loading raw stack...")
    raw = _load_raw(RAW_STACK)

    print(f"Loading curated masks from {CURATED_DIR}")
    mask_paths_all = sorted(CURATED_DIR.glob("dend_*.tif"))
    if SELECTED_MASKS:
        sel_set = {s.strip() for s in SELECTED_MASKS}
        mask_paths = [p for p in mask_paths_all if _core_stem(p) in sel_set]
    else:
        mask_paths = mask_paths_all

    if not mask_paths:
        all_found_core = [_core_stem(p) for p in mask_paths_all]
        print("⚠️ No matching masks found for selection:", SELECTED_MASKS)
        print("   Available core names:", all_found_core)
        return

    print(f"Processing {len(mask_paths)} selected masks: {[ _core_stem(p) for p in mask_paths ]}")

    # Load masks and apply offset correction
    masks = []
    for p in mask_paths:
        mask = tifffile.imread(p).astype(bool)
        # Apply offset correction
        oz, oy, ox = MASK_OFFSET
        if oz != 0 or oy != 0 or ox != 0:
            mask_corrected = np.zeros_like(mask)
            # Calculate valid ranges for copying
            z_src = slice(max(0, -oz), mask.shape[0] - max(0, oz))
            y_src = slice(max(0, -oy), mask.shape[1] - max(0, oy))
            x_src = slice(max(0, -ox), mask.shape[2] - max(0, ox))
            z_dst = slice(max(0, oz), mask.shape[0] - max(0, -oz))
            y_dst = slice(max(0, oy), mask.shape[1] - max(0, -oy))
            x_dst = slice(max(0, ox), mask.shape[2] - max(0, -ox))
            mask_corrected[z_dst, y_dst, x_dst] = mask[z_src, y_src, x_src]
            mask = mask_corrected
        masks.append(mask)

    # Use full Z-range for background
    raw_sub = raw

    # Background projections
    def bg2d(ax):
        return _project(raw_sub, axis=ax, method=PROJ_METHOD)

    bg_xy = bg2d(0)
    bg_xz = bg2d(1)
    bg_yz = bg2d(2)

    # Contrast enhancement
    def enhance(im):
        im = exposure.rescale_intensity(im, in_range="image", out_range=(0, 1))
        return exposure.equalize_adapthist(im, clip_limit=0.01) if CLAHE_ON else im

    # Flip background images to match mask coordinate system
    bg_xy = np.flipud(enhance(bg_xy))
    bg_xz = np.flipud(enhance(bg_xz))
    bg_yz = np.flipud(enhance(bg_yz))

    # Projections per mask (binary) - flip to match background
    projs_xy = [np.flipud(m.any(axis=0)) for m in masks]  # (Y,X)
    projs_xz = [np.flipud(m.any(axis=1)) for m in masks]  # (Z,X)
    projs_yz = [np.flipud(m.any(axis=2)) for m in masks]  # (Z,Y)

    # Physical extents in µm (correct aspect)
    dz, dy, dx = VOXEL_SIZE
    ext_xy = [0, raw.shape[2] * dx, 0, raw.shape[1] * dy]
    ext_xz = [0, raw.shape[2] * dx, 0, raw_sub.shape[0] * dz]
    ext_yz = [0, raw.shape[1] * dy, 0, raw_sub.shape[0] * dz]

    # Helper to draw colored contours of multiple masks (match combo plot)
    cmap = plt.get_cmap("turbo")

    def draw(ax, bg, projs, extent, title):
        ax.imshow(bg, cmap="gray", origin="lower", extent=extent, interpolation="nearest")
        for k, pr in enumerate(projs):
            if pr.any():
                for c in find_contours(pr.astype(np.uint8), 0.5):
                    # Map contour coordinates to physical extent
                    # c[:, 1] is column (X), c[:, 0] is row (Y)
                    xs = extent[0] + (c[:, 1] / (pr.shape[1] - 1)) * (extent[1] - extent[0])
                    ys = extent[2] + (c[:, 0] / (pr.shape[0] - 1)) * (extent[3] - extent[2])
                    ax.plot(xs, ys, lw=1.15, color=cmap(k / max(1, len(projs) - 1)))
        ax.set_title(title)
        ax.set_xlabel("µm")
        ax.set_ylabel("µm")

    # ----- Combined triplanar figure (default unless only per-mask requested) -----
    if args.combined or not args.per_mask:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        draw(axs[0], bg_xy, projs_xy, ext_xy, "XY")
        draw(axs[1], bg_xz, projs_xz, ext_xz, "XZ")
        draw(axs[2], bg_yz, projs_yz, ext_yz, "YZ")
        fig.tight_layout()
        out_comb_png = OUT_DIR / ("selected_overlay_triplanar.png" if SELECTED_MASKS else "all_overlay_triplanar.png")
        out_comb_svg = OUT_DIR / ("selected_overlay_triplanar.svg" if SELECTED_MASKS else "all_overlay_triplanar.svg")
        fig.savefig(out_comb_png, dpi=160, bbox_inches="tight")
        fig.savefig(out_comb_svg, format='svg', bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved combined overlay: {out_comb_png} and {out_comb_svg}")

    # ----- Optional per-mask triplanar outputs -----
    if args.per_mask:
        for pth, m in zip(mask_paths, masks):
            name = _core_stem(pth)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            draw(axes[0], bg_xy, [m.any(axis=0)], ext_xy, "XY")
            draw(axes[1], bg_xz, [m.any(axis=1)], ext_xz, "XZ")
            draw(axes[2], bg_yz, [m.any(axis=2)], ext_yz, "YZ")
            fig.tight_layout()
            out_path_png = OUT_DIR / f"{name}_overlay_triplanar.png"
            out_path_svg = OUT_DIR / f"{name}_overlay_triplanar.svg"
            fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
            fig.savefig(out_path_svg, format='svg', bbox_inches="tight")
            plt.close(fig)
            print(f"  • Saved {out_path_png} and {out_path_svg}")

    print(f"All overlays saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
