#!/usr/bin/env python
"""
Module 2 — Original detector + 2D background for Napari

Behavior (matches your initial algorithm):
  1) Load each event crop (T,Z,Y,X)
  2) Threshold the whole event by a global percentile
  3) Identify active frames; group by small gaps
  4) Time-OR the frames in each group → 3D binary
  5) Gentle 2D clean per slice; 3D small-object removal
  6) 3D connected components; keep by volume range
  7) Deduplicate masks by Dice
  8) Save each 3D mask + a 2D background (Z-MIP of time-max) + preview + manifest row
"""

import gc
import csv
from pathlib import Path

import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops
from tqdm import tqdm

# ================== CONFIG ==================
DATE = "2025-08-27"
MOUSE = "rAi162_18"
RUN = "run7"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
EVENT_FOLDER = BASE / "preprocessed" / "event_crops"
OUT_LABELS   = BASE / "labelmaps"
OUT_PREV     = BASE / "labelmap_previews"
OUT_BGS      = BASE / "labelmap_backgrounds"   # 2D backgrounds for viz
MANIFEST     = BASE / "masks_manifest.csv"
for p in (OUT_LABELS, OUT_PREV, OUT_BGS): p.mkdir(parents=True, exist_ok=True)

# ---- Parameters (faithful to the original; adjust if needed) ----
VOXEL_SIZE = (9.4, 1.0, 1.2)       # (Z,Y,X) μm
VOXEL_VOL  = float(np.prod(VOXEL_SIZE))

INTENSITY_PERCENTILE = 99.9        # ← your original default
Y_CROP_BOTTOM = 3
MAX_FRAME_GAP = 2
MIN_EVENT_LENGTH = 1

MIN_VOL = 3000.0                    # μm³
MAX_VOL = None                 # μm³   (set to None to disable upper cap)

SLICE_OPEN_K  = 3                  # small open to tidy specks
SLICE_CLOSE_K = 3                  # small close to fill pinholes
SLICE_MIN_PIX = 20                 # 2D speck filter before 3D CC

DUPLICATE_DICE = 0.90
MAX_DENDRITES_TOTAL = 300

# ================== HELPERS ==================
def group_frames(frames, gap):
    if len(frames) == 0: return []
    frames = np.array(frames, dtype=int); frames.sort()
    groups, cur = [], [int(frames[0])]
    for f in frames[1:]:
        f = int(f)
        if f - cur[-1] <= gap: cur.append(f)
        else: groups.append(cur); cur = [f]
    groups.append(cur); return groups

def dice_coeff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape: return 0.0
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    tot   = a.sum(dtype=np.int64) + b.sum(dtype=np.int64)
    return (2.0 * inter / tot) if tot > 0 else 0.0

def z_mip_background_2d(stack_TZYX, t0, t1) -> np.ndarray:
    """2D background for Napari: Z-MIP of time-max over the event window."""
    seg = stack_TZYX[t0:t1]
    if seg.size == 0: return np.zeros(stack_TZYX.shape[2:], dtype=np.float16)
    vol_tmax = np.max(seg, axis=0)      # (Z,Y,X)
    return np.max(vol_tmax, axis=0).astype(np.float16)  # (Y,X)

# ================== MAIN ==================
def main():
    event_paths = sorted(EVENT_FOLDER.glob("event_group_*.tif"))
    print(f"Found {len(event_paths)} grouped event files.")

    raw_masks = []  # tuples: (mask3d, vol_um3, event_path, t_start, t_end, bg2d)
    extracted = 0

    se_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLICE_OPEN_K, SLICE_OPEN_K))
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLICE_CLOSE_K, SLICE_CLOSE_K))

    for path in tqdm(event_paths, desc="Extracting dendrites"):
        if extracted >= MAX_DENDRITES_TOTAL: break

        stack = tifffile.imread(path).astype(np.float32)  # (T,Z,Y,X)
        if Y_CROP_BOTTOM > 0:
            stack = stack[:, :, :-Y_CROP_BOTTOM, :]

        T, Z, Y, X = stack.shape
        if T == 0:
            del stack; continue

        # 1) Global percentile threshold over the whole event (original behavior)
        thr = float(np.percentile(stack, INTENSITY_PERCENTILE))
        binary = stack > thr                                # (T,Z,Y,X)

        # 2) Find active frames; group small gaps
        activity = binary.max(axis=(1, 2, 3))
        active_ts = np.where(activity > 0)[0]
        frame_groups = group_frames(active_ts, MAX_FRAME_GAP)

        # 3) For each group → 3D mask from time-OR; clean; 3D CC; volume filter
        for group in frame_groups:
            if extracted >= MAX_DENDRITES_TOTAL: break
            if len(group) < MIN_EVENT_LENGTH: continue

            t_start = int(group[0])
            t_end   = int(group[-1]) + 1

            # Time-OR to 3D (Z,Y,X)
            mask_3d = np.any(binary[t_start:t_end], axis=0)

            if not mask_3d.any(): continue

            # Gentle per-slice open/close and 2D speck filter
            cleaned = np.zeros_like(mask_3d, dtype=np.uint8)
            for z in range(Z):
                sl = (mask_3d[z].astype(np.uint8) * 255)
                sl = cv2.morphologyEx(sl, cv2.MORPH_OPEN,  se_open)
                sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, se_close)
                slb = sl > 0
                if SLICE_MIN_PIX > 0:
                    lbl2 = label(slb); keep2 = np.zeros_like(slb, bool)
                    for i2 in range(1, int(lbl2.max())+1):
                        rr = (lbl2 == i2)
                        if rr.sum() >= SLICE_MIN_PIX:
                            keep2 |= rr
                    slb = keep2
                cleaned[z] = slb

            # 3D speck removal by physical volume
            cleaned = remove_small_objects(cleaned.astype(bool), int(MIN_VOL / VOXEL_VOL), connectivity=1)
            if not cleaned.any(): continue

            # 3D connected components; keep by volume range
            lbl3 = label(cleaned)
            props = sorted(regionprops(lbl3), key=lambda r: r.area, reverse=True)

            # 4) Save components that pass volume gates
            for r in props:
                voxels = int(r.area)
                vol_um3 = voxels * VOXEL_VOL
                if vol_um3 < MIN_VOL: continue
                if (MAX_VOL is not None) and (vol_um3 > MAX_VOL): continue

                m = (lbl3 == r.label).astype(np.uint8)

                # Per-mask 2D background from the same time window (fast)
                bg2d = z_mip_background_2d(stack, t_start, t_end)

                raw_masks.append((m, float(vol_um3), str(path), t_start, t_end, bg2d))
                extracted += 1
                if extracted >= MAX_DENDRITES_TOTAL: break

        del stack, binary
        gc.collect()

    # ====== DEDUP BY 3D DICE ======
    print("Deduplicating masks...")
    unique = []
    for m, vol, ep, ts, te, bg in raw_masks:
        is_dup = False
        for u in unique:
            if m.shape != u["mask"].shape: continue
            if dice_coeff(m.astype(bool), u["mask"].astype(bool)) > DUPLICATE_DICE:
                is_dup = True; break
        if not is_dup:
            unique.append({"mask": m, "vol": vol, "event_path": ep, "t_start": ts, "t_end": te, "bg2d": bg})

    print(f"[✓] Retained {len(unique)} unique dendrites out of {len(raw_masks)}")

    # ====== SAVE MASKS, BACKGROUNDS, PREVIEWS, MANIFEST ======
    print("Saving outputs...")
    with open(MANIFEST, "w", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "dend_id",
                "labelmap_path",
                "background_path",
                "source_event_file",
                "event_t_start",
                "event_t_end",
                "voxel_size_z",
                "voxel_size_y",
                "voxel_size_x",
                "volume_um3",
            ],
        )
        writer.writeheader()

        for i, e in enumerate(unique):
            mask = e["mask"].astype(np.uint8)           # (Z,Y,X)
            bg2d = e["bg2d"]                            # (Y,X) float16
            src  = Path(e["event_path"]).name

            mask_path = OUT_LABELS / f"dend_{i:03d}_labelmap.tif"
            bg_path   = OUT_BGS   / f"dend_{i:03d}_background_2dMIP.tif"

            # Mask as single-object labelmap with (i+1)
            tifffile.imwrite(mask_path, (mask * (i + 1)).astype(np.uint16))
            # 2D background image
            tifffile.imwrite(bg_path, bg2d, dtype=np.float16)

            # Quick preview: overlay Z-MIP of mask on background
            mip_mask = np.max(mask, axis=0).astype(bool)
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(bg2d.astype(np.float32), cmap="gray")
            edges = cv2.Canny((mip_mask.astype(np.uint8) * 255), 0, 1) > 0
            ax.imshow(np.ma.masked_where(~edges, edges), cmap="autumn", alpha=0.8)
            ax.set_title(f"dend_{i:03d} (Z-MIP overlay)")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(OUT_PREV / f"dend_{i:03d}_preview.png", dpi=150)
            plt.close(fig)

            writer.writerow({
                "dend_id": i,
                "labelmap_path": str(mask_path),
                "background_path": str(bg_path),
                "source_event_file": src,
                "event_t_start": int(e["t_start"]),
                "event_t_end": int(e["t_end"]),
                "voxel_size_z": VOXEL_SIZE[0],
                "voxel_size_y": VOXEL_SIZE[1],
                "voxel_size_x": VOXEL_SIZE[2],
                "volume_um3": float(e["vol"]),
            })

    print(f"Manifest saved to: {MANIFEST}")
    print("Module 2 (original + background) complete.")

if __name__ == "__main__":
    main()