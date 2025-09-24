#!/usr/bin/env python
"""
Module 3F: Funnel Detector + Reviewer (mask-restricted, 3D)

Hotkeys (Reviewer):
  Left/Right        : prev / next detected event
  1 / 2 / 3 / 0     : set class → funnel_pure / funnel_with_streak / streak_only / non_funnel
  b                 : toggle ΔF/F visibility
  g                 : toggle 3F tag visibility (core=1, streak=2)
  n                 : jump to next low-quality event (fails any q_* flag)
  Ctrl+S            : save funnels_review.csv

Outputs (…/<DATE>/<MOUSE>/<RUN>/funnels):
  - funnels.csv                (headers always present)
  - funnels_labels.tif         (uint8, (T,Z,Y,X): 0 bg, 1=core@peak, 2=streak)
  - qc/*.png                   (ZY/YX overlays per event)
  - funnels_review.csv         (after Ctrl+S)
"""

from pathlib import Path
from typing import Tuple, List
import csv
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import tifffile
import napari

from skimage.morphology import ball, remove_small_objects, binary_dilation
from scipy.ndimage import binary_opening, label as cc_label
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ================= CONFIG =================
DATE  = "2025-08-18"
MOUSE = "rAi162_15"
RUN   = "run9"

VOXEL_SIZE = (9.4, 1.0, 1.2)   # (Z,Y,X) μm
FPS = 10.0

# Deterministic Y alignment (crop the *larger* volume by 3 px at BOTTOM to match)
Y_CROP = 3

# ---- Detection (more sensitive, still safe) ----
TRACE_Q      = 97.5   # spatial percentile inside mask for per-time trace
K_MAD        = 1.5    # z gate on trace (lower → more sensitive)
EVENT_MERGE  = 10     # frames to bridge small inactive gaps
WIN_FRAMES   = 22     # ± window around peak to check propagation
MIN_VOX      = 20     # min voxels for a seed/core at peak
BRIGHT_FLOOR = 0.03   # absolute ΔF/F floor to avoid zero background

# ---- Core shaping (cap-aware) ----
CORE_TOP_PCT = 7.5    # include top X% brightest voxels (in dendrite) at peak
CORE_DILATE  = 5      # dilation (voxels) after shaping
CORE_MIN_VOX = 40     # final min core size (voxels)
CLOSE_RADIUS = 2      # small closing inside band

# ---- Geometry thresholds ----
NEXUS_BAND_UM = 7.0   # slab thickness around apex along trunk axis
TAU_NEXUS     = 0.50  # fraction of core inside nexus band
TAU_TRI       = 0.25  # triangularity threshold
TAU_PROP      = 0.30  # propagation score threshold (depth/10 μm)
N_MIN_STREAK  = 3     # min frames a streak voxel must be active
BRIGHT_DFF    = 0.25  # flag “bright” cores

# ---- Dilation around mask (search neighborhoods) ----
DILATE_CORE   = 2     # for peak/core detection
DILATE_STREAK = 3     # for propagation search

# Canonical CSV columns
EVENT_COLS = [
    "event_id","dend_id","t_peak","apex_z","apex_y","apex_x",
    "base_vox","depth_extent_um","triangularity","nexus_fraction",
    "propagation_score","classification","peak_dff","bright",
    "snr_local","q_snr_ok","q_size_ok","q_shape_ok","q_local_ok",
    "thr_dff","t0","t1"
]

# ================= PATHS =================
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
DFF_PATH       = BASE / "preprocessed" / f"runA_{RUN}_{MOUSE}_green_reslice_dff_stack.tif"  # (T,Z,Y,X)
LMAPS_DIR      = BASE / "labelmaps_curated_dynamic"
LABELS_3D_PATH = LMAPS_DIR / "labelmap_3d.tif"                                              # (Z,Y,X)
OUTPUT_FOLDER  = BASE / "funnels"; OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CSV_PATH   = OUTPUT_FOLDER / "funnels.csv"
REVIEW_CSV = OUTPUT_FOLDER / "funnels_review.csv"
TAG_TIFF   = OUTPUT_FOLDER / "funnels_labels.tif"
QC_DIR     = OUTPUT_FOLDER / "qc"; QC_DIR.mkdir(exist_ok=True)

# ================= HELPERS =================

def mad(x: np.ndarray) -> float:
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))

def read_tiff_any(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    return arr if arr.ndim == 4 else arr[None]

def write_tiff_4d(path: Path, arr: np.ndarray):
    tifffile.imwrite(str(path), arr, imagej=False, bigtiff=True, compression='zlib')

def _stem_id(p: Path) -> str:
    s = p.stem
    return s.replace("_labelmap", "")

def build_labelmap_3d_if_missing(
    labelmap_3d_path: Path,
    labelmaps_folder: Path,
    pattern: str = "dend_*_labelmap.tif",
    dtype=np.uint16
) -> Tuple[Path, List[str]]:
    """Merge curated per-dendrite masks into a single (Z,Y,X) labelmap if missing."""
    if labelmap_3d_path.exists():
        return labelmap_3d_path, []
    files = sorted(labelmaps_folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No curated masks in {labelmaps_folder} matching '{pattern}'")
    first = tifffile.imread(str(files[0]))
    if first.ndim != 3:
        raise ValueError(f"Expected 3D labelmap; got {first.shape} for {files[0]}")
    Z, Y, X = first.shape
    out = np.zeros((Z, Y, X), dtype=dtype)
    used = []
    for i, p in enumerate(files, start=1):
        m = tifffile.imread(str(p))
        if m.shape != (Z, Y, X):
            raise ValueError(f"Shape mismatch: {p} has {m.shape}, expected {(Z,Y,X)}")
        m = (m > 0)
        overlap = (out > 0) & m
        if overlap.any():
            print(f"⚠️  Overlap detected for {p.name}; preserving earlier labels in overlapped voxels.")
            m &= ~overlap
        out[m] = i
        used.append(str(p))
    labelmap_3d_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(labelmap_3d_path), out, imagej=False, bigtiff=True)
    print(f"🧩 Built merged labelmap: {labelmap_3d_path} (N={len(used)} masks)")
    idx_csv = labelmap_3d_path.parent / "labelmap_3d_index.csv"
    with open(idx_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dend_id", "file", "stem"])
        for i, p in enumerate(used, start=1):
            w.writerow([i, p, _stem_id(Path(p))])
    print(f"📝 Index: {idx_csv}")
    return labelmap_3d_path, used

def largest_component_containing_max(binary: np.ndarray, values: np.ndarray) -> np.ndarray:
    labeled, n = cc_label(binary)
    if n == 0:
        return np.zeros_like(binary, bool)
    idx = np.unravel_index(np.argmax(values), values.shape)
    lab = labeled[idx]
    if lab == 0:
        sizes = [(l, (labeled == l).sum()) for l in range(1, n + 1)]
        lab = max(sizes, key=lambda t: t[1])[0]
    return labeled == lab

def find_events(dff4d: np.ndarray, mask3d: np.ndarray) -> tuple[list[dict], dict]:
    """
    Build a tolerant trace *inside the mask* (high spatial percentile),
    gate by z (MAD) OR an absolute floor to avoid zero background.
    """
    T = dff4d.shape[0]
    flat = dff4d[:, mask3d]                          # (T, Nvox)
    trace = np.percentile(flat, TRACE_Q, axis=1)     # less spiky → good for wide caps
    base  = np.median(trace)
    sigma = mad(trace) + 1e-9
    z = (trace - base) / sigma

    floor = base + max(0.02, BRIGHT_FLOOR)
    active = (z >= K_MAD) | (trace >= floor)

    events = []
    i = 0
    while i < T:
        if not active[i]:
            i += 1; continue
        j = i
        gap = 0
        while j + 1 < T and (active[j + 1] or gap < EVENT_MERGE):
            j += 1
            gap = 0 if active[j] else (gap + 1)
        seg = slice(i, j + 1)
        t_peak = i + int(np.argmax(trace[seg]))
        events.append(dict(t_start=i, t_end=j, t_peak=t_peak))
        i = j + 1

    diag = dict(max_z=float(z.max()), max_trace=float(trace.max()),
                thr_z=float(K_MAD), n_active=int(active.sum()), n_events=len(events))
    return events, diag

def estimate_axis(mask: np.ndarray, apex_zyx: tuple[int,int,int]) -> np.ndarray:
    zyx = np.argwhere(mask)
    if len(zyx) < 10:
        return np.array([1.0, 0.0, 0.0])
    d = np.linalg.norm(zyx - np.array(apex_zyx), axis=1)
    keep = zyx[d <= np.percentile(d, 60)]
    if len(keep) < 10: keep = zyx
    pts = keep - keep.mean(0)
    cov = np.cov(pts.T)
    _, vecs = np.linalg.eigh(cov)
    v0 = vecs[:, -1]
    return v0 / (np.linalg.norm(v0) + 1e-9)

def choose_down(v0: np.ndarray, mask: np.ndarray, apex: tuple[int,int,int]) -> np.ndarray:
    zyx = np.argwhere(mask)
    rel = zyx - np.array(apex)
    pos = (rel @ v0 > 0).sum()
    neg = (rel @ (-v0) > 0).sum()
    return v0 if pos >= neg else -v0

def slab_selector(shape, apex, vdown, thickness_vox):
    Z, Y, X = shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    rel = np.stack([zz - apex[0], yy - apex[1], xx - apex[2]], axis=-1).reshape(-1,3)
    dist = np.abs(rel @ vdown)
    return (dist.reshape(Z,Y,X) <= (thickness_vox/2))

def halfspace_below(shape, apex, vdown):
    Z, Y, X = shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    rel = np.stack([zz - apex[0], yy - apex[1], xx - apex[2]], axis=-1).reshape(-1,3)
    return (rel @ vdown).reshape(Z,Y,X) > 0

def triangularity_score(active3d: np.ndarray, apex: tuple[int,int,int]) -> float:
    Z, Y, X = active3d.shape
    mip_zy = active3d.max(axis=2)
    row = apex[0]
    widths, depths = [], []
    top = min(Z-1, row+3); bottom = max(0, row-8)
    for z in range(row, top+1):
        line = mip_zy[z]
        if line.any():
            ys = np.where(line)[0]
            widths.append(ys.max()-ys.min()+1); depths.append(z - row)
    for z in range(row-1, bottom-1, -1):
        line = mip_zy[z]
        if line.any():
            ys = np.where(line)[0]
            widths.append(ys.max()-ys.min()+1); depths.append(z - row)
    if len(widths) < 3: return 0.0
    A = np.vstack([np.array(depths), np.ones(len(depths))]).T
    a, b = np.linalg.lstsq(A, np.array(widths), rcond=None)[0]
    yhat = A @ np.array([a,b])
    r2 = 1 - (((np.array(widths)-yhat)**2).sum()) / (((np.array(widths)-np.mean(widths))**2).sum() + 1e-9)
    return float(np.clip((r2 if a < 0 else 0.0), 0, 1))

def propagation_metrics(win4d: np.ndarray, thr: float, apex: tuple[int,int,int], vdown: np.ndarray,
                        band: np.ndarray, below: np.ndarray, dend_hood2: np.ndarray) -> tuple[float, np.ndarray, float]:
    T, Z, Y, X = win4d.shape
    streak = np.zeros_like(win4d, bool)
    max_depth_um = 0.0
    d_norm = 10.0
    for t in range(T):
        act = win4d[t] >= thr
        act = binary_opening(act, structure=ball(1))
        act = remove_small_objects(act, 20)
        cand = act & below & (~band) & dend_hood2
        if not cand.any():
            continue
        lab, n = cc_label(cand)
        for l in range(1, n+1):
            comp = lab == l
            zyx = np.argwhere(comp)
            if zyx.size:
                depth_um = float(np.max((zyx[:,0] - apex[0]) * VOXEL_SIZE[0]))
                max_depth_um = max(max_depth_um, depth_um)
            streak[t] |= comp
    alive = streak.sum(axis=0) >= N_MIN_STREAK
    streak &= alive[None]
    prop_score = float(np.clip(max_depth_um / d_norm, 0, 1))
    streak &= (~band)[None]
    return prop_score, streak, max_depth_um

def save_qc_panels(dff_peak, core, band, apex, t_peak, dend_id, ei):
    # ZY
    mip_zy = dff_peak.max(axis=2); c_zy = core.max(axis=2); b_zy = band.max(axis=2)
    fig = plt.figure(figsize=(6,5)); ax = plt.gca()
    ax.imshow(mip_zy, origin='upper'); ax.contour(c_zy, [0.5]); ax.contour(b_zy, [0.5], linestyles='--')
    ax.plot(apex[2], apex[0], 'o', ms=4); ax.set_title(f'd{dend_id} e{ei} t{t_peak} ZY')
    fig.tight_layout(); fig.savefig(QC_DIR / f'd{dend_id}_e{ei}_t{t_peak}_ZY.png', dpi=200); plt.close(fig)
    # YX
    mip_yx = dff_peak.max(axis=0); c_yx = core.max(axis=0); b_yx = band.max(axis=0)
    fig = plt.figure(figsize=(6,5)); ax = plt.gca()
    ax.imshow(mip_yx, origin='upper'); ax.contour(c_yx, [0.5]); ax.contour(b_yx, [0.5], linestyles='--')
    ax.plot(apex[2], apex[1], 'o', ms=4); ax.set_title(f'd{dend_id} e{ei} t{t_peak} YX')
    fig.tight_layout(); fig.savefig(QC_DIR / f'd{dend_id}_e{ei}_t{t_peak}_YX.png', dpi=200); plt.close(fig)

# ================= DETECTION =================

def detect_and_write():
    # 1) ensure labels
    build_labelmap_3d_if_missing(LABELS_3D_PATH, LMAPS_DIR)

    # 2) load volumes
    dff4d = read_tiff_any(DFF_PATH)                 # (T,Z,Y,X)
    labels3d = tifffile.imread(str(LABELS_3D_PATH)) # (Z,Y,X)

    # 3) deterministic Y-align by cropping the larger (bottom side only)
    _, Zd, Yd, Xd = dff4d.shape
    Zl, Yl, Xl = labels3d.shape
    if (Zd, Xd) != (Zl, Xl):
        raise ValueError(f"Z/X mismatch: DFF (Z={Zd},X={Xd}) vs labels (Z={Zl},X={Xl})")
    dy = Yd - Yl
    if dy == 0:
        pass
    elif abs(dy) == Y_CROP:
        if dy > 0:
            dff4d = dff4d[:, :, :-Y_CROP, :]
            print(f"✂️ Cropped DFF Y {Yd}→{dff4d.shape[2]}")
        else:
            labels3d = labels3d[:, :, :-Y_CROP, :]
            print(f"✂️ Cropped labels Y {Yl}→{labels3d.shape[1]}")
    else:
        raise ValueError(f"Y mismatch not equal to Y_CROP ({Y_CROP}): DFF Y={Yd}, labels Y={Yl}")

    T,Z,Y,X = dff4d.shape
    tag = np.zeros_like(dff4d, np.uint8)
    rows = []

    ids = [int(l) for l in np.unique(labels3d) if l != 0]
    for dend_id in ids:
        dend = labels3d == dend_id
        if dend.sum() < MIN_VOX:
            continue

        events, diag = find_events(dff4d, dend)
        print(f"[d{dend_id:02d}] events={diag['n_events']}  maxZ={diag['max_z']:.2f}  maxTrace={diag['max_trace']:.3f}")

        for ei, ev in enumerate(events):
            t_peak = ev['t_peak']
            t0 = max(0, t_peak - WIN_FRAMES); t1 = min(T-1, t_peak + WIN_FRAMES)
            dff_peak = dff4d[t_peak]
            vals = dff_peak[dend]

            # local threshold (robust) with absolute floor
            thr = float(np.median(vals) + K_MAD * (mad(vals) + 1e-9))
            thr = max(thr, float(np.median(vals) + BRIGHT_FLOOR))

            # === SEED inside mask neighborhood ===
            dend_hood = binary_dilation(dend, ball(DILATE_CORE))
            seed = (dff_peak >= thr) & dend_hood
            seed = binary_opening(seed, structure=ball(1))
            seed = remove_small_objects(seed, MIN_VOX)

            # fallback for tiny-but-bright
            if not seed.any():
                thr2 = float(np.median(vals) + max(1.0, K_MAD) * (mad(vals) + 1e-9))
                tiny = (dff_peak >= max(thr2, np.median(vals) + 0.10)) & dend_hood
                tiny = remove_small_objects(tiny, max(4, MIN_VOX // 2))
                seed = tiny
            if not seed.any():
                continue

            # keep component with peak max
            core = largest_component_containing_max(seed, dff_peak)

            # === CAP-AWARE EXPANSION (toward full nexus cap) ===
            apex = np.unravel_index(np.argmax(dff_peak*core), core.shape)
            v0 = estimate_axis(dend, apex)
            vdown = choose_down(v0, dend, apex)

            # find bright cap within the *whole dendrite*, clamp to a thicker band near apex
            mask_vals = dff_peak[dend]
            if mask_vals.size > 0:
                pct_thr = np.percentile(mask_vals, 100 - CORE_TOP_PCT)
                topk_all = (dff_peak >= pct_thr) & dend
                thickness_vox_cap = max(1, int(round(1.25 * NEXUS_BAND_UM / VOXEL_SIZE[0])))
                band_cap = slab_selector((Z,Y,X), apex, vdown, thickness_vox_cap)
                cap = topk_all & band_cap
                core = (core | cap)
                core = binary_opening(core, structure=ball(CLOSE_RADIUS)) | core
                core = binary_dilation(core, ball(CORE_DILATE))
                core &= dend
                core = largest_component_containing_max(core, dff_peak)

            core = remove_small_objects(core, CORE_MIN_VOX)
            if not core.any():
                continue

            # geometry
            thickness_vox = max(1, int(round(NEXUS_BAND_UM / VOXEL_SIZE[0])))
            band  = slab_selector((Z,Y,X), apex, vdown, thickness_vox)
            below = halfspace_below((Z,Y,X), apex, vdown)

            nexus_fraction = float((core & band).sum() / (core.sum() + 1e-9))
            tri = triangularity_score(core, apex)

            # propagation in a looser neighborhood
            dend_hood2 = binary_dilation(dend, ball(DILATE_STREAK))
            win = dff4d[t0:t1+1]
            prop_score, streak4d, max_depth_um = propagation_metrics(
                win, thr, apex, vdown, band, below, dend_hood2
            )

            # classification
            if nexus_fraction >= TAU_NEXUS and tri >= TAU_TRI:
                cls = 'funnel_pure' if prop_score < TAU_PROP else 'funnel_with_streak'
            elif prop_score >= TAU_PROP:
                cls = 'streak_only'
            else:
                cls = 'non_funnel'

            # tags
            tag[t_peak][core] = np.maximum(tag[t_peak][core], 1)
            tmp = np.zeros_like(tag, bool); tmp[t0:t1+1] = streak4d
            tag[tmp] = np.maximum(tag[tmp], 2)

            peak = float(dff_peak[apex])
            snr = float((peak - np.median(vals)) / (mad(vals) + 1e-9))

            print(f"  [e{ei:02d}] seed={int(seed.sum())} → core={int(core.sum())} vox | thr={thr:.3f} "
                  f"peak={peak:.3f} tri={tri:.2f} nx={nexus_fraction:.2f} prop={prop_score:.2f}")

            rows.append(dict(
                event_id=f"d{dend_id}_e{ei}", dend_id=dend_id, t_peak=t_peak,
                apex_z=int(apex[0]), apex_y=int(apex[1]), apex_x=int(apex[2]),
                base_vox=int(core.sum()), depth_extent_um=max_depth_um,
                triangularity=tri, nexus_fraction=nexus_fraction,
                propagation_score=prop_score, classification=cls,
                peak_dff=peak, bright=bool(peak>=BRIGHT_DFF),
                snr_local=snr, q_snr_ok=bool(snr>=3.0),
                q_size_ok=bool(core.sum()>=MIN_VOX),
                q_shape_ok=bool(tri>=TAU_TRI), q_local_ok=bool(nexus_fraction>=TAU_NEXUS),
                thr_dff=thr, t0=t0, t1=t1
            ))

            save_qc_panels(dff_peak, core, band, apex, t_peak, dend_id, ei)

    # 4) outputs
    df_out = pd.DataFrame(rows, columns=EVENT_COLS)
    df_out.to_csv(CSV_PATH, index=False)
    write_tiff_4d(TAG_TIFF, tag)
    print(f"✅ funnels.csv → {CSV_PATH} ({len(df_out)} events)")
    print(f"✅ funnels_labels.tif → {TAG_TIFF}")

# ================= REVIEWER =================

def reviewer():
    # load
    dff = read_tiff_any(DFF_PATH)
    labels = tifffile.imread(str(LABELS_3D_PATH))

    # same Y-crop logic for display
    _, Zd, Yd, Xd = dff.shape
    Zl, Yl, Xl = labels.shape
    if (Zd, Xd) == (Zl, Xl):
        dy = Yd - Yl
        if dy == 0:
            pass
        elif abs(dy) == Y_CROP:
            if dy > 0: dff = dff[:, :, :-Y_CROP, :]
            else:      labels = labels[:, :, :-Y_CROP, :]
        else:
            print(f"⚠️ Viewer: unexpected Y mismatch DFF={Yd}, labels={Yl} (expected ±{Y_CROP})")

    # robust contrast for display
    dff = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)
    dff = np.clip(dff, -0.5, 2.0)
    Ts = max(1, dff.shape[0] // 8); Zs = max(1, dff.shape[1] // 8)
    Ys = max(1, dff.shape[2] // 8); Xs = max(1, dff.shape[3] // 8)
    samp = dff[::Ts, ::Zs, ::Ys, ::Xs].ravel()
    lo, hi = (-0.1, 0.8) if samp.size == 0 else tuple(np.percentile(samp, (1, 99)))

    # events (tolerate empty CSV)
    try:
        df = pd.read_csv(CSV_PATH)
    except EmptyDataError:
        df = pd.DataFrame(columns=EVENT_COLS)

    v = napari.Viewer(ndisplay=3)
    dff_layer = v.add_image(dff, name='ΔF/F', contrast_limits=(lo,hi),
                            rendering='attenuated_mip', interpolation2d='nearest',
                            scale=VOXEL_SIZE)
    v.add_labels(labels, name='curated_3D', blending='translucent', opacity=0.35,
                 scale=VOXEL_SIZE)

    tag_layer = None
    try:
        tags = read_tiff_any(TAG_TIFF)
        tag_layer = v.add_labels(tags, name='3F_tags (0/1/2)', blending='additive',
                                 opacity=0.6, scale=VOXEL_SIZE)
    except FileNotFoundError:
        print("ℹ️ No tags file found; proceeding without tag layer.")

    if df.empty:
        print("ℹ️ No funnel events detected — viewer opened for inspection only.")
        napari.run(); return

    # ensure review cols
    if 'reviewed_class' not in df.columns: df['reviewed_class'] = ''
    if 'notes' not in df.columns: df['notes'] = ''

    i = [0]
    def goto(i_new: int):
        i[0] = max(0, min(len(df)-1, i_new))
        r = df.iloc[i[0]]
        v.dims.current_step = (int(r.t_peak), int(r.apex_z), int(r.apex_y), int(r.apex_x))
        msg = (f"{i[0]+1}/{len(df)} | {r.event_id} | auto={r.classification} | reviewed={r.reviewed_class or '-'}  "
               f"tri={r.triangularity:.2f} nx={r.nexus_fraction:.2f} prop={r.propagation_score:.2f} snr={r.snr_local:.1f}")
        print(msg)

    def set_class(c: str):
        df.at[df.index[i[0]], 'reviewed_class'] = c
        goto(i[0])

    # focus to canvas to ensure key events reach viewer
    try:
        v.window.qt_viewer.canvas.setFocus()
    except Exception:
        pass

    # --- Hotkeys (force-override defaults) ---
    @v.bind_key('Right', overwrite=True)
    def _next(viewer): goto(i[0] + 1)

    @v.bind_key('Left', overwrite=True)
    def _prev(viewer): goto(i[0] - 1)

    @v.bind_key('1', overwrite=True)
    def _c1(viewer): set_class('funnel_pure')

    @v.bind_key('2', overwrite=True)
    def _c2(viewer): set_class('funnel_with_streak')

    @v.bind_key('3', overwrite=True)
    def _c3(viewer): set_class('streak_only')

    @v.bind_key('0', overwrite=True)
    def _c0(viewer): set_class('non_funnel')

    @v.bind_key('b', overwrite=True)
    def _toggle_img(viewer):
        dff_layer.visible = not dff_layer.visible
        print(f"ΔF/F visible={dff_layer.visible}")

    @v.bind_key('g', overwrite=True)
    def _toggle_tags(viewer):
        if tag_layer is not None:
            tag_layer.visible = not tag_layer.visible
            print(f"tags visible={tag_layer.visible}")

    @v.bind_key('n', overwrite=True)
    def _next_lowq(viewer):
        start = i[0] + 1
        bad = df.index[(~df.q_snr_ok) | (~df.q_size_ok) | (~df.q_shape_ok) | (~df.q_local_ok)].tolist()
        after = [j for j in bad if j >= start]
        goto(after[0] if after else (bad[0] if bad else i[0]))

    @v.bind_key('Control-S', overwrite=True)
    def _save(viewer):
        df.to_csv(REVIEW_CSV, index=False)
        print(f"💾 Saved → {REVIEW_CSV}")

    print("=== REVIEWER HOTKEYS ===")
    print("Left/Right: prev/next  |  1/2/3/0: set class  |  b: ΔF/F on/off  |  g: tags on/off")
    print("n: next low-quality  |  Ctrl+S: save")

    goto(0)
    napari.run()

# ================= MAIN =================
if __name__ == "__main__":
    detect_and_write()
    reviewer()
