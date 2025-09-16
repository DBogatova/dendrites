#!/usr/bin/env python
"""
Module 9: Peri-event ACh analyses with shuffle controls (with dendrite selection)

Inputs (from Module 5D):
  BASE/traces_caBG_achNeighborhood/*_CaBG_AChNeighborhood.csv
  Columns: time_s, Ca_bgsub_pct, ACh_core_pct, ACh_ring1_pct, ACh_ring2_pct

Selection:
  Choose which dendrites to include in both plots and statistics:
  - SELECT_MODE = "all" | "only" | "exclude"
  - DENDRITES_LIST = ["dend_001", "dend_010", ...] or empty
  - DENDRITES_LIST_FILE = path to a txt file with one dendrite name per line (optional)

Outputs (for the current RUN only):
  BASE/eta_results/
    per_mask_plots/<name>_ETA.pdf
    per_event.csv
    per_dendrite.csv
    per_run_summary.csv
    run_overview.pdf
    selection_log.txt         (what was included/excluded)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from scipy.stats import sem
import gc

# -------- Matplotlib sane defaults --------
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.default'] = 'regular'

# ========= USER CONFIG (video-scoped) =========
DATE = "2025-08-27"
MOUSE = "rAi162_18"
RUN   = "run7"

# ---- DENDRITE SELECTION ----
# Modes:
#  "all"     -> analyze all dendrites found in traces folder
#  "only"    -> analyze ONLY those listed (names must match 'dend_###' stems)
#  "exclude" -> analyze all dendrites EXCEPT those listed
SELECT_MODE = "only"
DENDRITES_LIST = [
   "dend_001", "dend_004", "dend_010", "dend_011",
    "dend_015", "dend_016", "dend_017", "dend_035"
]
# Optional: path to a text file with one dendrite name per line (overrides/extends DENDRITES_LIST)
DENDRITES_LIST_FILE = "Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data/2025-08-27/rAi162_18/good_dendrites.txt"  # e.g. "/path/to/keep_list.txt"

# Optional event-quality filter
MIN_EVENTS_PER_DENDRITE = 1   # discard dendrites with fewer events than this

# ---- Event detection on Ca (core−shell %ΔF/F) ----
CA_MIN_PROMINENCE = 2.0     # % ΔF/F
CA_MIN_DISTANCE_S = 2.0     # s refractory between Ca events
CA_MIN_HEIGHT     = 0.5     # % ΔF/F (floor)

# ---- Peri-event window (seconds) ----
WIN_PRE_S  = 8.0
WIN_POST_S = 8.0

# ---- Shuffle control ----
N_SHUFFLES     = 500
REFRACTORY_S   = 1.5
RANDOM_SEED    = 1337

# ---- ACh effect windows (seconds) ----
ACH_PEAK_WINDOW = (0.0, 3.0)   # where we look for ACh peak post Ca onset
ACH_BASELINE    = (-3.0, 0.0)  # pre-event baseline window (within peri window)

# ---- Plotting ----
SMOOTH_WINDOW = 1  # samples; 1 disables extra smoothing

# ========= PATHS =========
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE         = PROJECT_ROOT / "data" / DATE / MOUSE / RUN

TRACES_DIR   = BASE / "traces_caBG_achNeighborhood"
OUT_DIR      = BASE / "eta_results"
PER_MASK_DIR = OUT_DIR / "per_mask_plots"
for p in (OUT_DIR, PER_MASK_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ========= HELPERS =========
def moving_avg(x, w):
    if w is None or w <= 1:
        return x
    w = int(w)
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")

def s_to_idx(t, s):
    dt = np.median(np.diff(t))
    return int(np.round(s / dt))

def extract_windows(signal, centers_idx, half_left, half_right):
    L = half_left + half_right + 1
    out, keep = [], []
    N = len(signal)
    for c in centers_idx:
        a, b = c - half_left, c + half_right + 1
        if a < 0 or b > N:
            continue
        out.append(signal[a:b])
        keep.append(c)
    return (np.array(out) if out else np.zeros((0, L), float)), np.array(keep, int)

def baseline_correct(wins, left_idx, right_idx):
    if wins.size == 0:
        return wins
    base = wins[:, left_idx:right_idx].mean(axis=1, keepdims=True)
    return wins - base

def random_events_in_range(n_events, N, min_dist, rng):
    if n_events <= 0:
        return np.array([], dtype=int)
    attempts = 0
    while attempts < 2000:
        cand = np.sort(rng.integers(0, N, size=n_events))
        if len(cand) <= 1 or np.min(np.diff(cand)) >= min_dist:
            return cand
        attempts += 1
    # fallback: greedy
    cand = np.sort(rng.integers(0, N, size=n_events * 3))
    picked = [cand[0]]
    for c in cand[1:]:
        if c - picked[-1] >= min_dist:
            picked.append(c)
        if len(picked) >= n_events:
            break
    return np.array(picked[:n_events], dtype=int)

def load_name_list():
    names = set([n.strip() for n in DENDRITES_LIST if n.strip()])
    if DENDRITES_LIST_FILE:
        p = Path(DENDRITES_LIST_FILE)
        if p.exists():
            with p.open() as f:
                for line in f:
                    s = line.strip()
                    if s:
                        names.add(s)
    return sorted(names)

# ========= MAIN =========
def main():
    # Discover trace CSVs (one per mask)
    csvs_all = sorted(TRACES_DIR.glob("*_CaBG_AChNeighborhood.csv"))
    if not csvs_all:
        raise FileNotFoundError(f"No Module 5D CSVs found in {TRACES_DIR}")

    # Build selection set
    listed = load_name_list()  # list like ["dend_001", ...]
    listed_set = set(listed)

    # Partition by name
    def stem_to_maskname(path: Path):
        return path.stem.replace("_CaBG_AChNeighborhood", "")

    all_names = [stem_to_maskname(p) for p in csvs_all]
    include_names = []
    if SELECT_MODE == "all":
        include_names = all_names
    elif SELECT_MODE == "only":
        include_names = [n for n in all_names if n in listed_set]
    elif SELECT_MODE == "exclude":
        include_names = [n for n in all_names if n not in listed_set]
    else:
        raise ValueError(f"SELECT_MODE must be one of 'all'|'only'|'exclude', got {SELECT_MODE}")

    name_to_path = {stem_to_maskname(p): p for p in csvs_all}
    csvs = [name_to_path[n] for n in include_names if n in name_to_path]

    # Log selection
    sel_log = OUT_DIR / "good_dendrites.txt"
    with sel_log.open("w") as f:
        f.write(f"RUN = {RUN} | SELECT_MODE = {SELECT_MODE}\n")
        f.write(f"Listed names ({len(listed)}): {listed}\n\n")
        f.write(f"All found ({len(all_names)}): {all_names}\n\n")
        f.write(f"Included ({len(include_names)}): {include_names}\n")
        f.write(f"Excluded ({len(set(all_names)-set(include_names))}): {sorted(set(all_names)-set(include_names))}\n")
    print(f"Selection log → {sel_log}")

    if not csvs:
        print("Nothing to analyze after selection; exiting.")
        return

    # Pre-read first file for timing/grid
    tmp = pd.read_csv(csvs[0])
    t = tmp["time_s"].to_numpy()
    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    half_left  = s_to_idx(t, WIN_PRE_S)
    half_right = s_to_idx(t, WIN_POST_S)
    L = half_left + half_right + 1

    base_l = s_to_idx(t, ACH_BASELINE[0] + WIN_PRE_S)
    base_r = s_to_idx(t, ACH_BASELINE[1] + WIN_PRE_S)
    peak_l = s_to_idx(t, ACH_PEAK_WINDOW[0] + WIN_PRE_S)
    peak_r = s_to_idx(t, ACH_PEAK_WINDOW[1] + WIN_PRE_S)

    min_dist_samples = int(np.round(CA_MIN_DISTANCE_S * fs))
    shuffle_min_dist = int(np.round(REFRACTORY_S * fs))

    rng = np.random.default_rng(RANDOM_SEED)

    per_event_rows = []
    per_dend_rows  = []

    for path in csvs:
        name = path.stem.replace("_CaBG_AChNeighborhood", "")
        df = pd.read_csv(path)
        t = df["time_s"].to_numpy()
        ca = df["Ca_bgsub_pct"].to_numpy()
        a_core = df["ACh_core_pct"].to_numpy()
        a_r1   = df["ACh_ring1_pct"].to_numpy()
        a_r2   = df["ACh_ring2_pct"].to_numpy()

        # Display smoothing (optional)
        ca_dsp    = moving_avg(ca, SMOOTH_WINDOW)
        a_core_dsp= moving_avg(a_core, SMOOTH_WINDOW)
        a_r1_dsp  = moving_avg(a_r1, SMOOTH_WINDOW)
        a_r2_dsp  = moving_avg(a_r2, SMOOTH_WINDOW)

        # --- Ca event detection ---
        peaks, _ = find_peaks(
            ca, prominence=CA_MIN_PROMINENCE, distance=min_dist_samples, height=CA_MIN_HEIGHT
        )
        if len(peaks) == 0:
            print(f"[{name}] No Ca events (prom>{CA_MIN_PROMINENCE}%, dist>{CA_MIN_DISTANCE_S}s). Skipped.")
            continue

        # Windows
        ca_wins,  peaks_kept = extract_windows(ca_dsp,    peaks, half_left, half_right)
        aC_wins, _           = extract_windows(a_core_dsp, peaks, half_left, half_right)
        a1_wins, _           = extract_windows(a_r1_dsp,   peaks, half_left, half_right)
        a2_wins, _           = extract_windows(a_r2_dsp,   peaks, half_left, half_right)

        # Edge-safe baseline correction
        ca_wins = baseline_correct(ca_wins, base_l, base_r)
        aC_wins = baseline_correct(aC_wins, base_l, base_r)
        a1_wins = baseline_correct(a1_wins, base_l, base_r)
        a2_wins = baseline_correct(a2_wins, base_l, base_r)

        nE = ca_wins.shape[0]
        if nE < MIN_EVENTS_PER_DENDRITE:
            print(f"[{name}] Only {nE} events (<{MIN_EVENTS_PER_DENDRITE}). Skipped.")
            continue

        # ETAs and SEMs
        eta_ca,  sem_ca  = np.nanmean(ca_wins, axis=0),  sem(ca_wins, axis=0, nan_policy="omit")
        eta_aC,  sem_aC  = np.nanmean(aC_wins, axis=0),  sem(aC_wins, axis=0, nan_policy="omit")
        eta_a1,  sem_a1  = np.nanmean(a1_wins, axis=0),  sem(a1_wins, axis=0, nan_policy="omit")
        eta_a2,  sem_a2  = np.nanmean(a2_wins, axis=0),  sem(a2_wins, axis=0, nan_policy="omit")

        # Shuffle for ACh core ETA
        N = len(ca)
        sh_eta_core = []
        for _ in range(N_SHUFFLES):
            sidx = random_events_in_range(nE, N, shuffle_min_dist, rng)
            w, _ = extract_windows(a_core_dsp, sidx, half_left, half_right)
            if w.size == 0:
                continue
            w = baseline_correct(w, base_l, base_r)
            sh_eta_core.append(np.nanmean(w, axis=0))
        sh_eta_core = np.array(sh_eta_core) if len(sh_eta_core) else np.zeros((0, L))

        # Per-event metrics
        t_win = (np.arange(-half_left, half_right + 1) * dt)
        for k, c_idx in enumerate(peaks_kept):
            post = slice(peak_l, peak_r)
            aC_ev = aC_wins[k]
            peak_amp = float(np.nanmax(aC_ev[post]))
            latency_idx = int(np.nanargmax(aC_ev[post]))
            latency_s = (peak_l + latency_idx - half_left) * dt
            area = float(np.trapz(aC_ev[post], t_win[post]))
            ca_ev = ca_wins[k]
            ca_peak = float(np.nanmax(ca_ev[post]))

            per_event_rows.append({
                "mouse": MOUSE, "run": RUN, "mask": name,
                "event_frame": int(c_idx),
                "n_events_for_mask": int(nE),
                "ACh_core_peak_pct": peak_amp,
                "ACh_core_latency_s": latency_s,
                "ACh_core_area_pctxs": area,
                "Ca_peak_pct": ca_peak,
            })

        # Per-dendrite summary + shuffle p
        if sh_eta_core.shape[0] > 0:
            real_peak = float(np.nanmax(eta_aC[peak_l:peak_r]))
            sh_peaks = np.nanmax(sh_eta_core[:, peak_l:peak_r], axis=1)
            p_right = float((np.sum(sh_peaks >= real_peak) + 1) / (len(sh_peaks) + 1))
        else:
            real_peak, p_right = float(np.nanmax(eta_aC[peak_l:peak_r])), np.nan

        per_dend_rows.append({
            "mouse": MOUSE, "run": RUN, "mask": name,
            "n_events": int(nE),
            "ETA_ACh_core_peak_pct": real_peak,
            "ETA_ACh_core_peak_p_right": p_right,
            "ETA_Ca_peak_pct": float(np.nanmax(eta_ca[peak_l:peak_r])),
        })

        # Plot per-mask ETA
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax1.plot(t_win, eta_ca, color=(0.2,0.7,0.2), lw=1.8, label="Ca (mean)")
        ax1.fill_between(t_win, eta_ca - sem_ca, eta_ca + sem_ca, color=(0.2,0.7,0.2), alpha=0.2)
        ax1.axvline(0, color='k', lw=1, alpha=0.5)
        ax1.set_ylabel("Ca ΔF/F (%)")
        ax1.set_title(f"{name} — Event-triggered averages")
        ax1.grid(alpha=0.3)

        ax2.plot(t_win, eta_aC, color=(0.85,0.1,0.1), lw=1.6, label="ACh core")
        ax2.fill_between(t_win, eta_aC - sem_aC, eta_aC + sem_aC, color=(0.85,0.1,0.1), alpha=0.15)
        ax2.plot(t_win, eta_a1, color=(0.75,0.15,0.15), lw=1.2, label="ACh near")
        ax2.plot(t_win, eta_a2, color=(0.60,0.20,0.20), lw=1.0, label="ACh far")
        ax2.axvline(0, color='k', lw=1, alpha=0.5)
        ax2.set_xlabel("Time from Ca event (s)")
        ax2.set_ylabel("ACh ΔF/F (%)")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(PER_MASK_DIR / f"{name}_ETA.pdf", format="pdf")
        plt.close(fig)

        gc.collect()

    # Save CSVs
    per_event_df = pd.DataFrame(per_event_rows)
    per_dend_df  = pd.DataFrame(per_dend_rows)

    per_event_csv = OUT_DIR / "per_event.csv"
    per_dend_csv  = OUT_DIR / "per_dendrite.csv"
    per_event_df.to_csv(per_event_csv, index=False)
    per_dend_df.to_csv(per_dend_csv, index=False)

    # Per-run summary + overview
    if not per_dend_df.empty:
        run_summary = (per_dend_df
            .groupby(["mouse","run"])
            .agg(
                n_dendrites=("mask","nunique"),
                mean_ACh_peak=("ETA_ACh_core_peak_pct","mean"),
                median_ACh_peak=("ETA_ACh_core_peak_pct","median"),
                frac_sig=("ETA_ACh_core_peak_p_right", lambda x: np.mean(x < 0.05) if len(x)>0 else np.nan),
                mean_Ca_peak=("ETA_Ca_peak_pct","mean"),
            )
            .reset_index()
        )
        run_summary_csv = OUT_DIR / "per_run_summary.csv"
        run_summary.to_csv(run_summary_csv, index=False)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(per_dend_df["ETA_ACh_core_peak_pct"].dropna(), bins=20, edgecolor='k', alpha=0.85)
        ax.set_xlabel("ACh ETA peak (0–3 s) [%]")
        ax.set_ylabel("Dendrite count")
        ax.set_title(f"Run overview — {MOUSE} {RUN} (selected dendrites)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "run_overview.pdf", format="pdf")
        plt.close(fig)

        print(f"Saved run summary → {run_summary_csv}")
        print(f"Overview figure → {OUT_DIR/'run_overview.pdf'}")

    print(f"Per_event.csv → {per_event_csv}")
    print(f"Per_dendrite.csv → {per_dend_csv}")
    print(f"Per-mask ETA plots → {PER_MASK_DIR}")
    print(f"Selection mode = {SELECT_MODE}; analyzed {len(per_dend_df)} dendrites.")

if __name__ == "__main__":
    main()
