#!/usr/bin/env python
"""
Trace QA + Duplicate Detection (clean summary)

- Loads ΔF/F traces from CSV (first column = Frame or Time)
- Finds duplicate masks via correlation
- Scores each trace as GOOD / OK / NOISE
- Prints per-trace summary: "trace_name   label"
- Saves detailed outputs (features, correlations, duplicates)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq

# ========= CONFIG =========
DATE  = "2025-08-29"
MOUSE = "rAi162_18"
RUN   = "run6"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
TRACE_PATH = BASE / "traces" / "dff_traces_curated_bgsub.csv"
OUTPUT_DIR = BASE / "traces"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORR_THRESHOLD = 0.92

# QA thresholds
MIN_EVENT_PROM_PCT  = 0.8
GOOD_SNR            = 8.0
OK_SNR              = 5.5
GOOD_SPECTRAL_RATIO = 5.0
OK_SPECTRAL_RATIO   = 3.8
MAX_NEG_FRAC        = 0.2
MAX_DRIFT_PCT       = 3.0

# ==========================

def robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + 1e-9

def bandpower_ratio(x, fps, low1=0.05, high1=2.0, low2=5.0, high2=20.0):
    n = len(x)
    freqs = rfftfreq(n, d=1.0/fps)
    P = np.abs(rfft(x))**2
    def _int(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return P[m].sum() + 1e-12
    return _int(low1, high1) / _int(low2, high2)

def preprocess_traces(df):
    tcol = df.columns[0]
    t = df[tcol].values
    if t.max() > 1e3:  # frames
        fps = 10.0
        time = t / fps
    else:              # seconds
        time = t
        dt = np.median(np.diff(time))
        fps = 1.0 / dt if dt > 0 else 10.0
    traces = df.iloc[:,1:].astype(float).copy()
    traces = traces.apply(lambda x: gaussian_filter1d(x.values, 1), axis=0, result_type='expand')
    return time, fps, traces

def trace_features(x, fps):
    sig = robust_sigma(x)
    idx, props = find_peaks(x, prominence=max(MIN_EVENT_PROM_PCT, 2*sig))
    n_evt = len(idx)
    max_prom = float(props['prominences'].max()) if n_evt else 0.0
    snr_peak = max_prom / (sig + 1e-9)
    neg_frac = float((x < -2.5*sig).mean())
    skew = float(pd.Series(x).skew())
    ac1 = float(np.corrcoef(x[:-1], x[1:])[0,1]) if len(x) > 3 else 0.0
    spr = bandpower_ratio(x - np.median(x), fps)
    thirds = np.array_split(np.asarray(x), 3)
    drift = abs(np.median(thirds[-1]) - np.median(thirds[0]))
    return dict(n_events=n_evt, snr_peak=snr_peak, neg_frac=neg_frac,
                skew=skew, ac1=ac1, spectral_ratio=spr, drift=drift)

def label_quality(feat):
    if (feat['neg_frac'] > MAX_NEG_FRAC) or (feat['drift'] > MAX_DRIFT_PCT):
        return "NOISE"
    if (feat['snr_peak'] >= GOOD_SNR and feat['spectral_ratio'] >= GOOD_SPECTRAL_RATIO 
        and feat['n_events'] >= 1 and feat['skew'] > 0 and feat['ac1'] > 0.15):
        return "GOOD"
    if (feat['snr_peak'] >= OK_SNR and feat['spectral_ratio'] >= OK_SPECTRAL_RATIO 
        and feat['ac1'] > 0.05):
        return "OK"
    return "NOISE"

def main():
    df = pd.read_csv(TRACE_PATH)
    time, fps, traces = preprocess_traces(df)

    qa_rows = []
    print("\n=== Trace Quality Summary ===")
    for name in traces.columns:
        x = traces[name].values.astype(float)
        feat = trace_features(x, fps)
        label = label_quality(feat)
        feat['trace'] = name
        feat['label'] = label
        qa_rows.append(feat)
        print(f"{name:15s}  {label}")   # <<< SIMPLE SUMMARY LINE

    qa = pd.DataFrame(qa_rows).set_index('trace')
    qa.to_csv(OUTPUT_DIR / "traces_quality.csv")

    print("\nSaved detailed results to traces_quality.csv")

if __name__ == "__main__":
    main()
