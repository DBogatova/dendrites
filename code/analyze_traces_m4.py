#!/usr/bin/env python
"""
Analyze Traces

This script analyzes calcium imaging traces to identify duplicates:
1. Loads trace data from CSV file
2. Calculates correlation between all trace pairs
3. Identifies groups of highly correlated traces (duplicates)
4. Saves results to CSV and text files
"""

import numpy as np
import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
DATE = "2025-04-22"
MOUSE = "rAi162_15"
RUN = "run6"

# === PATHS ===
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
TRACE_PATH = BASE / "traces" / "dff_traces_curated_bgsub.csv"
OUTPUT_DIR = BASE / "traces"

# === PARAMETERS ===
CORRELATION_THRESHOLD = 0.92  # Threshold for identifying duplicates

def find_duplicates(data):
    """
    Find duplicate traces based on correlation.
    
    Args:
        data: DataFrame containing trace data
        
    Returns:
        duplicate_groups: List of lists, each containing indices of duplicate traces
        corr_matrix: Correlation matrix between all traces
        trace_columns: Names of trace columns
    """
    # Get trace names and values
    trace_columns = data.columns[1:]  # Skip 'Frame' column
    traces = data[trace_columns].values.T
    
    # Calculate correlation matrix
    n_traces = traces.shape[0]
    corr_matrix = np.zeros((n_traces, n_traces))
    
    print("Calculating correlation matrix...")
    for i in range(n_traces):
        for j in range(i+1, n_traces):
            corr = np.corrcoef(traces[i], traces[j])[0, 1]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Group duplicates into clusters
    print("Identifying duplicate groups...")
    visited = set()
    duplicate_groups = []
    
    for i in range(n_traces):
        if i in visited:
            continue
            
        similar = np.where(corr_matrix[i] >= CORRELATION_THRESHOLD)[0].tolist()
        if len(similar) > 1:  # At least one other trace is similar to this one
            group = similar
            for idx in similar:
                visited.add(idx)
            duplicate_groups.append(sorted(group))
    
    return duplicate_groups, corr_matrix, trace_columns

def main():
    # Load traces
    print(f"Loading traces from {TRACE_PATH}")
    data = pd.read_csv(TRACE_PATH)
    
    # Find duplicates
    duplicate_groups, corr_matrix, trace_columns = find_duplicates(data)
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save correlation matrix
    print("Saving correlation matrix...")
    corr_df = pd.DataFrame(corr_matrix, 
                          index=trace_columns,
                          columns=trace_columns)
    corr_df.to_csv(output_dir / "trace_correlations.csv")
    
    # Print duplicate groups
    print("\nDuplicate trace groups:")
    if duplicate_groups:
        for i, group in enumerate(duplicate_groups):
            group_traces = [trace_columns[idx] for idx in group]
            print(f"Group {i+1}: {', '.join(group_traces)}")
    else:
        print("No duplicate traces found.")
    
    # Save duplicate groups to a file
    print("Saving duplicate groups to text file...")
    with open(output_dir / "duplicate_traces.txt", "w") as f:
        f.write(f"Duplicate traces (correlation threshold: {CORRELATION_THRESHOLD}):\n\n")
        if duplicate_groups:
            for i, group in enumerate(duplicate_groups):
                group_traces = [trace_columns[idx] for idx in group]
                f.write(f"Group {i+1}: {', '.join(group_traces)}\n")
        else:
            f.write("No duplicate traces found.\n")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()