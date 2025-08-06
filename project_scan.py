import os
from pathlib import Path

# List of known-used paths (update if needed)
used_paths = [
    "tiff-stacks/2025-04-22/runA_run6_rAi162_15_reslice_bin.tif",
    "tiff-stacks/2025-04-22/runA_run6_rAi162_15_processed_very_good.tif",
    "mini-stacks-for-labeling",
    "segmented-tiffs/dff_traces.pkl",
    "segmented-tiffs/branch_labels",
    "segmented-tiffs/multi_dendrites_dff_branches.tif",
    "segmented-tiffs/trace_with_branchmap"
]

# Normalize
used_set = set(Path(p).resolve() for p in used_paths)
project_root = Path(".").resolve()

print("\n=== Unused Files & Folders ===\n")
for path in project_root.rglob("*"):
    # Skip system/hidden files
    if path.name.startswith("."):
        continue

    # Skip if path or any parent is in used list
    if any(str(used) in str(path) for used in used_set):
        continue

    print(path.relative_to(project_root))
