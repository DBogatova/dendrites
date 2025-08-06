import shutil
from pathlib import Path

# Parameters
mouse = "rAi162_15"
run = "run6"
date = "2025-04-22"

# Destination layout
base = Path("data") / date / mouse / run
layout = {
    "raw": base / "raw",
    "events": base / "events",
    "labelmaps": base / "labelmaps",
    "branches": base / "branches",
    "traces": base / "traces",
    "overlays": base / "overlays"
}

# Create folders
for name, path in layout.items():
    path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created: {path}")

# Define what to move
move_plan = [
    ("tiff-stacks/2025-04-22/runA_run6_rAi162_15_reslice_bin.tif", layout["raw"]),
    ("tiff-stacks/2025-04-22/runA_run6_rAi162_15_processed_very_good.tif", layout["raw"]),
    *[(p, layout["events"]) for p in Path("mini-stacks-for-labeling").glob("event_*.tif")],
    *[(p, layout["labelmaps"]) for p in Path("mini-stacks-for-labeling/dendrite_labelmaps").glob("*.tif")],
    *[(p, layout["branches"]) for p in Path("segmented-tiffs/branch_labels").glob("*.tif")],
    *[(p, layout["traces"]) for p in Path("segmented-tiffs/trace_with_branchmap").glob("*.png")],
    ("segmented-tiffs/dff_traces.pkl", layout["traces"]),
    ("segmented-tiffs/multi_dendrites_dff_branches.tif", layout["overlays"]),
]

# Move files
print("\n=== MOVING FILES ===\n")
for src, dst_folder in move_plan:
    src_path = Path(src)
    if not src_path.exists():
        print(f"[SKIP] Missing: {src}")
        continue
    try:
        dst_path = dst_folder / src_path.name
        shutil.move(str(src_path), str(dst_path))
        print(f"[MOVED] {src_path} → {dst_path}")
    except Exception as e:
        print(f"[ERROR] Could not move {src_path}: {e}")
