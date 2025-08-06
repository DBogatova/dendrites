from pathlib import Path
import os

# Replace with confirmed list of unused files
unused_paths = [
    "code/scripts_to_delete/pca_on_mip.py",
    "code/scripts_to_delete/extract_one_dendrite.py",
    "code/scripts_to_delete/3d_mask_overlay_m3.py",
    "code/scripts_to_delete/3d_view_dendrites.py",
    "code/scripts_to_delete/anna_extract_dendrites.py",
    "code/scripts_to_delete/mask_from_pca.py",
    "code/scripts_to_delete/2d_mask_overlay.py",
    "code/scripts_to_delete/extract-dendrites.py",
    "code/scripts_to_delete/anna_3d_mask.py",
    "code/scripts_to_delete/grow_mask.py",
    "code/scripts_to_delete/dendrites_branches_corr.py",
    "code/scripts_to_delete/multi_3d_mask_overlay_m4.py",
    "code/scripts_to_delete/draw_mask_failedm2.py",
    "code/scripts_to_delete/mask_from_mip.py",
    "code/scripts_to_delete/mask_debug.py",
    "code/scripts_to_delete/background_removal.py",
    "code/scripts_to_delete/anna_labels_to_3d.py",
    "segmented-tiffs/iterative_dendrite_mask.tif",
    "segmented-tiffs/detected_dendrite_from_3d_frame122.tif",
    "segmented-tiffs/detected_dendrite_mask.tif",
    "segmented-tiffs/mip_dendrite_traces.csv",
    "segmented-tiffs/dff_traces.npy",
    "segmented-tiffs/trace_plots",
    "segmented-tiffs/multi_dendrites_dff.tif",
    "segmented-tiffs/dendrite_filtered_mask_4d.tif",
    "segmented-tiffs/dendrites_pca_active_frames.tif",
    "segmented-tiffs/labeled_dendrite_branches.tif",
    "segmented-tiffs/dendrite_labels_4d_projected.tif",
    "segmented-tiffs/detected_dendrite_single.tif",
    "segmented-tiffs/mip_dendrite_labels.tif",
    "segmented-tiffs/masked_dendrite_stack.tif",
    "segmented-tiffs/mip_dendrites_merged_4d.tif",
    "segmented-tiffs/grown_dendrite_mask_3d.tif",
    "segmented-tiffs/dendrite_mask_from_pca.tif",
    "segmented-tiffs/grown_dendrite_mask_single12.tif",
    "segmented-tiffs/dendrites_pca_components.tif",
    "segmented-tiffs/dendrite_trace.csv",
    "segmented-tiffs/grown_dendrite_from_seed.tif",
    "segmented-tiffs/dff_stack_for_analysis.tif",
    "segmented-tiffs/masked_dendrites_from_mip.tif",
    "segmented-tiffs/final_3d_dendrite_labels.tif",
    "segmented-tiffs/masked_dendrite_stack_deltaF_over_F.tif",
    "segmented-tiffs/mip_dendrite_labels_cleaned.tif",
    "segmented-tiffs/final_4d_dendrite_mask.tif",
    "segmented-tiffs/anna_3d_dendrites.tif",
    "segmented-tiffs/grown_dendrite_mask_single.tif",
    "figures/deltaF_F_one_dendrite.png",
    "figures/top-20-pca-mip.png",
    "dendrites-branches-tiffs/dendrites_branches.tif",
    "tiff-stacks/2025-04-22/runA_run6_rAi162_15_reslice_bin-1dendrite.tif",
    "tiff-stacks/2025-04-22/dendrite_masks_labeled.tif",
    "tiff-stacks/2025-04-22/runA_run6_rAi162_15_reslice_bin-3D.tif",
    "tiff-stacks/2025-04-22/avg_dendrite_event_volume_from_label.tif",
    "tiff-stacks/2025-04-22/dendrite_traces.csv"
]


for p in unused_paths:
    path = Path(p)
    if path.exists():
        if path.is_dir():
            print(f"Deleting folder: {p}")
            os.rmdir(path)  # or use shutil.rmtree(path) if not empty
        else:
            print(f"Deleting file: {p}")
            path.unlink()
