# 🧠 Apical Dendrite ΔF/F Analysis Pipeline

This repository contains a modular analysis pipeline for extracting and visualizing dendritic calcium activity from 4D imaging data. It supports frame selection, dendrite segmentation, functional branch clustering, ΔF/F computation, and high-quality visualization.

---

## 📁 Folder Structure

Data is organized by date, mouse, and run in the following structure:

```
data/
└── 2025-04-22/
    └── rAi162_15/
        └── run6/
            ├── raw/                    # Raw TIFFs from microscope
            │   ├── runA_run6_rAi162_15_processed_very_good.tif
            │   └── runA_run6_rAi162_15_reslice_bin.tif
            ├── events/                 # Selected mini-stacks (Module 1 output)
            ├── labelmaps/              # Dendrite masks (Module 2 output)
            ├── branches/               # Branch label volumes (Module 3 output)
            ├── traces/                 # ΔF/F trace data (.pkl)
            ├── overlays/               # ΔF/F volume with all branches
            └── trace_with_branchmap/   # Composite plots (Module 4 output)
```

---

## 🧩 Modular Pipeline Overview

| Module       | Script                         | Purpose                                                                                                                      |
| ------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Module 1** | `get_events_m1.py`     | Compute ΔF/F from 3D MIP, identify active frames interactively or manually, and extract event mini-stacks from raw 4D stack. |
| **Module 2** | `auto_mask_m2.py`   | Compute ΔF/F on mini-stacks, segment dendrites, cluster into branches using KMeans, and save labeled masks.                  |
| **Module 3** | `3d_mask_overlay_m3.py` | Overlay masks onto full raw 4D stack, compute ΔF/F per branch, generate 4D ΔF/F overlays and extract traces.                 |
| **Module 4** | `save_traces_m4.py`       | Create composite figures: 2D color-coded branch MIP + aligned ΔF/F traces for each dendrite.                                 |
| **Module 5** | `create_3d_movie_m5.py`    | Visualize full 4D ΔF/F stack in 3D with Napari using voxel scale, background filtering, and cube outline.                    |

All modules are configured using:

```python
DATE = "YYYY-MM-DD"
MOUSE = "mouse_id"
RUN = "runN"
```

These variables control all file paths and outputs, making the scripts reusable for any new imaging run.

---

## ⚙️ How to Run the Pipeline

1. Place your raw data in the folder:

   ```
   data/YYYY-MM-DD/MOUSE_ID/RUN_ID/raw/
   ```

   with filenames:

   * `runA_<run>_<mouse>_processed_very_good.tif` (3D MIP)
   * `runA_<run>_<mouse>_reslice_bin.tif` (4D raw stack)

2. Run each module in order:

   ```bash
   python code/module1_select_frames.py
   python code/module2_label_dendrites.py
   python code/module3_overlay_and_split.py
   python code/module4_plot_traces.py
   python code/module5_view_dff_clean.py
   ```

3. Review outputs in:

   * `events/`: selected high-activity timepoints
   * `labelmaps/`: segmented dendrites
   * `branches/`: functionally distinct subregions
   * `traces/`: extracted time series
   * `trace_with_branchmap/`: final figures
   * `overlays/`: ΔF/F branch visualization

---

## 📦 Environment Setup (Recommended)

Use a virtual environment to avoid dependency issues:

```bash
cd path/to/apical-dendrites-2025
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To create `requirements.txt` after installing all packages:

```bash
pip freeze > requirements.txt
```

Reactivate your environment any time with:

```bash
source .venv/bin/activate
```

---

## 📦 Dependencies

To install manually if not using `requirements.txt`:

```bash
pip install numpy scipy scikit-learn scikit-image tifffile matplotlib napari tqdm
pip install "napari[pyqt5]"
```

---

## ✅ Example Output

Each dendrite results in a figure combining:

* A **color-coded 2D MIP** showing functional branches
* **ΔF/F activity traces** for each branch, vertically offset for clarity
* An interactive 3D viewer to explore time and space dynamics

---

## 🧼 Cleanup Notes

To identify and delete unused files, use:

* `dry_run_reorganize.py` to preview file moves
* `reorganize_run6.py` to move files into correct structure
* `delete_files.py` to remove obsolete test data

---
