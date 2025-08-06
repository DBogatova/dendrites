# Apical Dendrites Analysis Pipeline

This README provides step-by-step instructions for running the analysis pipeline for the organoid mouse run6 dataset.

## Setup Environment

First, set up a Python environment with all required dependencies:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib scipy scikit-image tifffile napari[all] tqdm imageio seaborn
```

## Analysis Pipeline

The analysis consists of 5 main scripts that should be run in sequence:

### 1. Initial Processing (find_events_m1.py)

This script performs initial processing of the raw data:

```bash
python code/find_events_m1.py
```

This will:
- Load the raw stack
- Calculate F0 baseline
- Generate initial visualizations

### 2. Automatic Mask Generation (auto_mask_m2.py)

This script automatically generates masks for dendrites:

```bash
python code/auto_mask_m2.py
```

This will:
- Process the ΔF/F stack
- Apply thresholding to identify active regions
- Generate initial masks in the `labelmaps` folder

### 3. Filter Selected Masks (filter_selected_masks_m3.py)

This script allows you to filter and refine the automatically generated masks:

```bash
python code/filter_selected_masks_m3.py
```

This will:
- Load the auto-generated masks
- Apply filtering criteria
- Save the filtered masks to the `labelmaps_curated_dynamic` folder

### 4. Analyze Traces (analyze_traces_m4.py)

This script analyzes the calcium traces from the selected masks:

```bash
python code/analyze_traces_m4.py
```

This will:
- Extract calcium traces from each mask
- Identify duplicate traces
- Generate trace classification metrics
- Save results to `trace_classifications.csv`

### 5. Save Traces and Generate Visualizations (save_traces_m5.py)

This script generates final visualizations and saves the processed traces:

```bash
python code/save_traces_m5.py
```

This will:
- Generate individual trace plots with MIPs
- Create a combined plot of all traces
- Save the traces to CSV and pickle formats

### 6. Create 3D Movie (create_3d_movie_m5.py) - Optional

This script creates a 3D visualization movie of selected cells:

```bash
python code/create_3d_movie_m6.py
```

This will:
- Create a 3D visualization of selected cells
- Generate a rotating MP4 video showing the cells from different angles

## Data Structure

The scripts expect the following data structure:

```
data/
└── 2025-03-26/
    └── organoid/
        └── run6/
            ├── raw/                          # Raw data files
            │   └── runA_run6_organoid_reslice_bin.tif
            ├── labelmaps/                    # Auto-generated masks
            ├── labelmaps_curated_dynamic/    # Manually curated masks
            ├── traces/                       # Extracted traces
            ├── trace_previews_curated/       # Trace visualizations
            └── overlays/                     # 3D visualizations
```

## Troubleshooting

- If you encounter memory errors, try processing smaller chunks of data
- For visualization issues in Napari, make sure you have the latest version installed
- If scripts fail to find files, check that the paths in the scripts match your data structure

## Notes

- The scripts are configured for the specific dataset (run6, organoid mouse)
- To analyze a different dataset, you'll need to modify the DATE, MOUSE, and RUN variables in each script
- Some scripts have hardcoded parameters that may need adjustment for different datasets

For any questions or issues ask Daria