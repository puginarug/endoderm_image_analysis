# Endoderm Cell Migration Image Analysis Pipeline

A comprehensive Python-based pipeline for analyzing endoderm cell migration from live-cell microscopy data. This workflow performs cell segmentation, tracking, and quantitative analysis of migration metrics.

## Overview

This pipeline processes multi-dimensional time-lapse microscopy data (ND2 format) to extract quantitative metrics of cell migration, including velocity, directionality, mean squared displacement (MSD), and directional autocorrelation (DACF).

## Pipeline Workflow

The analysis consists of the following sequential steps:

### 1. Image Preprocessing (`1. nd2_reader_max_projection.ipynb`)
- Loads ND2 microscopy files containing multi-channel, multi-timepoint, Z-stack data
- Creates maximum intensity projections along the Z-axis for each timepoint
- Performs contrast enhancement (CLAHE) for better visualization
- Exports processed images as multi-page TIF files

### 2. Cell Segmentation (`2. cellpose_segmentation_for_btrack_BATCH.ipynb`)
- Uses Cellpose deep learning model for cell instance segmentation
- Batch processes multiple movies
- Generates segmentation masks compatible with btrack tracking software

### 3. Cell Tracking (`3. btrack_BATCH.ipynb`)
- Implements Bayesian tracking (btrack) to link segmented cells across time
- Configuration file: `3.1 cell_config_david.json`
- Batch processes multiple experiments
- Generates cell trajectories from segmentation masks

### 4. Region of Interest Selection (`4. mask_fitting_GUI.ipynb`)
- Interactive GUI for defining regions of interest (ROI)
- Fits geometric masks (e.g., circles, ellipses) to experimental boundaries
- Parameter files:
  - `4.1 masks_parameters_2024_07_21_FOXA2_1to100_TdTom_c21_day2To3.json`
  - `4.2 masks_parameters_2024_09_09_FoxA2_tdTom_c211to300_BmpAct_day2To3.json`

### 5. Track DataFrame Creation (`5. tracks_df_creation_from_folder_*.ipynb`)
Multiple notebooks for different experimental setups:
- **2D experiments**: `5. tracks_df_creation_from_folder_2d_experiments.ipynb`
- **2D manual tracks**: `5. tracks_df_creation_from_folder_2d_experiments_manual_tracks.ipynb`
- **3D experiments**: `5. tracks_df_creation_from_folder_3d_experiments.ipynb`
- **3D manual tracks**: `5. tracks_df_creation_from_folder_3d_experiments_manual_tracks.ipynb`

Consolidates tracking data with spatial coordinates, velocities, and mask information.

### 6. Migration Statistics (`6. migration_statistics.ipynb`)
Comprehensive analysis of cell migration metrics:
- Velocity distributions
- Mean squared displacement (MSD)
- Directional autocorrelation function (DACF)
- Turning angle distributions
- Step length distributions
- Trajectory visualizations

### 7. Density Dependence Analysis (`7. dencity_dependence_2d.ipynb`)
Analyzes how cell migration behavior depends on local cell density.

### 8. Particle Image Velocimetry Analysis (`8. PIV_results_analysis_BATCH.ipynb`)
Batch analysis of PIV (Particle Image Velocimetry) results to quantify collective migration patterns and flow fields.

## Key Features

- **Automated batch processing** for multiple experiments
- **Robust migration metrics** including MSD, DACF, velocity, and directionality
- **Support for both 2D and 3D microscopy data**
- **Manual and automated tracking workflows**
- **Interactive ROI selection** for complex geometries
- **Statistical analysis** with weighted aggregation across movies
- **Publication-ready visualizations**

## Requirements

### Python Dependencies

```
numpy
pandas
matplotlib
opencv-python (cv2)
scipy
nd2reader
tifffile
cellpose
btrack
jupyter
```

### Optional Tools
- ImageJ/Fiji (for viewing multi-page TIF files)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/puginarug/endoderm_image_analysis.git
cd endoderm_image_analysis
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib opencv-python scipy nd2reader tifffile cellpose btrack jupyter
```

3. Launch Jupyter:
```bash
jupyter notebook
```

## Usage

### Quick Start

1. **Prepare your data**: Place ND2 files in a dedicated directory
2. **Run notebooks sequentially** (1 → 2 → 3 → 4 → 5 → 6)
3. **Adjust configuration parameters** as needed for your experiment
4. **Extract migration metrics** from notebook 6

### Example Workflow

```python
# Example: Computing migration metrics from trajectory data
from migration_metrics_functions import compute_msd_dacf_per_movie

# Load your tracks dataframe
# df should contain columns: 'track_id', 'x_microns', 'y_microns', 't', 'step', 'file'
msd_summary, dacf_summary = compute_msd_dacf_per_movie(df, max_lag=50)
```

## Migration Metrics Module

The `migration_metrics_functions.py` module provides functions for computing:

- **Turning angles**: `compute_turning_angles()`
- **Mean squared displacement (MSD)**: `calculate_msd()`
- **Velocity autocorrelation (VACF/DACF)**: `calculate_autocorrelation()`
- **Step length distributions**: `compute_step_length_distribution()`
- **Weighted aggregation across movies**: `compute_msd_dacf_per_movie()`

See function docstrings for detailed parameter descriptions.

## Data Structure

### Expected Input
- **Format**: ND2 (Nikon microscopy format)
- **Dimensions**: Time (T) × Z-stack (Z) × Channels (C) × Y × X
- **Channels**:
  - Channel 0: Fluorescence (used for segmentation/tracking)
  - Channel 1: Brightfield (optional reference)

### Output Files
- **Max projections**: Multi-page TIF files (one page per timepoint)
- **Segmentation masks**: Compatible with btrack format
- **Track dataframes**: Pandas DataFrames with trajectory information
- **Analysis results**: CSV files and publication-ready plots

## Configuration Files

- `3.1 cell_config_david.json`: btrack configuration for tracking parameters
- `4.1 masks_parameters_*.json`: ROI mask parameters for specific experiments

## Citation

If you use this pipeline in your research, please cite:

[Add your publication reference here]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## License

[Add license information]

## Contact

For questions or support, please open an issue on GitHub or contact [your contact information].

## Acknowledgments

This pipeline uses the following open-source tools:
- [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation
- [btrack](https://github.com/quantumjot/btrack) for Bayesian cell tracking
- [nd2reader](https://github.com/rbnvrw/nd2reader) for reading Nikon ND2 files
