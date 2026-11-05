# Age-Related Differences in Regional Activity and Functional Connectivity During Naturalistic Social Cognition

This repository contains analysis code for examining age-related differences in regional brain activity and dynamic functional connectivity associated with continuous judgments of perceived awkwardness during naturalistic movie-watching.

The study examines how young and older adults engage distributed cortical networks during dynamic social interpretation. Regional activity and functional connectivity are modeled simultaneously to estimate their unique associations with moment-to-moment changes in perceived awkwardness.

## Study Overview

The analytic sample consisted of 184 participants:

- **Young adults:** N = 102 (ages 18–35)
- **Older adults:** N = 82 (ages 61–92)

Participants watched two episodes of the mockumentary-style television program *Nathan for You* (Episodes: "Petting Zoo" and "Souvenir Shop") during fMRI without explicit instructions to make social judgments. Following scanning, participants watched the same episodes again and provided continuous joystick ratings of perceived awkwardness.

The primary analyses examined associations between awkwardness and:

1. **Regional activity (nodal activity):** standardized BOLD activity within individual brain parcels.
2. **Functional connectivity (edge time series):** moment-to-moment co-fluctuation between pairs of brain parcels.

Both features were included in the same participant-level regression model.

## Analysis Overview

The analysis consists of two primary stages.

### 1. Participant-Level Modeling

For each participant and each unique pair of nodes (*i*, *j*), an ordinary least squares model was fitted to estimate the unique association of nodal activity and interregional co-fluctuation with perceived awkwardness.

For each pair of nodes:

- The BOLD time series for each node were standardized across the concatenated movie runs.
- An edge time series was calculated as the element-wise product of the two standardized nodal signals:

  `Edge(i,j) = z(i) × z(j)`

- The HRF-convolved awkwardness rating was modeled as a function of activity in node *i*, activity in node *j*, their edge time series, and a run indicator.
- An optional additional nuisance regressor could be included.

To account for temporal autocorrelation, participant-level models underwent autoregressive prewhitening. An AR(25) model was estimated from residuals sampled across 100 randomly selected edges. The resulting whitening filter was applied to the behavioral regressor, nodal activity terms, edge time series, and run indicator. The first and last 25 time points were omitted following filtering.

Participant-level outputs include coefficient estimates and associated p-values for both nodal activity terms and the edge term, along with model fit statistics and estimated autoregressive coefficients.

### 2. Group-Level Modeling

Participant-level coefficients were combined for feature-wise group analyses.

The group-level analyses examined:

- **Overall awkwardness effects:** whether regional activity or functional connectivity was associated with awkwardness across age groups.
- **Age differences:** whether the association between awkwardness and regional activity or functional connectivity differed between young and older adults.

Age group was effect coded as:

- Young adults = +0.5
- Older adults = −0.5

The model intercept represents the overall effect across the two age groups, whereas the age coefficient represents the young-adult versus older-adult difference.

Group-level models used heteroskedasticity-consistent HC3 standard errors. P-values were corrected across features using false discovery rate (FDR) correction (*q* < 0.05).

Spatial spin permutation testing (10,000 permutations) was used to evaluate network-level significance for nodal and within-/between-network connectivity effects.


### `level1/`: Participant-Level Analysis

Contains the participant-level edge-time-series regression and SLURM submission scripts.

`ets_sub_linreg_prewhiten.py` performs the simultaneous nodal activity and edge time-series regression, including autoregressive prewhitening.

`indices.npy` contains the node-pair indexing used to construct edge time series.

### `level2/`: Group-Level Analysis and Visualization

Contains the scripts for loading participant-level results, constructing nodal and edge matrices, fitting group-level models, performing statistical correction and spatial permutation tests, and generating visualizations.

- `mat_manip.py`: Matrix reorganization, network indexing, spatial spin tests, and associated utilities.
- `termpull.py`: Group-level model organization, feature-wise testing, matrix construction, and network-level permutation testing.
- `output_driver_refactor.py`: Group-level analysis driver and generation of figures, including cortical surface maps, connectivity matrices, boxplots, and connectome visualizations.
- `plotting_funcs.py`: Plotting utilities for functional connectivity matrices, cortical surface maps, and network borders.

### `subject_lists/`: Participant Lists

Contains the overall analytic sample and separate young- and older-adult participant lists used for group-level analyses.

## Required Inputs

Neuroimaging and behavioral data are not included in this repository.

The participant-level analysis expects the following parcellated time series for each participant:

```text
<data_directory>/
└── <participant_id>/
    ├── <participant_id>_pettingzoo_interpolated_ts_custom_parc.npy
    └── <participant_id>_souvenirshop_interpolated_ts_custom_parc.npy
```

The arrays should contain time points × nodes. The current script truncates the two runs to 233 and 249 time points, respectively.

### Behavioral Regressors

HRF-convolved behavioral regressors are expected under:

```text
regs/
└── hrf/
    ├── awk_hrf.npy
    └── <participant_id>_awk_hrf.npy
```

The first file contains the independent consensus awkwardness rating. Participant-specific ratings are stored using the participant identifier in the filename.

The participant-level script accepts either `consensus` or `individual` as the behavioral-rating specification.

### Optional Resting-State Data

The resting-state sensitivity analysis uses precomputed resting-state functional connectivity matrices from XCP-D-denoised resting-state data.

The group-level driver expects participant-specific Pearson correlation matrices in tab-separated format and transforms these values using Fisher's *r*-to-*z* transformation before including them as covariates.

## Software Requirements

The code is implemented in Python and uses the following packages:

- NumPy
- SciPy
- statsmodels
- pandas
- Matplotlib
- seaborn
- Nilearn

Participant-level scripts additionally use Python's standard-library modules for file handling, serialization, and timing.

## virtual environment setup

# Conda
conda create -n awkwardness_ets python=3.11
conda activate awkwardness_ets
python -m pip install -r requirements.txt

# Standard Python venv
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# uv
uv venv
uv pip install -r requirements.txt

The included SLURM scripts were written for the Indiana University computing environment and assume an existing Conda environment named `awkwardness_ets`.

## Running the Analysis

### `level1/`: Participant-Level Analysis

The intended command-line interface for a single participant using the participant-level script is:

```bash
cd level1

python ets_sub_linreg_prewhiten.py \
    <data_directory> \
    <participant_id> \
    <behavior_label> \
    <consensus or_individual> \
    [additional_regressor_path]
```

For example, `behavior_label` may be `awk`, with the rating specification set to either `consensus` or `individual`.

The script currently loads `indices.npy` relative to the working directory, so it should be run from `level1/` unless that path is modified.


### `level2/`: Group-Level Analysis and Visualization

All group-level statistical analyses and primary figure generation reported in the manuscript were performed using `run_group_analysis.py`, following completion of the participant-level analyses in `level1/`.

**`run_group_analysis.py` is the main analysis script for the manuscript.** It loads participant-level model estimates, performs feature-wise group analyses of overall awkwardness associations and age differences, applies multiple-comparison correction and network-level spatial permutation testing, and generates the corresponding statistical visualizations.

The script also supports analyses controlling for resting-state functional connectivity. Analysis settings and input paths are configured directly within the script rather than through a command-line interface.

The remaining Python files provide supporting functionality:

- `termpull.py` — Loads participant-level estimates and organizes nodal and edge results for group-level analysis.
- `mat_manip.py` — Matrix manipulation, network organization, and spatial permutation testing.
- `plotting_funcs.py` — Visualization of cortical activity, connectivity matrices, and network-level results.

## Outputs

Participant-level regression outputs are saved as serialized Python pickle files containing:

- Nodal activity coefficient estimates
- Edge time-series coefficient estimates
- Associated p-values
- Model F-statistics and p-values
- R² and adjusted R²
- Estimated autoregressive coefficients

Group-level analysis generates:

- Feature-wise statistical results
- FDR-thresholded nodal and connectivity matrices
- Network-level spin permutation results
- Cortical surface projections
- Within- and between-network connectivity matrices
- Participant-level coefficient distributions and boxplots
- Connectome visualizations of significant edges

Results and figures are organized into `data/` and `figs/` directories generated during analysis.


## Citation

If you use this code, please cite the associated manuscript:

[Full manuscript citation to be added upon publication.]

The methods build on existing work on edge time series and dynamic functional connectivity, see French et al., 2026, https://doi.org/10.1162/NETN.a.571 and Merritt et al., 2026, https://doi.org/10.1016/j.isci.2026.115949 for similar applications.
