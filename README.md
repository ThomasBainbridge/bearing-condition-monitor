# Bearing Condition Monitor

A vibration-based machine learning and signal-processing project for bearing fault diagnosis using the Case Western Reserve University (CWRU) bearing dataset.

## Overview

This project builds an engineering-focused bearing fault diagnosis pipeline from raw vibration signals through to classification results. The workflow covers MATLAB data ingestion, signal inspection, time-domain and frequency-domain analysis, engineered diagnostic feature extraction, dataset balancing, model training, and validation across unseen operating loads.

The repository now documents **Version 2** of the project.

## Version 2 Summary

Version 2 extends the original Version 1 pipeline into a more realistic and more demanding bearing fault diagnosis study.

Version 2 includes:

- raw `.mat` data loading from the CWRU dataset
- drive-end vibration signal extraction
- raw signal, FFT, and spectrogram visualisation
- fixed-window signal segmentation
- structured dataset definition through an internal dataset registry
- engineered feature extraction using both baseline and enhanced feature sets
- balanced feature-table generation
- train/test splitting by operating load
- model comparison using Logistic Regression, Random Forest, and XGBoost
- held-out load testing
- leave-one-load-out validation
- saved processed datasets and comparison tables

## Version 2 Objective

The Version 2 objective is to identify bearing condition from vibration data using engineered diagnostic features and machine learning, while making the diagnosis problem harder and more realistic than in Version 1.

The classification task uses four health states:

- Normal
- Inner race fault
- Ball fault
- Outer race fault

Version 2 uses a controlled but expanded subset of the CWRU dataset with:

- 12 kHz drive-end vibration data
- four operating loads: 0, 1, 2, and 3 HP
- three fault sizes for faulty classes:
  - 0.007 inch
  - 0.014 inch
  - 0.021 inch
- outer race faults fixed at the 6:00 position

## Dataset

Version 2 uses a controlled 40-file subset of the CWRU bearing vibration dataset.

Files included:

- 4 normal files
- 12 inner race fault files
- 12 ball fault files
- 12 outer race fault files

### Normal files

- `normal_0.mat`
- `normal_1.mat`
- `normal_2.mat`
- `normal_3.mat`

### Inner race fault files

- `ir007_0.mat`
- `ir007_1.mat`
- `ir007_2.mat`
- `ir007_3.mat`
- `ir014_0.mat`
- `ir014_1.mat`
- `ir014_2.mat`
- `ir014_3.mat`
- `ir021_0.mat`
- `ir021_1.mat`
- `ir021_2.mat`
- `ir021_3.mat`

### Ball fault files

- `b007_0.mat`
- `b007_1.mat`
- `b007_2.mat`
- `b007_3.mat`
- `b014_0.mat`
- `b014_1.mat`
- `b014_2.mat`
- `b014_3.mat`
- `b021_0.mat`
- `b021_1.mat`
- `b021_2.mat`
- `b021_3.mat`

### Outer race fault files

- `or007_6_0.mat`
- `or007_6_1.mat`
- `or007_6_2.mat`
- `or007_6_3.mat`
- `or014_6_0.mat`
- `or014_6_1.mat`
- `or014_6_2.mat`
- `or014_6_3.mat`
- `or021_6_0.mat`
- `or021_6_1.mat`
- `or021_6_2.mat`
- `or021_6_3.mat`

Each signal is segmented into overlapping windows before feature extraction.

## Version 2 Workflow

The Version 2 pipeline is structured as follows:

1. Load raw `.mat` vibration files
2. Extract drive-end time-series data and RPM metadata
3. Visualise raw time histories, FFTs, and spectrograms
4. Segment signals into fixed windows
5. Extract engineered diagnostic features from each window
6. Build labelled feature tables from the dataset registry
7. Balance the dataset so each source file contributes equally
8. Split the data by operating load
9. Train machine learning classifiers
10. Compare models across baseline and enhanced feature sets
11. Evaluate performance on unseen operating loads
12. Perform leave-one-load-out validation

## Engineered Features

Version 2 compares two feature sets.

### Baseline Feature Set

The baseline feature set retains the original Version 1-style diagnostic features:

- Mean
- Standard deviation
- RMS
- Peak-to-peak amplitude
- Crest factor
- Shape factor
- Impulse factor
- Clearance factor
- Skewness
- Kurtosis
- Dominant frequency

### Enhanced Feature Set

The enhanced Version 2 feature set includes all baseline features plus:

- Variance
- Absolute mean
- Maximum absolute amplitude
- Dominant spectral amplitude
- Frequency centre
- Spectral entropy

This allows Version 2 to compare the original simpler engineered feature space against a richer feature representation.

## Models Evaluated

Version 2 evaluates three classifiers:

- Logistic Regression
- Random Forest
- XGBoost

Logistic Regression is used as a simple linear baseline.

Random Forest is used as a strong nonlinear ensemble model.

XGBoost is used as a gradient-boosted tree model for stronger comparison on the expanded dataset.

## Key Version 2 Results

Version 2 produced several important results:

- the enhanced feature set significantly improved performance compared with the baseline feature set
- the largest improvement from enhanced features was seen in Logistic Regression, showing that the richer signal representation made the class structure much more separable
- XGBoost achieved the best mean performance across leave-one-load-out validation
- Random Forest remained extremely competitive and showed slightly stronger worst-case robustness across held-out loads
- load 0 was the most difficult generalisation case, showing that operating-condition shift still matters even with richer features

Overall, Version 2 shows that feature engineering had the biggest effect on performance, while XGBoost and Random Forest were the strongest classifiers on the expanded multi-fault-size problem.

## Results Documentation

Detailed results are documented here:

- `Results/version1_results.md`
- `Results/version2_results.md`

## Project Structure

```text
bearing-condition-monitor/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│   └── bcmonitor/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── datasets.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── plotting.py
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
│
├── models/
├── reports/
│   └── figures/
│
├── Results/
│   ├── version1_results.md
│   └── version2_results.md
│
├── app/
│   └── streamlit_app.py
│
└── tests/