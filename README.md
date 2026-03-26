# Bearing Condition Monitor

A vibration-based machine learning and signal-processing project for bearing fault diagnosis using the Case Western Reserve University (CWRU) bearing dataset.

## Overview

This project builds an engineering-focused bearing fault diagnosis pipeline from raw vibration signals through to classification results. The workflow covers MATLAB data ingestion, signal inspection, time-domain and frequency-domain analysis, engineered diagnostic feature extraction, dataset balancing, model training, and validation across unseen operating loads.

This repository currently documents **Version 1** of the project.

## Version 1 Summary

Version 1 focuses on building a technically credible end-to-end fault diagnosis pipeline using engineered vibration features and classical machine learning.

Version 1 includes:

- raw `.mat` data loading from the CWRU dataset
- drive-end vibration signal extraction
- raw signal, FFT, and spectrogram visualisation
- fixed-window signal segmentation
- engineered feature extraction
- balanced feature-table generation
- train/test splitting by operating load
- baseline classification with Logistic Regression and Random Forest
- leave-one-load-out validation
- confusion matrices, feature-importance analysis, and saved trained models

## Version 1 Objective

The Version 1 objective is to identify bearing condition from vibration data using engineered diagnostic features and machine learning.

The classification task uses four health states:

- Normal
- Inner race fault
- Ball fault
- Outer race fault

Version 1 focuses on a controlled subset of the CWRU dataset using:

- 12 kHz drive-end vibration data
- 0.007 inch fault diameter for faulty classes
- four operating loads: 0, 1, 2, and 3 HP

## Dataset

Version 1 uses a controlled 16-file subset of the CWRU bearing vibration dataset.

Files included:

- 4 normal files
- 4 inner race fault files
- 4 ball fault files
- 4 outer race fault files

Source files:

- `normal_0.mat`
- `normal_1.mat`
- `normal_2.mat`
- `normal_3.mat`
- `ir007_0.mat`
- `ir007_1.mat`
- `ir007_2.mat`
- `ir007_3.mat`
- `b007_0.mat`
- `b007_1.mat`
- `b007_2.mat`
- `b007_3.mat`
- `or007_6_0.mat`
- `or007_6_1.mat`
- `or007_6_2.mat`
- `or007_6_3.mat`

Each signal is segmented into overlapping windows before feature extraction.

## Version 1 Workflow

The Version 1 pipeline is structured as follows:

1. Load raw `.mat` vibration files
2. Extract drive-end time-series data and RPM metadata
3. Visualise raw time histories, FFTs, and spectrograms
4. Segment signals into fixed windows
5. Extract engineered diagnostic features from each window
6. Build a labelled feature dataset
7. Balance the dataset so each source file contributes equally
8. Split the data by operating load
9. Train machine learning classifiers
10. Evaluate performance on unseen operating loads

## Engineered Features

The Version 1 feature set includes:

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

These were selected because they are standard vibration diagnostics features and have clear engineering interpretation.

Version 1 results can be seen in docs/version1_results.md

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
├── app/
│   └── streamlit_app.py
│
└── tests/