from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bcmonitor.data_loader import load_cwru_file
from bcmonitor.preprocessing import segment_signal


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def peak_to_peak(signal: np.ndarray) -> float:
    return float(np.max(signal) - np.min(signal))


def crest_factor(signal: np.ndarray) -> float:
    signal_rms = rms(signal)
    if signal_rms == 0.0:
        return 0.0
    return float(np.max(np.abs(signal)) / signal_rms)


def shape_factor(signal: np.ndarray) -> float:
    mean_abs = np.mean(np.abs(signal))
    if mean_abs == 0.0:
        return 0.0
    return float(rms(signal) / mean_abs)


def impulse_factor(signal: np.ndarray) -> float:
    mean_abs = np.mean(np.abs(signal))
    if mean_abs == 0.0:
        return 0.0
    return float(np.max(np.abs(signal)) / mean_abs)


def clearance_factor(signal: np.ndarray) -> float:
    denominator = np.mean(np.sqrt(np.abs(signal))) ** 2
    if denominator == 0.0:
        return 0.0
    return float(np.max(np.abs(signal)) / denominator)


def skewness(signal: np.ndarray) -> float:
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val == 0.0:
        return 0.0
    return float(np.mean(((signal - mean_val) / std_val) ** 3))


def kurtosis(signal: np.ndarray) -> float:
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val == 0.0:
        return 0.0
    return float(np.mean(((signal - mean_val) / std_val) ** 4))


def dominant_frequency(signal: np.ndarray, sample_rate: float) -> float:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    idx = int(np.argmax(spectrum))
    return float(freqs[idx])


def extract_basic_features(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": rms(signal),
        "peak_to_peak": peak_to_peak(signal),
        "crest_factor": crest_factor(signal),
        "shape_factor": shape_factor(signal),
        "impulse_factor": impulse_factor(signal),
        "clearance_factor": clearance_factor(signal),
        "skewness": skewness(signal),
        "kurtosis": kurtosis(signal),
        "dominant_frequency": dominant_frequency(signal, sample_rate),
    }


def extract_feature_table_from_signal(
    signal: np.ndarray,
    sample_rate: float,
    label: str,
    source_file: str,
    window_size: int = 2048,
    step_size: int = 1024,
) -> pd.DataFrame:
    windows = segment_signal(signal, window_size=window_size, step_size=step_size)

    rows = []
    for i, window in enumerate(windows):
        row = {
            "label": label,
            "source_file": source_file,
            "window_id": i,
            **extract_basic_features(window, sample_rate),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_table(
    file_specs: list[dict],
    sample_rate: float = 12000.0,
    window_size: int = 2048,
    step_size: int = 1024,
) -> pd.DataFrame:
    """
    Build a feature table from multiple raw .mat files.

    file_specs example:
    [
        {"file_name": "normal_0.mat", "label": "normal"},
        {"file_name": "ir007_0.mat", "label": "inner_race"},
    ]
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    tables = []

    for spec in file_specs:
        file_name = spec["file_name"]
        label = spec["label"]

        bearing_signal = load_cwru_file(raw_dir / file_name, sample_rate=sample_rate)

        table = extract_feature_table_from_signal(
            signal=bearing_signal.signal,
            sample_rate=sample_rate,
            label=label,
            source_file=file_name,
            window_size=window_size,
            step_size=step_size,
        )
        tables.append(table)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)