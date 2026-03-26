from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bcmonitor.data_loader import load_cwru_file
from bcmonitor.preprocessing import segment_signal


BASELINE_FEATURE_COLUMNS = [
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "skewness",
    "kurtosis",
    "dominant_frequency",
]

ENHANCED_EXTRA_FEATURE_COLUMNS = [
    "variance",
    "absolute_mean",
    "max_abs",
    "dominant_amplitude",
    "frequency_center",
    "spectral_entropy",
]

FEATURE_SETS = {
    "baseline": BASELINE_FEATURE_COLUMNS,
    "enhanced": BASELINE_FEATURE_COLUMNS + ENHANCED_EXTRA_FEATURE_COLUMNS,
}


def get_feature_columns(feature_set: str = "baseline") -> list[str]:
    feature_set = feature_set.strip().lower()

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. "
            f"Expected one of: {sorted(FEATURE_SETS.keys())}"
        )

    return FEATURE_SETS[feature_set].copy()


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


def variance(signal: np.ndarray) -> float:
    return float(np.var(signal))


def absolute_mean(signal: np.ndarray) -> float:
    return float(np.mean(np.abs(signal)))


def max_abs(signal: np.ndarray) -> float:
    return float(np.max(np.abs(signal)))


def _positive_frequency_spectrum(
    signal: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    signal_centered = signal - np.mean(signal)
    freqs = np.fft.rfftfreq(len(signal_centered), d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal_centered))

    if len(freqs) > 1:
        freqs = freqs[1:]
        spectrum = spectrum[1:]

    return freqs, spectrum


def dominant_amplitude(signal: np.ndarray, sample_rate: float) -> float:
    _, spectrum = _positive_frequency_spectrum(signal, sample_rate)
    if spectrum.size == 0:
        return 0.0
    return float(np.max(spectrum))


def frequency_center(signal: np.ndarray, sample_rate: float) -> float:
    freqs, spectrum = _positive_frequency_spectrum(signal, sample_rate)

    if spectrum.size == 0:
        return 0.0

    spectrum_sum = np.sum(spectrum)
    if spectrum_sum == 0.0:
        return 0.0

    return float(np.sum(freqs * spectrum) / spectrum_sum)


def spectral_entropy(signal: np.ndarray, sample_rate: float) -> float:
    _, spectrum = _positive_frequency_spectrum(signal, sample_rate)

    if spectrum.size == 0:
        return 0.0

    spectrum_sum = np.sum(spectrum)
    if spectrum_sum == 0.0:
        return 0.0

    probabilities = spectrum / spectrum_sum
    probabilities = probabilities[probabilities > 0.0]

    if probabilities.size == 0:
        return 0.0

    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy)


def extract_baseline_features(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
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


def extract_enhanced_features(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    baseline = extract_baseline_features(signal, sample_rate)

    enhanced = {
        "variance": variance(signal),
        "absolute_mean": absolute_mean(signal),
        "max_abs": max_abs(signal),
        "dominant_amplitude": dominant_amplitude(signal, sample_rate),
        "frequency_center": frequency_center(signal, sample_rate),
        "spectral_entropy": spectral_entropy(signal, sample_rate),
    }

    return {**baseline, **enhanced}


def extract_features(
    signal: np.ndarray,
    sample_rate: float,
    feature_set: str = "baseline",
) -> dict[str, float]:
    feature_set = feature_set.strip().lower()

    if feature_set == "baseline":
        return extract_baseline_features(signal, sample_rate)

    if feature_set == "enhanced":
        return extract_enhanced_features(signal, sample_rate)

    raise ValueError(
        f"Unknown feature_set '{feature_set}'. "
        f"Expected one of: {sorted(FEATURE_SETS.keys())}"
    )


def _normalise_file_spec(spec: Any) -> dict[str, Any]:
    if is_dataclass(spec):
        spec = asdict(spec)
    elif not isinstance(spec, dict):
        raise TypeError("Each file spec must be a dict or dataclass instance.")

    required_keys = {"file_name", "label"}
    missing = required_keys.difference(spec.keys())
    if missing:
        raise KeyError(f"File spec is missing required keys: {sorted(missing)}")

    return dict(spec)


def extract_feature_table_from_signal(
    signal: np.ndarray,
    sample_rate: float,
    label: str,
    source_file: str,
    window_size: int = 2048,
    step_size: int = 1024,
    metadata: dict[str, Any] | None = None,
    feature_set: str = "baseline",
) -> pd.DataFrame:
    windows = segment_signal(signal, window_size=window_size, step_size=step_size)
    metadata = {} if metadata is None else dict(metadata)

    rows = []
    for i, window in enumerate(windows):
        row = {
            "label": label,
            "source_file": source_file,
            "window_id": i,
            **metadata,
            **extract_features(window, sample_rate, feature_set=feature_set),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_table(
    file_specs: list[Any],
    sample_rate: float = 12000.0,
    window_size: int = 2048,
    step_size: int = 1024,
    feature_set: str = "baseline",
) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    tables = []

    for raw_spec in file_specs:
        spec = _normalise_file_spec(raw_spec)
        file_name = spec.pop("file_name")
        label = spec.pop("label")
        file_sample_rate = float(spec.pop("sample_rate_hz", sample_rate))

        bearing_signal = load_cwru_file(raw_dir / file_name, sample_rate=file_sample_rate)

        table = extract_feature_table_from_signal(
            signal=bearing_signal.signal,
            sample_rate=file_sample_rate,
            label=label,
            source_file=file_name,
            window_size=window_size,
            step_size=step_size,
            metadata=spec,
            feature_set=feature_set,
        )
        tables.append(table)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)