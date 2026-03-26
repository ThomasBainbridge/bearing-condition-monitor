from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


@dataclass
class BearingSignal:
    file_path: Path
    signal: np.ndarray
    sample_rate: float | None
    rpm: float | None
    signal_key: str
    rpm_key: str | None


def _flatten_numeric_array(value: Any) -> np.ndarray | None:
    """Return a flattened numeric NumPy array if possible, otherwise None."""
    if not isinstance(value, np.ndarray):
        return None

    if value.size == 0:
        return None

    if not np.issubdtype(value.dtype, np.number):
        return None

    return np.asarray(value).squeeze()


def _find_signal_key(mat_dict: dict[str, Any]) -> str:
    """
    Find the drive-end time-series key in a CWRU .mat file.

    Typical examples:
    X097_DE_time
    X105_DE_time
    """
    candidate_keys = []

    for key in mat_dict.keys():
        if key.startswith("__"):
            continue
        if "DE" in key and "time" in key:
            candidate_keys.append(key)

    if not candidate_keys:
        raise KeyError("Could not find a drive-end time-series key containing 'DE' and 'time'.")

    # Prefer the shortest valid key in case there are several
    candidate_keys.sort(key=len)
    return candidate_keys[0]


def _find_rpm_key(mat_dict: dict[str, Any]) -> str | None:
    """Find the RPM key if present."""
    for key in mat_dict.keys():
        if key.startswith("__"):
            continue
        if "RPM" in key.upper():
            return key
    return None


def load_cwru_file(file_path: str | Path, sample_rate: float | None = None) -> BearingSignal:
    """
    Load a single CWRU bearing .mat file.

    Parameters
    ----------
    file_path
        Path to the .mat file.
    sample_rate
        Optional sampling rate to assign manually. Use this if you know the file
        belongs to the 12 kHz dataset and want to store that explicitly.

    Returns
    -------
    BearingSignal
        Dataclass containing the signal, metadata, and detected keys.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    mat_dict = loadmat(file_path)

    signal_key = _find_signal_key(mat_dict)
    signal_array = _flatten_numeric_array(mat_dict[signal_key])

    if signal_array is None:
        raise ValueError(f"Signal under key '{signal_key}' could not be converted to a numeric array.")

    rpm_key = _find_rpm_key(mat_dict)
    rpm = None

    if rpm_key is not None:
        rpm_array = _flatten_numeric_array(mat_dict[rpm_key])
        if rpm_array is not None and rpm_array.size > 0:
            rpm = float(rpm_array.flat[0])

    return BearingSignal(
        file_path=file_path,
        signal=signal_array.astype(float),
        sample_rate=sample_rate,
        rpm=rpm,
        signal_key=signal_key,
        rpm_key=rpm_key,
    )


def load_raw_sample(file_name: str, sample_rate: float | None = 12000.0) -> BearingSignal:
    """
    Convenience loader for files inside data/raw/.
    """
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / file_name
    return load_cwru_file(file_path=file_path, sample_rate=sample_rate)


def describe_signal(bearing_signal: BearingSignal) -> dict[str, Any]:
    """
    Return a simple summary dictionary for quick inspection.
    """
    signal = bearing_signal.signal

    return {
        "file_name": bearing_signal.file_path.name,
        "n_samples": int(signal.size),
        "sample_rate_hz": bearing_signal.sample_rate,
        "rpm": bearing_signal.rpm,
        "signal_key": bearing_signal.signal_key,
        "rpm_key": bearing_signal.rpm_key,
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
    }