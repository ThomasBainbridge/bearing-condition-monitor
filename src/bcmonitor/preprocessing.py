from __future__ import annotations

import numpy as np


def segment_signal(
    signal: np.ndarray,
    window_size: int = 2048,
    step_size: int = 1024,
) -> np.ndarray:
    """
    Split a 1D signal into overlapping windows.

    Returns
    -------
    np.ndarray
        Array of shape (n_windows, window_size)
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D.")

    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive.")

    if len(signal) < window_size:
        raise ValueError("Signal is shorter than window_size.")

    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal[start:end])

    return np.asarray(windows, dtype=float)


def zscore_normalise(signal: np.ndarray) -> np.ndarray:
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val == 0.0:
        return signal - mean_val
    return (signal - mean_val) / std_val