from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


def get_figures_dir(project_root: Path) -> Path:
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def compute_fft(signal: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal)) / n
    return freqs, spectrum


def plot_raw_comparison(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    output_path: str | Path | None = None,
    n_samples: int = 3000,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(signal_a[:n_samples], label=label_a)
    plt.plot(signal_b[:n_samples], label=label_b, alpha=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_fft_comparison(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    sample_rate: float,
    label_a: str,
    label_b: str,
    title: str,
    output_path: str | Path | None = None,
    max_frequency: float = 2000.0,
) -> None:
    freqs_a, fft_a = compute_fft(signal_a, sample_rate)
    freqs_b, fft_b = compute_fft(signal_b, sample_rate)

    plt.figure(figsize=(12, 4))
    plt.plot(freqs_a, fft_a, label=label_a)
    plt.plot(freqs_b, fft_b, label=label_b, alpha=0.8)
    plt.xlim(0, max_frequency)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_signal_spectrogram(
    signal: np.ndarray,
    sample_rate: float,
    title: str,
    output_path: str | Path | None = None,
    max_frequency: float = 2000.0,
    nperseg: int = 256,
    noverlap: int = 128,
) -> None:
    freqs, times, spec = spectrogram(
        signal,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    plt.figure(figsize=(12, 4))
    plt.pcolormesh(times, freqs, 10 * np.log10(spec + 1e-12), shading="gouraud")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.ylim(0, max_frequency)
    plt.colorbar(label="Power/Frequency (dB/Hz)")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()