import numpy as np

from bcmonitor.preprocessing import segment_signal


def test_segment_signal_shape():
    signal = np.arange(5000, dtype=float)
    windows = segment_signal(signal, window_size=1000, step_size=500)

    assert windows.ndim == 2
    assert windows.shape[1] == 1000
    assert windows.shape[0] > 0