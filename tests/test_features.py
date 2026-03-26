import numpy as np

from bcmonitor.features import extract_basic_features


def test_extract_basic_features_keys():
    signal = np.random.randn(2048)
    features = extract_basic_features(signal, sample_rate=12000.0)

    expected_keys = {
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
    }

    assert expected_keys.issubset(features.keys())