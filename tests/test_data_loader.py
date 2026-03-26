from bcmonitor.data_loader import describe_signal, load_raw_sample


def test_load_normal_file():
    signal = load_raw_sample("normal_0.mat", sample_rate=12000.0)
    summary = describe_signal(signal)

    assert signal.signal.ndim == 1
    assert signal.signal.size > 0
    assert summary["file_name"] == "normal_0.mat"