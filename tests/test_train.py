import pandas as pd

from bcmonitor.train import add_load_id_column, split_by_load


def test_add_load_id_column():
    df = pd.DataFrame({
        "source_file": ["normal_0.mat", "b007_3.mat"],
        "label": ["normal", "ball"],
        "mean": [0.1, 0.2],
    })

    out = add_load_id_column(df)

    assert "load_id" in out.columns
    assert out["load_id"].tolist() == [0, 3]


def test_split_by_load():
    df = pd.DataFrame({
        "source_file": ["normal_0.mat", "normal_3.mat"],
        "label": ["normal", "normal"],
        "mean": [0.1, 0.2],
        "std": [1.0, 1.1],
        "rms": [1.0, 1.1],
        "peak_to_peak": [2.0, 2.1],
        "crest_factor": [3.0, 3.1],
        "shape_factor": [1.1, 1.2],
        "impulse_factor": [4.0, 4.1],
        "clearance_factor": [5.0, 5.1],
        "skewness": [0.0, 0.1],
        "kurtosis": [3.0, 3.2],
        "dominant_frequency": [100.0, 120.0],
    })

    train_df, test_df, X_train, X_test, y_train, y_test = split_by_load(
        df=df,
        train_loads=[0],
        test_loads=[3],
    )

    assert len(train_df) == 1
    assert len(test_df) == 1
    assert X_train.shape[1] == 11
    assert y_test.iloc[0] == "normal"