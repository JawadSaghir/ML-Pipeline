import numpy as np
import pandas as pd
import pytest

from validation import validate_feature_frame, validate_ohlcv_data


def test_valid_ohlcv_passes(synthetic_ohlcv):
    validate_ohlcv_data(synthetic_ohlcv)


def test_empty_ohlcv_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_ohlcv_data(pd.DataFrame())


def test_missing_column_rejected(synthetic_ohlcv):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_ohlcv_data(synthetic_ohlcv.drop(columns="Volume"))


def test_non_datetime_index_rejected(synthetic_ohlcv):
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_ohlcv_data(synthetic_ohlcv.reset_index(drop=True))


def test_duplicate_dates_rejected(synthetic_ohlcv):
    duplicated = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[:1]])
    with pytest.raises(ValueError, match="Duplicate dates"):
        validate_ohlcv_data(duplicated)


def test_non_positive_price_rejected(synthetic_ohlcv):
    bad = synthetic_ohlcv.copy()
    bad.iloc[0, bad.columns.get_loc("Close")] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_ohlcv_data(bad)


def test_negative_volume_rejected(synthetic_ohlcv):
    bad = synthetic_ohlcv.copy()
    bad.iloc[0, bad.columns.get_loc("Volume")] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_ohlcv_data(bad)


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"f1": [0.1, 0.2, 0.3], "f2": [1.0, 2.0, 3.0], "Target": [0, 1, 0]},
        index=pd.date_range("2024-01-01", periods=3),
    )


def test_valid_feature_frame_passes(feature_frame):
    validate_feature_frame(feature_frame, ["f1", "f2"], "Target")


def test_non_binary_target_rejected(feature_frame):
    bad = feature_frame.assign(Target=[0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        validate_feature_frame(bad, ["f1", "f2"], "Target")


def test_infinite_feature_rejected(feature_frame):
    bad = feature_frame.assign(f1=[0.1, np.inf, 0.3])
    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_frame(bad, ["f1", "f2"], "Target")


def test_missing_feature_rejected(feature_frame):
    bad = feature_frame.assign(f1=[0.1, np.nan, 0.3])
    with pytest.raises(ValueError, match="Missing or non-numeric"):
        validate_feature_frame(bad, ["f1", "f2"], "Target")
