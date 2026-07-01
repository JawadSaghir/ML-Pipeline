from __future__ import annotations

import logging
import os
from typing import Iterable

import colorlog
import numpy as np
import pandas as pd


log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "validation.log"))
file_handler.setLevel(logging.DEBUG)

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s]-[%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def validate_ohlcv_data(
    data: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
) -> None:
    """Validate the raw OHLCV dataset used by the pipeline."""
    required_columns = list(required_columns or ["Close", "High", "Low", "Open", "Volume"])

    if data.empty:
        raise ValueError("Input data is empty.")

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex on the raw data.")

    if data.index.duplicated().any():
        raise ValueError("Duplicate dates found in the raw data.")

    numeric_frame = data[required_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        invalid_columns = numeric_frame.columns[numeric_frame.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values found in: {invalid_columns}")

    price_columns = [column for column in ["Open", "High", "Low", "Close"] if column in data.columns]
    if (numeric_frame[price_columns] <= 0).any().any():
        raise ValueError("Price columns must contain strictly positive values.")

    if (numeric_frame["Volume"] < 0).any():
        raise ValueError("Volume must be non-negative.")

    logger.debug("Raw OHLCV data validated successfully with shape %s", data.shape)


def validate_feature_frame(
    data: pd.DataFrame,
    features: Iterable[str],
    label: str,
) -> None:
    """Validate the engineered feature frame before splitting."""
    required_columns = list(features) + [label]

    if data.empty:
        raise ValueError("Feature frame is empty.")

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.index.duplicated().any():
        raise ValueError("Duplicate dates found in the feature frame.")

    numeric_frame = data[required_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        invalid_columns = numeric_frame.columns[numeric_frame.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values found in: {invalid_columns}")

    if not set(numeric_frame[label].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column must contain binary values 0 and 1.")

    finite_mask = np.isfinite(numeric_frame.to_numpy())
    if not finite_mask.all():
        raise ValueError("Feature frame contains non-finite values.")

    logger.debug("Feature frame validated successfully with shape %s", data.shape)


def main() -> None:
    logger.info("Validation helpers are available for import from this module.")


if __name__ == "__main__":
    main()
