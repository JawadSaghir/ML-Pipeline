from __future__ import annotations

import logging
import os
from pathlib import Path

import colorlog
import pandas as pd
import yaml

from ingestion import load_data
from validation import validate_feature_frame


log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "preprocessing.log"))
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


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        if not params:
            raise ValueError(f"Parameter file {params_path} is empty.")
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as error:
        logger.error("YAML error: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error: %s", error)
        raise


def _write_frame(frame: pd.DataFrame, file_path: str) -> None:
    output_file = Path(file_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index_label="Date")


def preprocessing(params_path: str = "configs/params.yaml") -> dict[str, pd.DataFrame]:
    """
    Split the engineered feature table into chronological train/test sets.
    """
    params = load_params(params_path)
    data_path = params["data_path"]
    features = params["features"]
    label = params["label"]
    splits = params["splits"]
    test_size = float(params.get("test_size", 0.2))

    feature_df = load_data(data_path)
    validate_feature_frame(feature_df, features=features, label=label)

    missing_columns = [column for column in features + [label] if column not in feature_df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in {data_path}: {missing_columns}")

    model_df = feature_df[features + [label]].dropna().copy()
    if model_df.empty:
        raise ValueError("No rows remain after dropping missing feature values.")

    split_index = int(len(model_df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(model_df):
        raise ValueError(
            f"Invalid test_size={test_size}. It leaves no training or test rows."
        )

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label].astype(int)
    y_test = test_df[label].astype(int)

    _write_frame(X_train, splits["X_train"])
    _write_frame(X_test, splits["X_test"])
    _write_frame(y_train.to_frame(name=label), splits["y_train"])
    _write_frame(y_test.to_frame(name=label), splits["y_test"])

    if isinstance(model_df.index, pd.DatetimeIndex):
        logger.info(
            "train: %d rows (%s -> %s)",
            len(train_df),
            train_df.index.min().date(),
            train_df.index.max().date(),
        )
        logger.info(
            "test : %d rows (%s -> %s)",
            len(test_df),
            test_df.index.min().date(),
            test_df.index.max().date(),
        )
    else:
        logger.info("train: %d rows", len(train_df))
        logger.info("test : %d rows", len(test_df))

    if not y_test.empty:
        up_rate = y_test.mean() * 100
        down_rate = 100 - up_rate
        logger.info("target balance (test): up=%.1f%%  down=%.1f%%", up_rate, down_rate)

    logger.info("Split files written to %s", Path(splits["X_train"]).parent)
    logger.info("Preprocessing completed successfully.")
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def main() -> None:
    try:
        preprocessing()
    except Exception as error:
        logger.error("Preprocessing failed: %s", error)
        raise


if __name__ == "__main__":
    main()
