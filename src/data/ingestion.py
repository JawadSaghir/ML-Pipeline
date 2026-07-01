from __future__ import annotations

import logging
import os
from pathlib import Path

import colorlog
import pandas as pd
import yaml

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency in some environments
    yf = None

from validation import validate_ohlcv_data


log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "ingestion.log"))
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


DEFAULT_TICKER = "AAPL"
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-01-01"
DEFAULT_OUTPUT_PATH = "data/AAPL_data.csv"


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


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file using the first column as a DatetimeIndex."""
    try:
        df = pd.read_csv(file_path, index_col=0)
        if df.empty:
            raise ValueError(f"No rows found in {file_path}.")
        df = df.drop(index=[marker for marker in ("Ticker", "Date") if marker in df.index], errors="ignore")
        df.index = pd.to_datetime(df.index, errors="raise")
        df.index.name = "Date"
        logger.debug("Data loaded from %s with shape %s", file_path, df.shape)
        return df
    except pd.errors.ParserError as error:
        logger.error("Failed to parse the CSV file: %s", error)
        raise
    except FileNotFoundError as error:
        logger.error("File not found: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error occurred while loading the data: %s", error)
        raise


def _download_ohlcv(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError(
            "yfinance is required to download data, but it is not installed."
        )

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = ["Close", "High", "Low", "Open", "Volume"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns from downloaded data: {missing_columns}")

    data = data[required_columns].dropna().sort_index()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data


def ingestion(params_path: str = "configs/params.yaml") -> pd.DataFrame:
    """
    Download the AAPL OHLCV dataset or reuse an existing local copy.

    The raw CSV is written to ``data/AAPL_data.csv`` so later stages can read it
    without reaching back out to Yahoo Finance.
    """
    params = load_params(params_path)
    ingestion_params = params.get("ingestion", {})

    ticker = ingestion_params.get("ticker", DEFAULT_TICKER)
    start = ingestion_params.get("start", DEFAULT_START)
    end = ingestion_params.get("end", DEFAULT_END)
    output_path = ingestion_params.get("raw_data_path", DEFAULT_OUTPUT_PATH)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and output_file.stat().st_size > 0:
        logger.info("Raw data already exists at %s", output_path)
        data = load_data(output_path)
        validate_ohlcv_data(data)
        logger.info("Ingestion stage completed successfully.")
        return data

    data = _download_ohlcv(ticker=ticker, start=start, end=end)
    validate_ohlcv_data(data)
    data.to_csv(output_path, index_label="Date")

    logger.info("Downloaded %d rows for %s from Yahoo Finance", len(data), ticker)
    logger.info("Raw data saved to %s", output_path)
    logger.info("Ingestion stage completed successfully.")
    return load_data(output_path)


def main() -> None:
    try:
        ingestion()
    except Exception as error:
        logger.error("Ingestion stage failed: %s", error)
        raise


if __name__ == "__main__":
    main()
