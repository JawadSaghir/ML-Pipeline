import os
import logging
import numpy as np
import pandas as pd
import colorlog
import yaml

from ingestion import load_data


log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))
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
    }
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
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        if not params:
            raise ValueError(f"Parameter file {params_path} is empty.")
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def feature_engineering(params_path: str = "configs/params.yaml") -> pd.DataFrame:
    """
    Build the engineered feature set from previously ingested OHLCV data.

    Reads the raw CSV produced by the ingestion stage (it does NOT download),
    derives stationary features and the forward-looking target, then writes the
    processed feature CSV.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    params = load_params(params_path)
    df = load_data(params["ingestion"]["raw_data_path"])

    try:
        HORIZON = 5
        feature_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

        if feature_df.isna().all().any():
            missing_columns = feature_df.columns[feature_df.isna().all()].tolist()
            raise ValueError(
                f"Non-numeric or empty input data in columns: {missing_columns}"
            )

        feature_df = feature_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if feature_df.empty:
            raise ValueError("No usable OHLCV rows remain after cleaning the input data.")

        # returns / momentum (stationary)
        feature_df['Return_1D']   = feature_df['Close'].pct_change()
        feature_df['Log_Return']  = np.log(feature_df['Close'] / feature_df['Close'].shift(1))
        feature_df['Momentum_5']  = feature_df['Close'] / feature_df['Close'].shift(5)  - 1
        feature_df['Momentum_10'] = feature_df['Close'] / feature_df['Close'].shift(10) - 1

        # price vs moving averages (ratio => stationary, unlike raw price levels)
        feature_df['Close_vs_MA5']  = feature_df['Close'] / feature_df['Close'].rolling(5).mean()  - 1
        feature_df['Close_vs_MA20'] = feature_df['Close'] / feature_df['Close'].rolling(20).mean() - 1

        # volatility of returns
        feature_df['Vol_5']  = feature_df['Return_1D'].rolling(5).std()
        feature_df['Vol_20'] = feature_df['Return_1D'].rolling(20).std()

        # intraday spreads, normalised by price
        feature_df['HL_Spread'] = (feature_df['High'] - feature_df['Low'])  / feature_df['Close']
        feature_df['OC_Spread'] = (feature_df['Open'] - feature_df['Close']) / feature_df['Close']

        # volume (relative, not raw level)
        feature_df['Volume_Change']  = feature_df['Volume'].pct_change()
        feature_df['Volume_vs_MA20'] = feature_df['Volume'] / feature_df['Volume'].rolling(20).mean() - 1

        # RSI(14)
        delta = feature_df['Close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        feature_df['RSI_14'] = 100 - 100 / (1 + gain / loss)

        # MACD (normalised by price so it is comparable across the $24->$300 era)
        ema12 = feature_df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = feature_df['Close'].ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        feature_df['MACD']      = macd / feature_df['Close']
        feature_df['MACD_Hist'] = (macd - macd.ewm(span=9, adjust=False).mean()) / feature_df['Close']

        # Bollinger %B (where price sits inside its 20d band, ~0..1)
        ma20  = feature_df['Close'].rolling(20).mean()
        std20 = feature_df['Close'].rolling(20).std()
        feature_df['BB_pctB'] = (feature_df['Close'] - (ma20 - 2 * std20)) / (4 * std20)

        # calendar
        feature_df['DayOfWeek'] = feature_df.index.dayofweek
        feature_df['Month']     = feature_df.index.month
        feature_df['Quarter']   = feature_df.index.quarter

        # TARGET: 1 if Close is higher HORIZON days ahead, else 0
        feature_df['Target'] = (feature_df['Close'].shift(-HORIZON) > feature_df['Close']).astype(int)
        feature_df.loc[feature_df.index[-HORIZON:], 'Target'] = np.nan  # future unknown

        model_df = feature_df.dropna().copy()
        if model_df.empty:
            raise ValueError("Feature engineering removed all rows; check the input coverage.")

        output_path = params["data_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model_df.to_csv(output_path)
        logger.info("Processed features saved to %s with shape %s", output_path, model_df.shape)
        logger.info("Feature engineering completed successfully.")
        return model_df
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

if __name__ == "__main__":
    try:
        feature_engineering()
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise