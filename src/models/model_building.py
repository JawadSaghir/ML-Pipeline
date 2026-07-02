from pathlib import Path
import logging
import pickle
import pandas as pd
import numpy as np
import os
import colorlog
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))
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


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a processed dataset from a CSV file.

    The CSV is expected to use the trading date as its index (as written by the
    ingestion/feature-engineering stages), so the date index is restored on read.

    :param file_path: Path to the CSV file
    :return: Loaded DataFrame with a parsed DatetimeIndex
    """
    try:
        df = pd.read_csv(file_path, index_col=0)
        if df.empty:
            raise ValueError(f"No rows found in {file_path}.")
        df = df.drop(index=[marker for marker in ("Ticker", "Date") if marker in df.index], errors="ignore")
        df.index = pd.to_datetime(df.index, errors="raise")
        df.index.name = "Date"
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def load_target(file_path: str) -> pd.Series:
    """Load a single-column target CSV as a Series with its date index restored."""
    df = load_data(file_path)
    return df.iloc[:, 0]


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_params: dict) -> RandomForestClassifier:
    """
    Train a RandomForest classifier using the provided training data and parameters.

    :param X_train: Training features
    :param y_train: Training labels
    :param model_params: Dictionary of model parameters
    :return: Trained model object
    """
    try:
        logger.debug('Training on %d rows and %d features', X_train.shape[0], X_train.shape[1])
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        logger.info('Model trained successfully with parameters: %s', model_params)
        return model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained model on the test set and log the metrics.

    :param model: Trained model object
    :param X_test: Test features
    :param y_test: Test labels
    :return: Dictionary of evaluation metrics
    """
    try:
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        logger.info(
            'Evaluation -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f',
            metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'],
        )
        return metrics
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_model(model: object, model_path: str) -> None:
    """
    Save the trained model to a specified path.

    :param model: Trained model object
    :param model_path: Path to save the model
    """
    try:
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info('Model saved successfully at %s', model_path)
    except Exception as e:
        logger.error('Error saving the model: %s', e)
        raise


def main():
    try:
        logger.info('Starting model-building stage.')
        params = load_params('configs/params.yaml')
        splits = params['splits']

        X_train = load_data(splits['X_train'])
        X_test = load_data(splits['X_test'])
        y_train = load_target(splits['y_train'])
        y_test = load_target(splits['y_test'])
        logger.info('Loaded splits: %d train rows, %d test rows', len(X_train), len(X_test))

        model = train_model(X_train, y_train, params['model_params'])
        evaluate_model(model, X_test, y_test)
        save_model(model, params['model_save_path'])
        logger.info("Model is successfully saved")
        logger.info('Model-building stage completed successfully.')
    except Exception as e:
        logger.error('Model-building stage failed: %s', e)
        raise


if __name__ == "__main__":
    main()
