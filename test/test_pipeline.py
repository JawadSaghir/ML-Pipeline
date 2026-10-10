"""End-to-end run of every pipeline stage on synthetic data.

Each stage is executed as a script (exactly like ``dvc repro`` does) inside a
temporary working directory, so the real ``data/`` and ``models_artifacts/``
are never touched and no network access is needed: ingestion reuses an
existing raw CSV instead of downloading from Yahoo Finance.
"""

import pickle
import shutil
import subprocess
import sys

import pandas as pd

from conftest import REPO_ROOT

STAGES = [
    "src/data/ingestion.py",
    "src/data/feature_engineering.py",
    "src/data/preprocessing.py",
    "src/models/model_building.py",
]


def test_full_pipeline_runs_on_synthetic_data(tmp_path, synthetic_ohlcv):
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    (tmp_path / "data").mkdir()
    synthetic_ohlcv.to_csv(tmp_path / "data" / "AAPL_data.csv", index_label="Date")

    for stage in STAGES:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / stage)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{stage} failed:\n{result.stderr}"

    x_train = pd.read_csv(tmp_path / "data/processed/X_train.csv", index_col=0)
    x_test = pd.read_csv(tmp_path / "data/processed/X_test.csv", index_col=0)
    assert len(x_train) > len(x_test) > 0
    # Chronological split: every test row is later than every training row.
    assert pd.to_datetime(x_train.index).max() < pd.to_datetime(x_test.index).min()

    # Safe to unpickle: this file was written moments ago by the stage above.
    with open(tmp_path / "models_artifacts/random_forest.pkl", "rb") as file:
        model = pickle.load(file)
    predictions = model.predict(x_test)
    assert len(predictions) == len(x_test)
    assert set(predictions) <= {0, 1}
