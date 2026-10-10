"""Shared pytest fixtures.

Pipeline modules import their siblings by bare name (``from ingestion import
load_data``), which works because each stage is run as a script from its own
directory. Mirror that here so tests can import them directly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """~2 years of random-walk business-day OHLCV data (no network needed)."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=500, name="Date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, len(dates)))
    volume = rng.integers(1_000_000, 5_000_000, len(dates))
    return pd.DataFrame(
        {"Close": close, "High": high, "Low": low, "Open": open_, "Volume": volume},
        index=dates,
    )
