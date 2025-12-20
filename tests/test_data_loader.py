import os
import sys

# ensure repo root is importable
repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest
from models.data_loader import DataLoader


def test_load_csv_and_columns():
    loader = DataLoader()
    df = loader.load_csv()
    assert df is not None, "DataLoader.load_csv returned None"
    # After normalization, the loader should expose this column
    assert 'Modal Price (Rs./Quintal)' in df.columns, "Expected modal price column missing"


def test_prepare_lstm_data_shapes():
    loader = DataLoader()
    # use sample market/crop that exists in the dataset used in dev workspace
    market = 'Bangalore'
    crop = 'Tomato'
    X, y = loader.prepare_lstm_data(market, crop, lookback=30)
    # If dataset lacks enough rows for this market/crop, the test should fail explicitly
    assert X is not None and y is not None, "prepare_lstm_data returned None (not enough data?)"
    assert X.ndim == 3 and X.shape[1] == 30 and X.shape[2] == 1
    assert y.ndim == 2 and y.shape[1] == 1
