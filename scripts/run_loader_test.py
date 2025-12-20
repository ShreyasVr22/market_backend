import os
import sys

# Ensure repo root is on sys.path so `models` package can be imported
repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.data_loader import DataLoader

# Resolve CSV path relative to repo root
print('Runner: using CSV_PATH from .env if present, otherwise default data file')

loader = DataLoader()  # DataLoader will prefer CSV_PATH from .env
df = loader.load_csv()
if df is None:
    print('Failed to load CSV')
else:
    print('Rows:', len(df))
    print('Columns:', list(df.columns))
    print('\nSample rows:')
    print(df.head(3).to_string(index=False))

    # Try filter and latest price
    market='Bangalore'
    crop='Tomato'
    print('\nFiltering for market and crop sample')
    filtered = loader.filter_by_market_crop(market, crop)
    if filtered is None or filtered.empty:
        print('No filtered data found for market/crop')
    else:
        cols = [c for c in ['Market Name','Variety','Modal Price (Rs./Quintal)','Reported Date'] if c in filtered.columns]
        print('Filtered rows:', len(filtered))
        print(filtered[cols].head(3).to_string(index=False))

    # Try prepare LSTM data (small example)
    print('\nPreparing LSTM data (lookback=30)')
    X, y = loader.prepare_lstm_data(market, crop, lookback=30)
    if X is None:
        print('Not enough data to prepare LSTM sequences or price column missing')
    else:
        print('X shape:', X.shape)
        print('y shape:', y.shape)
        print('Sample y (first 5):', y[:5].flatten())
    df = loader.load_csv()
    if df is None:
        print('Failed to load CSV')
    else:
        print('Rows:', len(df))
        print('Columns:', list(df.columns))
        print('\nSample rows:')
        print(df.head(3).to_string(index=False))

    # Try filter and latest price
    market='Bangalore Central'
    crop='Tomato'
    print('\nFiltering for market and crop sample')
    filtered = loader.filter_by_market_crop(market, crop)
    if filtered is None:
        print('No filtered data found for market/crop')
    else:
        cols = [c for c in ['Market Name','Variety','Modal Price (Rs./Quintal)','Reported Date'] if c in filtered.columns]
        print('Filtered rows:', len(filtered))
        print(filtered[cols].head(3).to_string(index=False))
