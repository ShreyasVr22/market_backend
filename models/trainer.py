import os
import pickle
import os
import pandas as pd
from models.lstm_model import LSTMPredictor
from models.data_loader import DataLoader


def train_and_save_all(loader: DataLoader, districts_data: dict, epochs=None, batch_size=None, out_dir='trained_models'):
    """
    Train LSTM models for all district/market/crop combinations.
    Defaults: read from env TRAIN_EPOCHS/TRAIN_BATCH_SIZE or use higher-quality defaults (50 epochs, batch_size 16).
    """
    # allow env override
    try:
        env_epochs = int(os.getenv('TRAIN_EPOCHS')) if os.getenv('TRAIN_EPOCHS') else None
    except Exception:
        env_epochs = None
    try:
        env_batch = int(os.getenv('TRAIN_BATCH_SIZE')) if os.getenv('TRAIN_BATCH_SIZE') else None
    except Exception:
        env_batch = None

    if epochs is None:
        epochs = env_epochs if env_epochs is not None else 50
    if batch_size is None:
        batch_size = env_batch if env_batch is not None else 16
    os.makedirs(out_dir, exist_ok=True)
    report = {'trained': [], 'skipped': [], 'failed': []}

    # minimum numeric samples required to attempt training (can be overridden via env)
    try:
        min_samples = int(os.getenv('TRAIN_MIN_SAMPLES')) if os.getenv('TRAIN_MIN_SAMPLES') else 60
    except Exception:
        min_samples = 60

    for district, info in districts_data.items():
        markets = info.get('markets', [])
        crops = info.get('crops', [])
        for market in markets:
            for crop in crops:
                # Build canonical filename
                safe_market = market.replace(' ', '_')
                safe_crop = crop.replace(' ', '_')
                base = os.path.join(out_dir, f"{district}_{safe_market}_{safe_crop}")
                h5_path = base + '.h5'
                scaler_path = base + '_scaler.pkl'

                # Skip if already present
                if os.path.exists(h5_path) and os.path.exists(scaler_path):
                    report['skipped'].append({'district': district, 'market': market, 'crop': crop, 'reason': 'exists'})
                    continue

                try:
                    # Quick pre-check: ensure enough numeric price samples exist before preparing sequences
                    df_check = loader.filter_by_market_crop(market, crop)
                    if df_check is None or df_check.empty:
                        report['skipped'].append({'district': district, 'market': market, 'crop': crop, 'reason': 'no_data'})
                        continue

                    price_col = 'Modal Price (Rs./Quintal)'
                    if price_col not in df_check.columns:
                        report['skipped'].append({'district': district, 'market': market, 'crop': crop, 'reason': 'no_price_column'})
                        continue

                    numeric_prices = pd.to_numeric(df_check[price_col], errors='coerce').dropna()
                    if len(numeric_prices) < min_samples:
                        report['skipped'].append({'district': district, 'market': market, 'crop': crop, 'reason': 'insufficient_numeric_samples', 'count': int(len(numeric_prices))})
                        continue

                    # Prepare data (this will coerce/drop NaNs internally)
                    X, y = loader.prepare_lstm_data(market, crop)
                    if X is None or y is None or len(X) < 1:
                        report['skipped'].append({'district': district, 'market': market, 'crop': crop, 'reason': 'insufficient_data_after_prepare'})
                        continue

                    # Train
                    predictor = LSTMPredictor(lookback=X.shape[1])
                    predictor.build_model(input_shape=(X.shape[1], X.shape[2]))
                    predictor.train(X, y, epochs=epochs, batch_size=batch_size)

                    # Save keras model
                    predictor.save_model(h5_path)

                    # Save scaler from loader (DataLoader.scaler used in prepare_lstm_data)
                    try:
                        scaler = loader.scaler
                        with open(scaler_path, 'wb') as sf:
                            pickle.dump(scaler, sf)
                    except Exception as e:
                        print(f"✗ Failed to save scaler for {market}-{crop}: {e}")

                    report['trained'].append({'district': district, 'market': market, 'crop': crop, 'model': h5_path})
                except Exception as e:
                    report['failed'].append({'district': district, 'market': market, 'crop': crop, 'error': str(e)})
    return report
