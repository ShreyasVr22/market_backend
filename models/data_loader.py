import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv


class DataLoader:
    def __init__(self, csv_path=None):
        """csv_path: optional path passed by caller. The loader will prefer the
        `CSV_PATH` environment variable (loaded from `.env`) if present, then
        the provided `csv_path`, and finally `data/india_commodity_data.csv`.
        """
        self.csv_path = csv_path
        self.data = None
        self.scaler = MinMaxScaler()

    def load_csv(self):
        """Load Kaggle per-commodity dataset"""
        try:
            # Load environment variables from .env (if present)
            try:
                load_dotenv()
            except Exception:
                pass

            # Prefer CSV_PATH env var if it exists and points to a file
            env_path = os.getenv('CSV_PATH')
            chosen_path = None
            # Only allow explicitly provided CSV path or environment CSV_PATH.
            # Remove any hardcoded Kaggle/Downloads fallback to avoid accidental
            # loading of unexpected datasets.
            if env_path and os.path.exists(env_path):
                chosen_path = env_path
            elif self.csv_path and os.path.exists(self.csv_path):
                chosen_path = self.csv_path

            if not chosen_path:
                print("✗ No valid CSV path found. Set CSV_PATH environment variable or pass a valid path to DataLoader(csv_path=...).")
                return None

            self.data = pd.read_csv(chosen_path)

            # Normalize column names to a consistent set used by the loader
            self._normalize_columns()

            print(f"✓ CSV loaded: {len(self.data)} records from {chosen_path}")
            return self.data
        except Exception as e:
            print(f"✗ Error loading CSV: {e}")
            return None

    def filter_by_market_crop(self, market: str, crop: str):
        """
        Filter data for specific market and crop.
        NOTE: In this Kaggle dataset, 'Variety' and 'Group' describe the commodity.
        Many files are per-commodity, so filtering by 'crop' may not change much.
        """
        if self.data is None:
            self.load_csv()
        if self.data is None:
            return None

        df = self.data.copy()

        # Filter by market name (partial match)
        if market:
            if 'Market Name' not in df.columns:
                return None
            df = df[df['Market Name'].astype(str).str.contains(market, case=False, na=False)]

        # Optionally filter by crop name in Variety/Group/Commodity
        if crop:
            if 'Variety' in df.columns or 'Commodity' in df.columns:
                left = df['Variety'].astype(str) if 'Variety' in df.columns else df['Commodity'].astype(str)
                if 'Group' in df.columns:
                    right = df['Group'].astype(str)
                else:
                    right = pd.Series([''] * len(df), index=df.index)
                df = df[left.str.contains(crop, case=False, na=False) | right.str.contains(crop, case=False, na=False)]

        if df.empty:
            return None

        # Use Reported Date as time index
        if 'Reported Date' in df.columns:
            df['Reported Date'] = pd.to_datetime(df['Reported Date'], errors='coerce')
            df = df.dropna(subset=['Reported Date'])
            df = df.sort_values('Reported Date')

        return df

    def prepare_lstm_data(self, market: str, crop: str, lookback=30):
        """Prepare data for LSTM training based on Modal Price."""
        df = self.filter_by_market_crop(market, crop)
        if df is None:
            return None, None

        price_col = 'Modal Price (Rs./Quintal)'
        if price_col not in df.columns:
            return None, None

        # Coerce to numeric and drop missing values to avoid object-dtype arrays
        prices_series = pd.to_numeric(df[price_col], errors='coerce').dropna()
        if len(prices_series) < lookback:
            return None, None

        prices = prices_series.values.reshape(-1, 1).astype('float32')
        try:
            prices_scaled = self.scaler.fit_transform(prices)
        except Exception:
            return None, None

        X, y = [], []
        for i in range(len(prices_scaled) - lookback):
            X.append(prices_scaled[i:i+lookback])
            y.append(prices_scaled[i+lookback])

        return np.array(X), np.array(y)

    def get_latest_price(self, market: str, crop: str):
        """Get latest available modal price."""
        df = self.filter_by_market_crop(market, crop)
        if df is None or df.empty:
            return None

        price_col = 'Modal Price (Rs./Quintal)'
        if price_col not in df.columns:
            return None

        try:
            # Coerce to numeric and take last valid value
            series = pd.to_numeric(df[price_col], errors='coerce').dropna()
            if series.empty:
                return None
            return float(series.iloc[-1])
        except Exception:
            return None

    def _normalize_columns(self):
        """Normalize common column name variants to the loader's expected names.

        This maps variations found in the Kaggle dataset to the expected names:
        - 'Market' -> 'Market Name'
        - 'Arrival_Date' or 'Arrival Date' or 'Reported Date' -> 'Reported Date'
        - 'Modal_Price' or 'Modal Price' or 'Modal_Price (Rs./Quintal)' -> 'Modal Price (Rs./Quintal)'
        - 'Commodity' -> 'Variety' (used for crop filtering)
        It also coerces price columns to numeric (removing commas) and trims strings.
        """
        if self.data is None:
            return

        df = self.data
        # trim whitespace and normalize case for column names (keep original case mapping)
        cols = {c: c.strip() for c in df.columns}
        df.rename(columns=cols, inplace=True)

        colmap = {}
        # Map any market-like column to 'Market Name' (case-insensitive)
        if 'Market Name' not in df.columns:
            market_like = next((c for c in df.columns if 'market' in c.lower()), None)
            if market_like and market_like != 'Market Name':
                colmap[market_like] = 'Market Name'

        # Date variants
        for dname in ['Arrival_Date', 'Arrival Date', 'Reported Date', 'ArrivalDate', 'Arrival_Date ']:
            if dname in df.columns and 'Reported Date' not in df.columns:
                colmap[dname] = 'Reported Date'

        # Price variants
        for pname in ['Modal_Price', 'Modal Price', 'Modal_Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)']:
            if pname in df.columns and 'Modal Price (Rs./Quintal)' not in df.columns:
                colmap[pname] = 'Modal Price (Rs./Quintal)'

        # Commodity name -> Variety
        if 'Commodity' in df.columns and 'Variety' not in df.columns:
            colmap['Commodity'] = 'Variety'

        if colmap:
            df.rename(columns=colmap, inplace=True)

        # Normalize string columns: strip whitespace
        for c in df.select_dtypes(include=['object']).columns:
            df[c] = df[c].astype(str).str.strip()

        # Coerce modal price to numeric (remove commas/currency and convert to float)
        if 'Modal Price (Rs./Quintal)' in df.columns:
            df['Modal Price (Rs./Quintal)'] = (
                df['Modal Price (Rs./Quintal)']
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('Rs.', '', regex=False)
                .str.replace('₹', '', regex=False)
            )
            df['Modal Price (Rs./Quintal)'] = pd.to_numeric(df['Modal Price (Rs./Quintal)'], errors='coerce')

        # Assign back
        self.data = df
