import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.scaler = MinMaxScaler()
    
    def load_csv(self):
        """Load Kaggle per-commodity dataset"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✓ CSV loaded: {len(self.data)} records from {self.csv_path}")
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
            df = df[df['Market Name'].str.contains(market, case=False, na=False)]

        # Optionally filter by crop name in Variety/Group
        if crop:
            if 'Variety' in df.columns:
                df = df[df['Variety'].astype(str).str.contains(crop, case=False, na=False) |
                        df['Group'].astype(str).str.contains(crop, case=False, na=False)]
        
        if df.empty:
            return None

        # Use Reported Date as time index
        if 'Reported Date' in df.columns:
            df['Reported Date'] = pd.to_datetime(df['Reported Date'])
            df = df.sort_values('Reported Date')

        return df
    
    def prepare_lstm_data(self, market: str, crop: str, lookback=30):
        """Prepare data for LSTM training based on Modal Price."""
        df = self.filter_by_market_crop(market, crop)
        if df is None or len(df) < lookback:
            return None, None
        
        price_col = 'Modal Price (Rs./Quintal)'
        if price_col not in df.columns:
            return None, None
        
        prices = df[price_col].values.reshape(-1, 1)
        prices_scaled = self.scaler.fit_transform(prices)
        
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
        
        return float(df[price_col].iloc[-1])
