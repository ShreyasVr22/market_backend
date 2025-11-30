import kagglehub
import pandas as pd
import os
from models.data_loader import DataLoader
from models.lstm_model import LSTMPredictor
import numpy as np

print("🚀 Starting Market Price LSTM Training...")

# 1. Download Kaggle dataset
print("📥 Downloading Kaggle dataset...")
path = kagglehub.dataset_download("vandeetshah/india-commodity-wise-mandi-dataset")
print(f"✓ Dataset path: {path}")

# Find CSV files
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files")

# Copy main CSV to data folder
if csv_files and not os.path.exists("data/india_commodity_data.csv"):
    os.makedirs("data", exist_ok=True)
    import shutil
    shutil.copy(os.path.join(path, csv_files[0]), "data/india_commodity_data.csv")
    print("✓ CSV copied to data/india_commodity_data.csv")

# 2. Load data
csv_path = "data/india_commodity_data.csv"
df = pd.read_csv(csv_path)

print(f"📊 Dataset shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

# Check available data
print("\n🗺️ Sample markets:")
print(df['Market Name'].unique()[:10])

print("\n🌾 Sample commodities (Group):")
print(df['Group'].value_counts().head(10))

print("\n📍 Districts in dataset:")
print(df['District Name'].unique()[:10])

# 3. Update DISTRICTS_DATA to match available Kaggle data
# First, see what's available for key crops
CROPS_TO_TRAIN = ["Rice", "Tomato", "Potato", "Onion", "Maize", "Ragi", "Groundnut"]

print("\n🔍 Checking data availability...")
available_data = {}

for crop in CROPS_TO_TRAIN:
    crop_data = df[df['Group'].str.contains(crop, case=False, na=False)]
    if len(crop_data) > 0:
        markets = crop_data['Market Name'].unique()[:3]
        available_data[crop] = {
            "count": len(crop_data),
            "markets": list(markets)
        }
        print(f"  {crop}: {len(crop_data)} records, markets: {list(markets)}")

print("\n✓ Dataset ready for training!")
print(f"Total records: {len(df)}")
print(f"Date range: {df['Reported Date'].min()} to {df['Reported Date'].max()}")
