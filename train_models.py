import os
import pandas as pd
from models.data_loader import DataLoader

print("🚀 Starting Market Price LSTM Training (using local CSV)...")

# Prefer CSV_PATH environment variable; otherwise use the provided Downloads CSV
CSV_PATH = os.getenv('CSV_PATH', r"C:\Users\HP\Downloads\35985678-0d79-46b4-9ed6-6f13308a1d24_201016dc9f0197de809ef278006ca85a.csv")

if not os.path.exists(CSV_PATH):
    raise SystemExit(f"CSV not found at {CSV_PATH}. Set CSV_PATH env var to a valid file.")

df = pd.read_csv(CSV_PATH)

print(f"📊 Dataset shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

print("\n🗺️ Sample markets:")
print(df['Market Name'].unique()[:10])

print("\n🌾 Sample commodities (Group):")
print(df['Group'].value_counts().head(10))

print("\n📍 Districts in dataset:")
print(df.get('District Name', pd.Series([])).unique()[:10])

# 3. Update DISTRICTS_DATA to match available local data
CROPS_TO_TRAIN = ["Rice", "Tomato", "Potato", "Onion", "Maize", "Ragi", "Groundnut"]

print("\n🔍 Checking data availability...")
available_data = {}

for crop in CROPS_TO_TRAIN:
    if 'Group' not in df.columns:
        continue
    crop_data = df[df['Group'].astype(str).str.contains(crop, case=False, na=False)]
    if len(crop_data) > 0:
        markets = crop_data['Market Name'].unique()[:3]
        available_data[crop] = {
            "count": len(crop_data),
            "markets": list(markets)
        }
        print(f"  {crop}: {len(crop_data)} records, markets: {list(markets)}")

print("\n✓ Dataset ready for training!")
print(f"Total records: {len(df)}")
print(f"Date range: {df.get('Reported Date').min()} to {df.get('Reported Date').max()}")
