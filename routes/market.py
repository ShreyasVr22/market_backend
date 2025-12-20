from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import numpy as np
# Import `load_model` with a safe fallback so the code still loads when
# TensorFlow is not present in the environment (helps runtime resilience).
try:
    from tensorflow.keras.models import load_model
except Exception:
    try:
        # Fallback to standalone Keras if available
        from keras.models import load_model
    except Exception:
        # If neither is available, provide a placeholder to avoid NameError at import time.
        load_model = None
from models.data_loader import DataLoader
from utils.ceda_client import CEDAClient
from utils.db_utils import Database
from models.trainer import train_and_save_all
import pandas as pd
import math

router = APIRouter()

# Initialize clients
db = Database()
ceda = CEDAClient()

# Load CSV data once at startup. Prefer `CSV_PATH` env var or a path
# passed into DataLoader. Do NOT fall back to hardcoded Downloads/Kaggle paths.
CSV_PATH = os.getenv("CSV_PATH", None)
loader = DataLoader(CSV_PATH)
# Optional path where trained models are stored (absolute or repo-relative)
MODEL_PATH = os.getenv("MODEL_PATH", None)

# Defer heavy CSV loading to application startup to avoid blocking/ crashing
def startup_load():
    try:
        loader.load_csv()
    except Exception as e:
        print(f"✗ Startup CSV load failed: {e}")

    # Build districts -> markets -> crops mapping from loaded CSV
    try:
        DISTRICTS_DATA.clear()
        if loader.data is None:
            return

        df = loader.data.copy()
        # possible district column names in various datasets
        district_cols = ['District', 'District Name', 'District_Name', 'district', 'district_name']
        district_col = None
        for c in district_cols:
            if c in df.columns:
                district_col = c
                break

        # If we don't have an explicit district column, try to infer districts from 'Market Name'
        if district_col is None:
            # map every market to a pseudo-district equal to first token of market name
            df['__inferred_district'] = df['Market Name'].astype(str).apply(lambda s: s.split('_')[0].split('-')[0].split()[:1][0] if s else 'Unknown')
            district_col = '__inferred_district'

        # Determine crop columns (collect from Variety/Commodity/Group)
        crop_cols = [c for c in ['Variety', 'Commodity', 'Group'] if c in df.columns]

        grouped = df.groupby(district_col)
        for dist, group in grouped:
            markets = sorted(group['Market Name'].astype(str).str.strip().unique().tolist()) if 'Market Name' in group.columns else []
            # collect union of available crop columns to be more inclusive
            crops_set = set()
            for cc in crop_cols:
                crops_set.update(group[cc].astype(str).str.strip().unique().tolist())
            crops = sorted([c for c in crops_set if c and c.strip()])
            DISTRICTS_DATA[str(dist)] = {
                "markets": markets,
                "crops": crops
            }
        print(f"✓ Built DISTRICTS_DATA with {len(DISTRICTS_DATA)} districts")
    except Exception as e:
        print(f"✗ Failed to build DISTRICTS_DATA: {e}")
# Cache for trained models
MODEL_CACHE = {}

# District-Market-Crop mapping (from frontend)
# Remove hardcoded/fallback district data. Use CSV data from `loader` as the source
DISTRICTS_DATA = {}

class PriceData(BaseModel):
    market: str
    crop: str
    current_price: float
    price_30_days: float
    price_60_days: float
    price_90_days: float

def load_trained_model(district, market, crop):
    """Load trained LSTM model from cache"""
    key = f"{district}_{market.replace(' ', '_')}_{crop}"
    if key not in MODEL_CACHE:
        # Try multiple filename patterns and formats (.h5, .pkl)
        # base candidate filenames (relative to model directories)
        base_candidates = [
            f"{district}_{market.replace(' ', '_')}_{crop}.h5",
            f"{crop}_{market.replace(' ', '_')}.h5",
            f"{crop}_{market.replace(' ', '_')}.pkl",
            f"{district}_{market.replace(' ', '_')}_{crop}.pkl",
            f"{crop}_{market}.pkl",
        ]

        candidates = []
        # If MODEL_PATH provided, prefer that directory first
        if MODEL_PATH:
            for b in base_candidates:
                candidates.append(os.path.join(MODEL_PATH, b))

        # then check the repo-local trained_models/ folder
        for b in base_candidates:
            candidates.append(os.path.join('trained_models', b))

        loaded = False
        for model_path in candidates:
            if not os.path.exists(model_path):
                continue
            try:
                if model_path.endswith('.h5'):
                    keras_model = load_model(model_path)
                    # try to load scaler saved alongside model
                    scaler = None
                    scaler_path = os.path.splitext(model_path)[0] + "_scaler.pkl"
                    if os.path.exists(scaler_path):
                        try:
                            import pickle
                            with open(scaler_path, 'rb') as sf:
                                scaler = pickle.load(sf)
                        except Exception as se:
                            print(f"✗ Failed to load scaler {scaler_path}: {se}")
                    MODEL_CACHE[key] = {"model": keras_model, "scaler": scaler}
                    print(f"✓ Loaded Keras LSTM model: {model_path}")
                else:
                    # try pickle for other formats - expect dict {'model':..., 'scaler':...} or model object
                    import pickle
                    with open(model_path, 'rb') as f:
                        obj = pickle.load(f)
                    if isinstance(obj, dict) and ('model' in obj or 'scaler' in obj):
                        MODEL_CACHE[key] = obj
                    else:
                        MODEL_CACHE[key] = {"model": obj, "scaler": None}
                    print(f"✓ Loaded pickled model: {model_path}")
                loaded = True
                break
            except Exception as e:
                print(f"✗ Failed to load model {model_path}: {e}")
        if not loaded:
            # nothing found or failed to load
            pass
    return MODEL_CACHE.get(key)


def has_trained_model_file(district, market, crop):
    """Fast check if a trained model file exists for the combo (no loading)."""
    # try same candidate patterns as `load_trained_model` but only check file existence
    base_candidates = [
        f"{district}_{market.replace(' ', '_')}_{crop}.h5",
        f"{crop}_{market.replace(' ', '_')}.h5",
        f"{crop}_{market.replace(' ', '_')}.pkl",
        f"{district}_{market.replace(' ', '_')}_{crop}.pkl",
        f"{crop}_{market}.pkl",
    ]

    # check MODEL_PATH first if set
    if MODEL_PATH:
        for b in base_candidates:
            model_path = os.path.join(MODEL_PATH, b)
            if os.path.exists(model_path):
                return True

    # then check repo-local trained_models/
    for b in base_candidates:
        model_path = os.path.join('trained_models', b)
        if os.path.exists(model_path):
            return True
    return False
    return False

def get_lstm_prediction(model, loader, market, crop):
    """Generate LSTM predictions"""
    try:
        # Get recent price history
        recent_df = loader.filter_by_market_crop(market, crop)
        if len(recent_df) < 10:
            return None
        # Prefer the normalized Kaggle column name used by DataLoader and fallbacks
        possible_price_cols = [
            'Modal Price (Rs./Quintal)',
            'Modal_Price',
            'Modal Price',
            'Modal_Price (Rs./Quintal)',
            'Modal_Price',
            'Price'
        ]

        price_col = None
        for c in possible_price_cols:
            if c in recent_df.columns:
                price_col = c
                break

        if price_col is None:
            # no known price column present
            return None

        recent_prices = recent_df[price_col].dropna().tail(30).values
        
        if len(recent_prices) < 30:
            return None
        
        # Prepare scaler and model object
        keras_model = None
        scaler = None
        # model might be a dict returned by load_trained_model
        if isinstance(model, dict):
            keras_model = model.get('model')
            scaler = model.get('scaler')
        else:
            keras_model = model

        # If no saved scaler, fit a local one on recent prices (less ideal)
        from sklearn.preprocessing import MinMaxScaler
        if scaler is None:
            scaler = MinMaxScaler()
            recent_scaled = scaler.fit_transform(recent_prices.reshape(-1, 1))
        else:
            # Use saved scaler to transform
            recent_scaled = scaler.transform(recent_prices.reshape(-1, 1))

        X = recent_scaled.reshape(1, 30, 1)

        if keras_model is None:
            return None

        pred_scaled = keras_model.predict(X, verbose=0)
        # pred_scaled shape may be (1,1) or (n,1)
        try:
            pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        except Exception:
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        return float(pred_price)
    except Exception as e:
        print(f"✗ LSTM prediction error: {e}")
        return None

def fallback_predict_from_prices(prices, horizon=30):
    """Simple linear-extrapolation fallback when LSTM isn't available.

    prices: 1D array-like of recent numeric prices (oldest->newest)
    horizon: days ahead to predict (default 30)
    Returns float prediction or None
    """
    try:
        import numpy as _np
        arr = _np.array(prices, dtype=_np.float64)
        if arr.size == 0:
            return None
        if arr.size == 1:
            return float(arr[-1])
        # fit linear trend (degree 1)
        x = _np.arange(arr.size)
        slope, intercept = _np.polyfit(x, arr, 1)
        # predict at index (n-1 + horizon)
        pred_index = (arr.size - 1) + horizon
        pred = intercept + slope * pred_index
        if _np.isfinite(pred):
            return float(pred)
        return None
    except Exception as e:
        print(f"✗ Fallback prediction error: {e}")
        return None

@router.get("/districts")
def get_districts():
    """Get all districts"""
    # Return available districts built at startup
    if not DISTRICTS_DATA:
        # attempt to load CSV on demand
        try:
            loader.load_csv()
        except Exception:
            pass
        # rebuild mapping
        try:
            startup_load()
        except Exception:
            pass

    districts = sorted(list(DISTRICTS_DATA.keys()))
    return {"districts": districts, "total": len(districts)}

@router.get("/district/{district}")
def get_district_markets(district: str, market: Optional[str] = None):
    """Get markets for a district"""
    # Exact lookup from prebuilt DISTRICTS_DATA
    if not DISTRICTS_DATA:
        # ensure mapping exists
        try:
            startup_load()
        except Exception:
            pass

    if district in DISTRICTS_DATA:
        data = DISTRICTS_DATA[district]
        raw_markets = data.get('markets', [])
        raw_crops = data.get('crops', [])

        # Build market -> crops mapping. Provide full crop lists to the frontend
        # so markets and crops are visible even when no trained model exists yet.
        market_crops = {}
        markets_with_models = []
        for m in raw_markets:
            crops_with_model = []
            for c in raw_crops:
                try:
                    if has_trained_model_file(district, m, c):
                        crops_with_model.append(c)
                except Exception:
                    continue
            # keep track of markets that do have at least one trained crop
            if crops_with_model:
                markets_with_models.append(m)
            # always expose the full crop list for UI visibility
            market_crops[m] = sorted(raw_crops)

        # If client requested a specific market, return crops only for that market
        if market:
            # try exact and case-insensitive match
            match_market = None
            for m in markets_with_models:
                if m == market or m.lower() == market.lower():
                    match_market = m
                    break
            if match_market is None:
                raise HTTPException(status_code=404, detail="Market not found or no trained models for this market")
            return {
                "district": district,
                "market": match_market,
                "crops": market_crops.get(match_market, []),
                "total_crops": len(market_crops.get(match_market, []))
            }

        # Return flat list of markets as objects so the frontend can render
        # per-market crops directly. Keep `market_names` and `trained_markets`
        # for backward compatibility.
        markets_data = []
        for m in sorted(raw_markets):
            markets_data.append({
                "market": m,
                "crops": market_crops.get(m, [])
            })

        return {
            "district": district,
            "markets": markets_data,
            "market_names": sorted(raw_markets),
            "trained_markets": sorted(markets_with_models),
            "total_markets": len(raw_markets)
        }

    # Fallback: try case-insensitive match
    for dist_key in DISTRICTS_DATA.keys():
        if dist_key.lower() == district.lower():
            data = DISTRICTS_DATA[dist_key]
            return {
                "district": dist_key,
                "markets": data.get('markets', []),
                "crops": data.get('crops', []),
                "total_markets": len(data.get('markets', []))
            }

    raise HTTPException(status_code=404, detail="District not found")

@router.get("/prices/{market}/{crop}")
def get_prices(market: str, crop: str):
    """Get current + predicted prices for crop in market (LSTM + CEDA)"""
    # Validate market/crop existence using loaded CSV data (no fallback hardcoded districts)
    if loader.data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    # Ensure district mapping exists (startup_load may not have completed yet)
    if not DISTRICTS_DATA:
        try:
            startup_load()
        except Exception:
            pass

    # Try to infer district for the given market so trained models saved
    # with district prefixes can be located. Many saved models are named
    # like <District>_<Market>_<Crop>.h5 so we prefer to set `district`
    # when the market is present in the prebuilt `DISTRICTS_DATA`.
    district = None
    for dist, data in DISTRICTS_DATA.items():
        markets = data.get('markets', [])
        # exact or case-insensitive match
        for m in markets:
            if m == market or m.lower() == market.lower():
                district = dist
                break
        if district:
            break

    df_check = loader.filter_by_market_crop(market, crop)
    if df_check is None or df_check.empty:
        # Handle common frontend ordering: frontend may send {district}/{crop}
        # (e.g. /prices/Bangalore/Pudi). If the first parameter matches a
        # known district and the second matches a crop in that district,
        # return the crop prices for all markets in that district.
        district_key = None
        if market in DISTRICTS_DATA:
            district_key = market
        else:
            for dk in DISTRICTS_DATA.keys():
                if dk.lower() == market.lower():
                    district_key = dk
                    break

        if district_key:
            crops_list = DISTRICTS_DATA[district_key].get('crops', [])
            # If the second param matches a crop in this district, return
            # aggregated prices for that crop across all markets in the district.
            # allow exact or substring matches for crop names (case-insensitive)
            # Try a direct scan of the loaded CSV for this district + crop
            # to be robust when crops appear in different columns (Variety/Group/Commodity).
            df = loader.data.copy()
            # determine district column if present
            district_cols = ['District', 'District Name', 'District_Name', 'district', 'district_name']
            district_col = next((c for c in district_cols if c in df.columns), None)

            # Build crop mask across possible columns
            crop_mask = False
            crop_cols = [c for c in ['Variety', 'Commodity', 'Group'] if c in df.columns]
            if crop_cols:
                masks = [df[col].astype(str).str.contains(crop, case=False, na=False) for col in crop_cols]
                from functools import reduce
                import operator
                crop_mask = reduce(operator.or_, masks)

            # district match mask
            if district_col:
                dist_mask = df[district_col].astype(str).str.lower() == district_key.lower()
            else:
                # fallback: match district name in Market Name
                dist_mask = df['Market Name'].astype(str).str.contains(district_key, case=False, na=False)

            combined = df[crop_mask & dist_mask]
            if combined is not None and not combined.empty:
                # compute latest per-market price and average them
                price_col = 'Modal Price (Rs./Quintal)'
                if price_col in combined.columns:
                    per_market = combined.groupby('Market Name')[price_col].apply(lambda s: pd.to_numeric(s, errors='coerce').dropna().iloc[-1] if not pd.to_numeric(s, errors='coerce').dropna().empty else None)
                    prices = [float(v) for v in per_market if v is not None]
                else:
                    prices = []

                # try to gather any LSTM predictions available per market
                preds = []
                markets_seen = combined['Market Name'].astype(str).unique().tolist()
                for m in markets_seen:
                    try:
                        model = load_trained_model(district_key, m, crop)
                        if model:
                            pred = get_lstm_prediction(model, loader, m, crop)
                            if pred is not None:
                                preds.append(float(pred))
                    except Exception:
                        continue

                current_agg = round(sum(prices)/len(prices), 2) if prices else None
                pred_30_agg = round(sum(preds)/len(preds), 2) if preds else None

                return {
                    "district": district_key,
                    "crop": crop,
                    "market": district_key,
                    "prices": {
                        "current": current_agg,
                        "day_30": pred_30_agg,
                        "day_60": None,
                        "day_90": None
                    },
                    "currency": "INR",
                    "unit": "per kg",
                    "source": {
                        "current": "Local CSV (aggregated)",
                        "predictions": "LSTM (aggregated)" if preds else None
                    }
                }

        # No matches found - return 404 as before
        raise HTTPException(status_code=404, detail="Market or crop not found in dataset")

    # district variable may have been inferred above
    
    # 1. Try CEDA API for current price
    current_price = None
    try:
        ceda_data = ceda.get_current_prices(crop, market)
        if ceda_data:
            current_price = float(ceda_data[0].get('price', 2500))
    except:
        pass
    
    # 2. Fallback to Kaggle data
    if current_price is None:
        current_price = loader.get_latest_price(market, crop) or 2500
    
    # 3. LSTM Predictions - require trained model; do not use demo fallback
    model = load_trained_model(district, market, crop)
    pred_30 = pred_60 = pred_90 = None

    if model:
        pred_30 = get_lstm_prediction(model, loader, market, crop)
        if pred_30 is None:
            # model could not predict due to insufficient history
            pred_30 = None
        # we currently only compute 30-day LSTM; keep 60/90 as None unless extended
        pred_60 = None
        pred_90 = None

    # If no LSTM model prediction, attempt a fallback using recent CSV prices
    if pred_30 is None:
        try:
            recent_df = loader.filter_by_market_crop(market, crop)
            if recent_df is not None and not recent_df.empty:
                # find price column
                possible_price_cols = [
                    'Modal Price (Rs./Quintal)',
                    'Modal_Price',
                    'Modal Price',
                    'Modal_Price (Rs./Quintal)',
                    'Price'
                ]
                price_col = None
                for c in possible_price_cols:
                    if c in recent_df.columns:
                        price_col = c
                        break

                if price_col is not None:
                    recent_prices = pd.to_numeric(recent_df[price_col], errors='coerce').dropna().values
                    # allow fallback with as few as 3 samples
                    if len(recent_prices) >= 1:
                        fb = fallback_predict_from_prices(recent_prices, horizon=30)
                        if fb is not None:
                            pred_30 = float(fb)
        except Exception as e:
            print(f"✗ Fallback gather error: {e}")
    
    # Save to database
    db.insert_prediction(market, crop, current_price, pred_30, pred_60, pred_90)
    
    return {
        "market": market,
        "crop": crop,
        "district": district,
        "prices": {
            "current": round(current_price, 2) if current_price is not None else None,
            "day_30": round(pred_30, 2) if pred_30 is not None else None,
            "day_60": round(pred_60, 2) if pred_60 is not None else None,
            "day_90": round(pred_90, 2) if pred_90 is not None else None
        },
        "currency": "INR",
        "unit": "per kg",
        "source": {
            "current": "CEDA API" if ceda_data else "Local CSV",
            "predictions": "LSTM" if (model and pred_30 is not None) else None
        }
    }

@router.get("/market/{market}")
def get_market_all_crops(market: str):
    """Get all crops and prices for a market"""
    
    # Find district and crops
    district = None
    crops = []
    for dist, data in DISTRICTS_DATA.items():
        if market in data["markets"]:
            district = dist
            crops = data["crops"]
            break
    
    if not district:
        raise HTTPException(status_code=404, detail="Market not found")
    
    # Get prices for all crops that have trained models for this market
    market_prices = []
    usable_crops = [c for c in crops if has_trained_model_file(district, market, c)]
    for crop in usable_crops:
        try:
            price_data = get_prices(market, crop)  # Reuse logic
            market_prices.append(price_data["prices"])
        except:
            market_prices.append({
                "crop": crop,
                "current": 2500,
                "day_30": 2600,
                "day_60": 2700,
                "day_90": 2800
            })
    
    return {
        "district": district,
        "market": market,
        "crops_count": len(crops),
        "prices": market_prices
    }

@router.get("/district-all/{district}")
def get_district_all_markets_prices(district: str):
    """Get all markets and prices for a district"""
    
    if district not in DISTRICTS_DATA:
        raise HTTPException(status_code=404, detail="District not found")
    
    data = DISTRICTS_DATA[district]
    markets_data = []
    
    for market in data["markets"]:
        crops_data = []
        for crop in data["crops"]:
            try:
                price_data = get_prices(market, crop)
                crops_data.append(price_data["prices"])
            except:
                crops_data.append({
                    "crop": crop,
                    "current": 2500,
                    "day_30": 2600,
                    "day_60": 2700,
                    "day_90": 2800
                })
        
        markets_data.append({
            "market": market,
            "crops": crops_data
        })
    
    return {
        "district": district,
        "markets": markets_data
    }


@router.get("/samples/counts")
def get_samples_counts():
    """Return sample counts for each crop across the loaded CSV.

    Counts are based on numeric entries in the normalized price column
    and grouped by crop (union of Variety/Commodity/Group).
    """
    if loader.data is None:
        try:
            loader.load_csv()
        except Exception:
            pass

    if loader.data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = loader.data.copy()
    crop_cols = [c for c in ['Variety', 'Commodity', 'Group'] if c in df.columns]
    price_col = 'Modal Price (Rs./Quintal)'
    result = {}

    if not crop_cols or price_col not in df.columns:
        raise HTTPException(status_code=400, detail='Dataset missing crop or price columns')

    for col in crop_cols:
        # count numeric price entries per crop value
        sub = df[[col, price_col]].copy()
        sub[price_col] = pd.to_numeric(sub[price_col], errors='coerce')
        sub = sub.dropna(subset=[price_col])
        grouped = sub.groupby(col)[price_col].count()
        for crop, cnt in grouped.items():
            key = str(crop)
            result[key] = result.get(key, 0) + int(cnt)

    return {"counts": result, "total_crops": len(result)}

@router.get("/health")
def health():
    return {
        "status": "Market API running",
        "csv_loaded": loader.data is not None,
        "models_loaded": len(MODEL_CACHE),
        "database": os.path.exists("market_prices.db")
    }


@router.post("/train")
def train_models_endpoint(epochs: int = None, batch_size: int = None):
    """Trigger training of missing models for all DISTRICTS_DATA entries.
    Accepts optional query params `epochs` and `batch_size`. If omitted, values are read
    from environment variables `TRAIN_EPOCHS`/`TRAIN_BATCH_SIZE` or default to 50/16.
    """
    report = train_and_save_all(loader, DISTRICTS_DATA, epochs=epochs, batch_size=batch_size)
    # Clear cache so newly trained models are picked up on next request
    MODEL_CACHE.clear()
    return report
