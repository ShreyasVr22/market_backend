import os
import sys
import json
import traceback

# ensure repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from routes import market as market_module
except Exception:
    traceback.print_exc()
    raise

def main():
    try:
        print('Ensuring CSV and mappings loaded...')
        market_module.startup_load()
        district = 'Bangalore'
        market = 'Bangalore'
        crop = '(Whole)'
        print(f"Loading trained model for {district}/{market}/{crop}...")
        model_obj = market_module.load_trained_model(district, market, crop)
        if not model_obj:
            print('No model found or failed to load')
            return
        print('Model loaded. Running LSTM prediction...')
        pred = market_module.get_lstm_prediction(model_obj, market_module.loader, market, crop)
        print('Prediction result:', pred)
        out = {'district': district, 'market': market, 'crop': crop, 'prediction': pred}
        print(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
