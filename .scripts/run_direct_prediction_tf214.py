import os
import sys
import json
import traceback

# ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from routes import market
except Exception:
    traceback.print_exc()
    raise

def main():
    try:
        print('Loading CSV and building district/market mapping...')
        market.startup_load()
        print('Calling get_prices(market="Bangalore", crop="(Whole)")')
        res = market.get_prices('Bangalore', '(Whole)')
        print('Result:')
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
