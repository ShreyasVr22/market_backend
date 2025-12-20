import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from routes.market import startup_load, get_districts, get_district_markets, get_prices

def run():
    # Ensure CSV and mappings are loaded
    startup_load()

    d = get_districts()
    print("/districts ->")
    print(json.dumps(d, indent=2))

    districts = d.get('districts', [])
    if not districts:
        print("No districts available")
        return

    district = districts[0]
    md = get_district_markets(district)
    print(f"/district/{district} ->")
    print(json.dumps(md, indent=2))

    markets = md.get('markets', [])
    crops = md.get('crops', [])
    if not markets or not crops:
        print("No market/crop available for further price check")
        return

    market = markets[0]
    crop = crops[0]
    print(f"Calling get_prices for market={market!r}, crop={crop!r}")
    try:
        p = get_prices(market, crop)
        print(json.dumps(p, indent=2))
    except Exception as e:
        print("get_prices raised:", repr(e))

if __name__ == '__main__':
    run()
