import os
import sys
import json

# ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.data_loader import DataLoader


import os

def main(csv_path=None):
    if csv_path is None:
        csv_path = os.getenv('CSV_PATH', r"C:\Users\HP\Downloads\35985678-0d79-46b4-9ed6-6f13308a1d24_201016dc9f0197de809ef278006ca85a.csv")
    loader = DataLoader(csv_path)
    loader.load_csv()

    if loader.data is None:
        print(json.dumps({"error": "No CSV loaded"}, indent=2))
        return

    df = loader.data

    # Determine which columns are available
    has_district = 'District' in df.columns
    has_market = 'Market Name' in df.columns
    has_variety = 'Variety' in df.columns
    has_commodity = 'Commodity' in df.columns

    # Build sets
    all_markets = set()
    all_districts = set()
    all_crops = set()

    # For per-market crops
    market_to_crops = {}
    # For district -> markets mapping
    district_to_markets = {}

    for _, row in df.iterrows():
        market = str(row['Market Name']).strip() if has_market else None
        district = str(row['District']).strip() if has_district else None

        # Determine crop string preference: Variety then Commodity
        crop = None
        if has_variety and row.get('Variety') and str(row.get('Variety')).strip():
            crop = str(row.get('Variety')).strip()
        elif has_commodity and row.get('Commodity') and str(row.get('Commodity')).strip():
            crop = str(row.get('Commodity')).strip()

        if market and market != 'nan' and market != 'None':
            all_markets.add(market)
            if crop:
                all_crops.add(crop)
                market_to_crops.setdefault(market, set()).add(crop)

            if district and district != 'nan' and district != 'None':
                all_districts.add(district)
                district_to_markets.setdefault(district, set()).add(market)

    # Convert sets to sorted lists
    result = {
        'all_markets': sorted(list(all_markets)),
        'all_districts': sorted(list(all_districts)),
        'all_crops': sorted(list(all_crops)),
        'market_to_crops': {m: sorted(list(cs)) for m, cs in market_to_crops.items()},
        'district_to_markets': {d: sorted(list(ms)) for d, ms in district_to_markets.items()},
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
